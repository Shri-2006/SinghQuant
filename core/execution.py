"""
Broker execution adapter with fill reconciliation and strategy ownership.

This is the ONLY module that should submit or close orders. It guarantees:

  * a strategy can only sell what its own ledger says it owns, and never more
    than the shared account holds after every OTHER strategy's claim is
    honoured (no shorting, no liquidating another strategy's inventory),
                                                       -> audit C-02, S-03
  * a new order is refused while this strategy already has an open order for
    the symbol (no duplicate exposure from repeated signals or restarts),
                                                       -> audit H-06
  * every order carries a client_order_id written to the database BEFORE it
    is sent, so an order whose broker id was never recorded (crash or network
    timeout after the broker accepted it) is recovered by client id instead
    of being forgotten,                                -> second pass S-01
  * fills, partial fills and rejections are read back from the broker and
    applied to the ledger in the same transaction that records them on the
    order row,                                         -> audit H-03/H-06, S-05
  * every quantity is a share/coin quantity; dollars are converted exactly
    once, in the strategy's sizing step.

The `api` argument is an alpaca_trade_api.REST-like object; tests use
tests/fakes.py::FakeAlpaca.
"""
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core import portfolio
from core.config import ORDER_FILL_TIMEOUT_SECONDS
from core.logger import log_trade

TERMINAL_STATUSES = {"filled", "canceled", "cancelled", "expired", "rejected", "done_for_day",
                     "replaced", "stopped", "suspended", "unknown_at_broker", "reconcile_error"}
MIN_SELL_NOTIONAL = 1.0          # Alpaca will not accept a fractional order below $1
UNRESOLVED_ORDER_MAX_AGE_HOURS = 48   # an order we cannot refresh for this long stops blocking


def to_broker_symbol(symbol):
    """Polygon crypto tickers are 'X:BTCUSD'; Alpaca stores 'BTCUSD'."""
    return symbol[2:] if symbol.startswith("X:") else symbol


@dataclass
class OrderResult:
    strategy: str
    symbol: str
    side: str
    requested_qty: float
    status: str
    order_id: Optional[str] = None
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    message: str = ""
    extras: dict = field(default_factory=dict)

    @property
    def filled(self):
        return self.filled_qty > 0


def _attr(obj, name, default=None):
    v = getattr(obj, name, None)
    if v is None and hasattr(obj, "_raw"):
        v = obj._raw.get(name, default)
    return default if v is None else v


def _order_fill_state(order):
    status = str(_attr(order, "status", "unknown")).lower()
    fq = _attr(order, "filled_qty", 0) or 0
    fp = _attr(order, "filled_avg_price", None)
    return status, float(fq), (float(fp) if fp not in (None, "") else None)


def _is_not_found(exc):
    text = str(exc).lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return code == 404 or "404" in text or "not found" in text or "does not exist" in text


def _age_hours(iso_ts):
    try:
        dt = datetime.fromisoformat(iso_ts)
    except (TypeError, ValueError):
        return 0.0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def wait_for_fill(api, order_id, timeout=ORDER_FILL_TIMEOUT_SECONDS, poll=1.0,
                  sleep_fn=time.sleep, clock_fn=time.monotonic):
    """Polls the broker until the order is terminal or the timeout passes."""
    start = clock_fn()
    order = api.get_order(order_id)
    while True:
        status, _, _ = _order_fill_state(order)
        if status in TERMINAL_STATUSES:
            return order
        if clock_fn() - start >= timeout:
            return order
        sleep_fn(poll)
        order = api.get_order(order_id)


def _fetch_broker_order(api, o):
    """Looks an order up by broker id, falling back to client id. Returns (order, error)."""
    if o.get("broker_order_id"):
        try:
            return api.get_order(o["broker_order_id"]), None
        except Exception as e:
            if not _is_not_found(e):
                return None, e
            # fall through to client id lookup: the broker id may have been mis-recorded
    if o.get("client_order_id"):
        try:
            return api.get_order_by_client_order_id(o["client_order_id"]), None
        except Exception as e:
            return None, e
    return None, LookupError("no broker id and no client id")


def _apply_order_state(strategy, o, border, db_path):
    """Applies any unapplied fill from `border` to the ledger + order row atomically."""
    status, fq, fp = _order_fill_state(border)
    already = float(o["filled_qty"] or 0.0)
    delta = fq - already
    broker_id = str(_attr(border, "id", "")) or o.get("broker_order_id")
    if delta > 1e-12:
        if not fp:
            # Filled quantity without a price cannot be booked. Keep the order in an
            # OPEN local state ("awaiting_fill_price") so the next reconcile retries;
            # writing the broker's terminal status here would lose the fill forever.
            portfolio.update_order(o["id"], "awaiting_fill_price", broker_order_id=broker_id, db_path=db_path)
            return "awaiting_fill_price", already, fp
        try:
            portfolio.apply_fill(strategy, o["symbol"], o["side"], delta, fp, db_path=db_path,
                                 order_update=(o["id"], status, fq, fp))
        except ValueError as e:
            # A sell fill larger than the ledger: never let one bad order break every
            # future cycle. Record it, mark the order terminal, and continue.
            print(f"[execution] RECONCILE ERROR {strategy} {o['symbol']}: {e}")
            log_trade(strategy, o["symbol"], "RECONCILE", 0, delta, pnl=None, order_id=broker_id,
                      status="reconcile_error", reason=f"fill could not be booked: {e}", db_path=db_path)
            portfolio.update_order(o["id"], "reconcile_error", filled_qty=fq, filled_avg_price=fp,
                                   broker_order_id=broker_id, db_path=db_path)
            return "reconcile_error", fq, fp
    else:
        portfolio.update_order(o["id"], status, filled_avg_price=fp, broker_order_id=broker_id, db_path=db_path)
    if broker_id and not o.get("broker_order_id"):
        portfolio.update_order(o["id"], status, broker_order_id=broker_id, db_path=db_path)
    return status, fq, fp


def reconcile_open_orders(api, strategy, db_path=None):
    """
    Refreshes every locally-open order from the broker and applies any fill
    quantity not yet applied to the ledger. Called at startup (fills that
    happened while the process was down) and before each new order.
    Returns the list of orders that are still open. Never raises.
    """
    still_open = []
    for o in portfolio.open_orders(strategy, db_path=db_path):
        border, err = _fetch_broker_order(api, o)
        if border is None:
            if err is not None and _is_not_found(err):
                # The broker has no record of it (e.g. paper account reset): it cannot fill.
                print(f"[execution] {strategy} order {o['id']} unknown at broker; closing it locally")
                portfolio.update_order(o["id"], "unknown_at_broker", db_path=db_path)
                continue
            if _age_hours(o.get("submitted_at")) > UNRESOLVED_ORDER_MAX_AGE_HOURS:
                print(f"[execution] {strategy} order {o['id']} unresolved for >{UNRESOLVED_ORDER_MAX_AGE_HOURS}h; "
                      f"marking unresolved so it no longer blocks {o['symbol']} (check the broker manually)")
                portfolio.update_order(o["id"], "unresolved_stale", db_path=db_path)
                log_trade(strategy, o["symbol"], "RECONCILE", 0, o["requested_qty"], pnl=None,
                          status="unresolved_stale", reason=f"order could not be refreshed: {err}", db_path=db_path)
                continue
            still_open.append({**o, "status": "unresolved"})
            portfolio.update_order(o["id"], "unresolved", db_path=db_path)
            print(f"[execution] {strategy} could not refresh order {o['id']}: {err}")
            continue
        status, fq, fp = _apply_order_state(strategy, o, border, db_path)
        if status not in TERMINAL_STATUSES:
            still_open.append({**o, "status": status, "filled_qty": fq})
    return still_open


def has_open_order(api, strategy, symbol, db_path=None):
    return any(o["symbol"] == symbol for o in reconcile_open_orders(api, strategy, db_path=db_path))


def _account_position_qty(api, broker_symbol):
    try:
        return float(api.get_position(broker_symbol).qty)
    except Exception:
        return 0.0


def _write_off(strategy, symbol, qty, price, reason, risk_state, owned_qty, account_qty, db_path):
    portfolio.apply_fill(strategy, symbol, "sell", qty, price, db_path=db_path)
    log_trade(strategy, symbol, "RECONCILE", price, qty, pnl=None, reason=reason, status="reconciled",
              strategy_position_before=owned_qty, account_position_before=account_qty,
              risk_state=risk_state, db_path=db_path)


def submit_and_track(api, strategy, symbol, side, qty, signal_price, reason,
                     time_in_force="day", risk_state=None, model_output=None,
                     wait=True, db_path=None, sleep_fn=time.sleep, clock_fn=time.monotonic):
    """
    Submits a market order for `qty` shares/coins, waits for the fill, updates
    the strategy ledger and writes ONE trades row with the real outcome.
    Never raises for a broker rejection: returns an OrderResult with status.
    """
    side = side.lower()
    broker_symbol = to_broker_symbol(symbol)
    try:
        qty = float(qty)
    except (TypeError, ValueError):
        qty = float("nan")
    if not math.isfinite(qty) or qty <= 0:
        return OrderResult(strategy, symbol, side, qty, "refused", message="quantity must be a positive finite number")

    owned_qty, avg_entry, _ = portfolio.get_position(strategy, symbol, db_path=db_path)
    if side == "sell":
        if owned_qty <= 0:
            return OrderResult(strategy, symbol, side, qty, "refused",
                               message=f"{strategy} owns no {symbol}; refusing to sell (no shorting)")
        if qty > owned_qty * (1 + 1e-6):
            print(f"[execution] {strategy} asked to sell {qty} {symbol} but owns {owned_qty}; clamping")
            qty = owned_qty
    elif side != "buy":
        return OrderResult(strategy, symbol, side, qty, "refused", message=f"unknown side {side}")

    if has_open_order(api, strategy, symbol, db_path=db_path):
        return OrderResult(strategy, symbol, side, qty, "refused",
                           message=f"{strategy} already has an open {symbol} order; not submitting another")

    account_qty_before = _account_position_qty(api, broker_symbol)
    if side == "sell":
        # The account is shared. Honour every OTHER strategy's ledger claim first; this
        # strategy may only sell what is left (second-pass S-03). If nothing is left, the
        # shares were sold outside this strategy and the ledger is written off.
        others = portfolio.total_ledger_qty(symbol, exclude_strategy=strategy, db_path=db_path)
        available = max(0.0, account_qty_before - others)
        if available + 1e-9 < qty:
            print(f"[execution] DRIFT {strategy} {symbol}: ledger {owned_qty}, account {account_qty_before}, "
                  f"claimed by others {others}, available {available}")
            missing = owned_qty - available
            _write_off(strategy, symbol, missing, signal_price or avg_entry,
                       "ledger quantity missing at broker; written off", risk_state, owned_qty, account_qty_before, db_path)
            qty = available
            if qty <= 0:
                return OrderResult(strategy, symbol, side, missing, "reconciled",
                                   message="position no longer exists at broker; ledger written off")
        if qty * (signal_price or avg_entry) < MIN_SELL_NOTIONAL:
            # Below the broker minimum: unsellable dust. Write it off rather than
            # submitting an order that is rejected every cycle (second-pass S-06).
            _write_off(strategy, symbol, qty, signal_price or avg_entry,
                       "dust below broker minimum notional; written off", risk_state, owned_qty, account_qty_before, db_path)
            return OrderResult(strategy, symbol, side, qty, "reconciled", message="dust written off")

    client_order_id = f"sq-{strategy}-{uuid.uuid4().hex[:20]}"
    row_id = portfolio.record_order(strategy, symbol, side, qty, signal_price, "submitting", reason=reason,
                                    client_order_id=client_order_id, db_path=db_path)
    try:
        order = api.submit_order(symbol=broker_symbol, qty=qty, side=side, type="market",
                                 time_in_force=time_in_force, client_order_id=client_order_id)
    except Exception as e:
        # The broker may have accepted the order even though we saw an exception
        # (network timeout after processing). Check by client id before calling it rejected.
        order = None
        try:
            order = api.get_order_by_client_order_id(client_order_id)
        except Exception:
            order = None
        if order is None:
            portfolio.update_order(row_id, "rejected", db_path=db_path)
            log_trade(strategy, symbol, side.upper(), signal_price, qty, pnl=None, reason=reason,
                      signal_price=signal_price, status="rejected", filled_qty=0.0,
                      strategy_position_before=owned_qty, account_position_before=account_qty_before,
                      risk_state=risk_state, model_output=model_output, db_path=db_path)
            return OrderResult(strategy, symbol, side, qty, "rejected", message=f"broker rejected: {e}")
        print(f"[execution] submit raised ({e}) but the broker has the order; continuing with it")

    order_id = str(_attr(order, "id", ""))
    portfolio.update_order(row_id, "new", broker_order_id=order_id, db_path=db_path)

    if wait:
        try:
            order = wait_for_fill(api, order_id, sleep_fn=sleep_fn, clock_fn=clock_fn)
        except Exception as e:
            print(f"[execution] could not poll order {order_id}: {e}")

    status, filled_qty, filled_avg_price = _order_fill_state(order)
    realized = None
    booked_qty = 0.0
    if filled_qty > 0 and filled_avg_price:
        realized = portfolio.apply_fill(strategy, symbol, side, filled_qty, filled_avg_price, db_path=db_path,
                                        order_update=(row_id, status, filled_qty, filled_avg_price))
        booked_qty = filled_qty
        if side == "buy":
            realized = None
    else:
        # Nothing booked: keep filled_qty at 0 so a later reconcile books it when a price exists.
        portfolio.update_order(row_id, status, filled_avg_price=filled_avg_price, db_path=db_path)

    log_trade(strategy, symbol, side.upper(), signal_price, qty, pnl=realized, reason=reason,
              signal_price=signal_price, order_id=order_id, status=status,
              filled_qty=booked_qty, filled_avg_price=filled_avg_price,
              strategy_position_before=owned_qty, account_position_before=account_qty_before,
              risk_state=risk_state, model_output=model_output, db_path=db_path)

    if status == "rejected":
        msg = "broker rejected the order"
    elif status in TERMINAL_STATUSES and filled_qty <= 0:
        msg = f"order ended {status} with no fill"
    elif status not in TERMINAL_STATUSES:
        msg = f"order still {status}; will be reconciled next cycle"
    else:
        msg = "filled"
    return OrderResult(strategy, symbol, side, qty, status, order_id, booked_qty, filled_avg_price, realized, msg)


def close_strategy_position(api, strategy, symbol, signal_price, reason, time_in_force="day",
                            risk_state=None, model_output=None, db_path=None, **kw):
    """Sells exactly the quantity this strategy owns of `symbol`. Never account-wide."""
    owned_qty, _, _ = portfolio.get_position(strategy, symbol, db_path=db_path)
    if owned_qty <= 0:
        return OrderResult(strategy, symbol, "sell", 0.0, "refused", message="nothing owned")
    return submit_and_track(api, strategy, symbol, "sell", owned_qty, signal_price, reason,
                            time_in_force=time_in_force, risk_state=risk_state,
                            model_output=model_output, db_path=db_path, **kw)


def close_all_strategy_positions(api, strategy, mark_prices, reason, time_in_force="day",
                                 risk_state=None, db_path=None, **kw):
    """
    Emergency exit for ONE strategy: closes each position in its own ledger.
    Replaces api.close_all_positions(), which liquidated every strategy.
    """
    results = []
    for symbol, (qty, avg, _) in portfolio.list_positions(strategy, db_path=db_path).items():
        px = mark_prices.get(symbol) or avg
        results.append(close_strategy_position(api, strategy, symbol, px, reason,
                                               time_in_force=time_in_force, risk_state=risk_state,
                                               db_path=db_path, **kw))
    return results
