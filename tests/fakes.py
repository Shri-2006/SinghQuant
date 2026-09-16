"""
In-memory stand-in for alpaca_trade_api.REST used by the regression tests.

Supports the subset of the API SinghQuant uses: get_account, get_clock,
get_position, list_positions, submit_order, get_order,
get_order_by_client_order_id, close_position, close_all_positions.
Fill behaviour is configurable per test:
    fill_mode = "fill" | "partial" | "reject" | "pending" | "accept_then_raise"
("accept_then_raise" records the order at the broker, then raises from
submit_order, simulating a network timeout after the broker processed it.)
The account is ACCOUNT-WIDE exactly like Alpaca: positions have no owner.
"""
import itertools
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace


class FakeAPIError(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


class FakeAlpaca:
    def __init__(self, cash=1400.0, prices=None, fill_mode="fill", partial_fraction=0.5,
                 market_open=True, slippage=0.0):
        self.cash = float(cash)
        self.prices = dict(prices or {})
        self.positions = {}          # symbol -> {"qty", "avg"}
        self.orders = {}             # id -> order dict
        self.fill_mode = fill_mode
        self.partial_fraction = partial_fraction
        self.market_open = market_open
        self.slippage = slippage
        self._ids = itertools.count(1)
        self.submitted = []          # every submit_order call, in order
        self.closed_symbols = []
        self.close_all_calls = 0
        self.account_overrides = {}
        self.get_order_calls = 0
        self.fail_get_order = None   # exception to raise from get_order, if set
        self.fail_get_position = None  # exception to raise from get_position, if set

    # --- helpers for tests -------------------------------------------------
    def set_price(self, symbol, price):
        self.prices[symbol] = float(price)

    def seed_position(self, symbol, qty, avg):
        self.positions[symbol] = {"qty": float(qty), "avg": float(avg)}

    def fill_pending(self, order_id, fraction=1.0):
        """Fill (part of) a pending order later, simulating a delayed fill."""
        o = self.orders[order_id]
        remaining = o["qty"] - o["filled_qty"]
        self._apply_fill(o, remaining * fraction)

    def cancel_pending(self, order_id):
        self.orders[order_id]["status"] = "canceled"

    def forget_order(self, order_id):
        """Simulates a paper-account reset: the broker no longer knows the order."""
        del self.orders[order_id]

    # --- Alpaca-like surface -------------------------------------------------
    def _equity(self):
        return self.cash + sum(p["qty"] * self.prices.get(s, p["avg"]) for s, p in self.positions.items())

    def get_account(self):
        eq = self.account_overrides.get("portfolio_value", self._equity())
        raw = {"portfolio_value": str(eq), "equity": str(eq), "cash": str(self.cash),
               "buying_power": str(self.cash * 2), "last_equity": str(eq)}
        return SimpleNamespace(_raw=raw, portfolio_value=str(eq), equity=str(eq), cash=str(self.cash),
                               buying_power=str(self.cash * 2), last_equity=str(eq))

    def get_clock(self):
        now = datetime.now(timezone.utc)
        return SimpleNamespace(is_open=self.market_open, timestamp=now,
                               next_open=now + timedelta(hours=15), next_close=now + timedelta(hours=6))

    def get_position(self, symbol):
        if self.fail_get_position is not None:
            raise self.fail_get_position
        p = self.positions.get(symbol)
        if not p or p["qty"] == 0:
            raise FakeAPIError(f"position does not exist: {symbol}", 404)
        px = self.prices.get(symbol, p["avg"])
        return SimpleNamespace(symbol=symbol, qty=str(p["qty"]), avg_entry_price=str(p["avg"]),
                               current_price=str(px), market_value=str(p["qty"] * px),
                               unrealized_pl=str(p["qty"] * (px - p["avg"])),
                               unrealized_plpc=str((px - p["avg"]) / p["avg"] if p["avg"] else 0.0))

    def list_positions(self):
        return [self.get_position(s) for s, p in self.positions.items() if p["qty"] > 0]

    def _apply_fill(self, o, qty):
        if qty <= 0:
            return
        px = self.prices[o["symbol"]] * (1 + self.slippage if o["side"] == "buy" else 1 - self.slippage)
        prev_notional = o["filled_qty"] * (o["filled_avg_price"] or 0.0)
        o["filled_qty"] += qty
        o["filled_avg_price"] = (prev_notional + qty * px) / o["filled_qty"]
        p = self.positions.setdefault(o["symbol"], {"qty": 0.0, "avg": 0.0})
        if o["side"] == "buy":
            new_qty = p["qty"] + qty
            p["avg"] = (p["qty"] * p["avg"] + qty * px) / new_qty
            p["qty"] = new_qty
            self.cash -= qty * px
        else:
            p["qty"] -= qty
            self.cash += qty * px
            if p["qty"] <= 1e-12:
                del self.positions[o["symbol"]]
        o["status"] = "filled" if abs(o["filled_qty"] - o["qty"]) < 1e-9 else "partially_filled"

    def submit_order(self, symbol, qty, side, type="market", time_in_force="day", client_order_id=None):
        qty = float(qty)
        self.submitted.append({"symbol": symbol, "qty": qty, "side": side, "tif": time_in_force,
                               "client_order_id": client_order_id})
        if symbol not in self.prices:
            raise FakeAPIError(f"unknown symbol {symbol}", 422)
        if side == "sell" and self.positions.get(symbol, {"qty": 0})["qty"] + 1e-9 < qty:
            raise FakeAPIError("insufficient qty available for order", 403)
        if self.fill_mode == "reject":
            raise FakeAPIError("order rejected by broker", 403)
        oid = f"ord-{next(self._ids)}"
        o = {"id": oid, "symbol": symbol, "qty": qty, "side": side, "status": "new",
             "filled_qty": 0.0, "filled_avg_price": None, "client_order_id": client_order_id}
        self.orders[oid] = o
        if self.fill_mode == "fill":
            self._apply_fill(o, qty)
        elif self.fill_mode == "partial":
            self._apply_fill(o, qty * self.partial_fraction)
        elif self.fill_mode == "accept_then_raise":
            self._apply_fill(o, qty)
            raise FakeAPIError("read timed out", None)
        elif self.fill_mode == "pending":
            pass
        return self._order_ns(o)

    def _order_ns(self, o):
        return SimpleNamespace(id=o["id"], symbol=o["symbol"], qty=str(o["qty"]), side=o["side"],
                               status=o["status"], filled_qty=str(o["filled_qty"]),
                               filled_avg_price=None if o["filled_avg_price"] is None else str(o["filled_avg_price"]),
                               client_order_id=o.get("client_order_id"), _raw=dict(o))

    def get_order(self, order_id):
        self.get_order_calls += 1
        if self.fail_get_order is not None:
            raise self.fail_get_order
        if order_id not in self.orders:
            raise FakeAPIError(f"order not found: {order_id}", 404)
        return self._order_ns(self.orders[order_id])

    def get_order_by_client_order_id(self, client_order_id):
        for o in self.orders.values():
            if o.get("client_order_id") == client_order_id:
                return self._order_ns(o)
        raise FakeAPIError(f"order not found: {client_order_id}", 404)

    def list_orders(self, status="open", symbols=None, **kw):
        return [self._order_ns(o) for o in self.orders.values()
                if o["status"] in ("new", "partially_filled") and (not symbols or o["symbol"] in symbols)]

    def close_position(self, symbol):
        self.closed_symbols.append(symbol)
        p = self.positions.get(symbol)
        if not p:
            raise FakeAPIError("position does not exist", 404)
        return self.submit_order(symbol, p["qty"], "sell")

    def close_all_positions(self):
        self.close_all_calls += 1
        for s in list(self.positions):
            self.close_position(s)
