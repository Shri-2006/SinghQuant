"""
Strategy-level portfolio ledger.

WHY THIS EXISTS (ENGINEERING_AUDIT.md issues C-02 / C-03)
---------------------------------------------------------
All strategies share one Alpaca paper account. Alpaca positions are
account-wide and carry no notion of which bot bought them. Before the audit,
every strategy sized, valued, stopped-out and kill-switched against the
ACCOUNT's positions and equity, so:
  * a kill switch in one bot liquidated every bot's holdings,
  * a strategy could sell inventory it never bought,
  * "drawdown" was account equity versus one strategy's notional capital.

This module keeps, per strategy, in SQLite:
  * positions      : qty, average entry and last accepted mark per symbol that THIS strategy owns
  * strategy_state : the strategy's own cash budget, peak equity, kill-switch state
  * signal_state   : last bar/action per symbol (anti-churn state machine)
  * orders         : BROKER orders this strategy submitted (broker semantics only)
  * ledger_events  : the journal of EVERY inventory/cash mutation and why
                     (FILL = broker fill, ADOPT = migration, WRITE_OFF_DRIFT =
                     manual reconciliation of a flagged discrepancy, R-1). Third pass T-01/T-05: internal
                     accounting events are never represented as broker orders
                     or fills, and can never increase equity.

Strategy equity = cash_budget + sum(qty * mark_price). All quantities are in
shares (or coins). Dollars never enter the `qty` fields.
"""
import math
from datetime import datetime, timezone

from core.config import CAPITAL
from core.logger import db_connection, utc_now_iso, is_readonly

EVENT_FILL = "FILL"
EVENT_ADOPT = "ADOPT"
EVENT_WRITE_OFF_DRIFT = "WRITE_OFF_DRIFT"


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Strategy state (cash budget, peak equity, kill switch)
# ---------------------------------------------------------------------------

def ensure_strategy_state(strategy, db_path=None):
    """Creates the state row for a strategy on first use (cash = CAPITAL). No-op when read-only."""
    if is_readonly():
        return
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT strategy FROM strategy_state WHERE strategy = ?',
                           (strategy,)).fetchone()
        if row is None:
            conn.execute('''INSERT INTO strategy_state
                            (strategy, cash_budget, peak_equity, last_equity, critical_streak, halted, halted_reason, updated_at)
                            VALUES (?,?,?,?,0,0,NULL,?)''',
                         (strategy, CAPITAL[strategy], CAPITAL[strategy], CAPITAL[strategy], utc_now_iso()))


def get_strategy_state(strategy, db_path=None):
    ensure_strategy_state(strategy, db_path)
    with db_connection(db_path) as conn:
        row = conn.execute('''SELECT cash_budget, peak_equity, last_equity, critical_streak, halted, halted_reason,
                                     suspect_streak
                              FROM strategy_state WHERE strategy = ?''', (strategy,)).fetchone()
    if row is None:   # read-only reader before the strategy ever ran: report the configured start state
        row = (CAPITAL[strategy], CAPITAL[strategy], CAPITAL[strategy], 0, 0, None, 0)
    return {
        "cash_budget": row[0], "peak_equity": row[1], "last_equity": row[2],
        "critical_streak": row[3], "halted": bool(row[4]), "halted_reason": row[5],
        "suspect_streak": int(row[6] or 0),
    }


def update_strategy_state(strategy, db_path=None, **fields):
    ensure_strategy_state(strategy, db_path)
    allowed = {"cash_budget", "peak_equity", "last_equity", "critical_streak", "halted", "halted_reason",
               "suspect_streak"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"unknown strategy_state fields: {bad}")
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    vals = [int(v) if k == "halted" else v for k, v in fields.items()]
    with db_connection(db_path) as conn:
        conn.execute(f'UPDATE strategy_state SET {sets}, updated_at = ? WHERE strategy = ?',
                     (*vals, utc_now_iso(), strategy))


def set_halted(strategy, halted, reason=None, db_path=None):
    update_strategy_state(strategy, db_path, halted=halted, halted_reason=reason)


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------

def get_position(strategy, symbol, db_path=None):
    """Returns (qty, avg_entry_price, opened_at) owned by this strategy; (0.0, 0.0, None) if none."""
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT qty, avg_entry_price, opened_at FROM positions WHERE strategy = ? AND symbol = ?',
                           (strategy, symbol)).fetchone()
    if row is None:
        return 0.0, 0.0, None
    return float(row[0]), float(row[1]), row[2]


def get_position_qty(strategy, symbol, db_path=None):
    return get_position(strategy, symbol, db_path)[0]


def list_positions(strategy, db_path=None):
    """Returns {symbol: (qty, avg_entry_price, opened_at)} for this strategy."""
    with db_connection(db_path) as conn:
        rows = conn.execute('SELECT symbol, qty, avg_entry_price, opened_at FROM positions WHERE strategy = ? AND qty > 0',
                            (strategy,)).fetchall()
    return {r[0]: (float(r[1]), float(r[2]), r[3]) for r in rows}


def get_last_marks(strategy, db_path=None):
    """{symbol: last accepted mark price} for this strategy's positions (None if never marked)."""
    with db_connection(db_path) as conn:
        rows = conn.execute('SELECT symbol, last_mark FROM positions WHERE strategy = ? AND qty > 0', (strategy,)).fetchall()
    return {r[0]: (float(r[1]) if r[1] is not None else None) for r in rows}


def set_last_marks(strategy, marks, db_path=None):
    """Records the marks that were accepted for valuation this cycle."""
    if not marks:
        return
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        for symbol, px in marks.items():
            if px is not None and px > 0:
                conn.execute('UPDATE positions SET last_mark = ?, last_mark_at = ? WHERE strategy = ? AND symbol = ?',
                             (float(px), now, strategy, symbol))


def lower_last_marks(strategy, marks, db_path=None):
    """
    Ratchet-DOWN update of last_mark (used for flagged positions, R-1): the
    reference price of a position the broker no longer shows can only move
    down, so a genuine gradual decline is followed while a single implausible
    print (beyond MARK_MAX_STEP_CHANGE) is still rejected against it.
    """
    if not marks:
        return
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        for symbol, px in marks.items():
            if px is not None and px > 0:
                conn.execute('''UPDATE positions SET last_mark = ?, last_mark_at = ?
                                WHERE strategy = ? AND symbol = ?
                                  AND (last_mark IS NULL OR last_mark > ?)''',
                             (float(px), now, strategy, symbol, float(px)))


def total_ledger_qty(symbol, exclude_strategy=None, db_path=None):
    """
    Total quantity of `symbol` claimed by every strategy's ledger (optionally
    excluding one). Used by the execution layer to make sure the sum of all
    ledgers never exceeds what the shared account holds (second-pass S-03).
    Quantity FLAGGED as missing at the broker (remediation R-1) is not a claim
    on the account's shares: it is excluded, so a vanished position in one
    strategy cannot block another strategy from selling what it really owns.
    """
    claim = 'COALESCE(SUM(qty - COALESCE(discrepancy_qty, 0)), 0)'
    with db_connection(db_path) as conn:
        if exclude_strategy is None:
            row = conn.execute(f'SELECT {claim} FROM positions WHERE symbol = ?', (symbol,)).fetchone()
        else:
            row = conn.execute(f'SELECT {claim} FROM positions WHERE symbol = ? AND strategy != ?',
                               (symbol, exclude_strategy)).fetchone()
    return max(0.0, float(row[0] or 0.0))


# ---------------------------------------------------------------------------
# Discrepancy flags (remediation R-1, owner decision D1 option B)
#
# When the broker no longer holds (part of) a quantity this strategy's ledger
# owns, the missing quantity is FLAGGED on the position row. Nothing else
# changes: qty, average entry, cash and ownership are preserved, no loss is
# realized, and nothing is journaled as a fill or write-off. A human resolves
# the flag with tools/reconcile_position.py. Flagged quantity is valued
# conservatively (see strategy_equity) and is not a claim on account shares.
# ---------------------------------------------------------------------------

def get_discrepancies(strategy=None, db_path=None):
    """{(strategy, symbol): {...}} for every position with a flagged quantity."""
    sql = ('''SELECT strategy, symbol, qty, avg_entry_price, last_mark, discrepancy_qty, discrepancy_reason,
                     discrepancy_since, discrepancy_alerted
              FROM positions WHERE discrepancy_qty IS NOT NULL AND discrepancy_qty > 0''')
    params = ()
    if strategy:
        sql += ' AND strategy = ?'
        params = (strategy,)
    with db_connection(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return {(r[0], r[1]): {"qty": float(r[2]), "avg_entry_price": float(r[3]),
                           "last_mark": None if r[4] is None else float(r[4]),
                           "missing_qty": float(r[5]), "reason": r[6], "since": r[7], "alerted": bool(r[8])}
            for r in rows}


def get_discrepancy_qty(strategy, symbol, db_path=None):
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT discrepancy_qty FROM positions WHERE strategy = ? AND symbol = ?',
                           (strategy, symbol)).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.0


def flag_discrepancy(strategy, symbol, missing_qty, reason, db_path=None):
    """
    Flags `missing_qty` of this strategy's `symbol` as not held at the broker.
    Never changes qty, cost, cash or ownership. Returns True when the flag is
    new or its quantity changed (callers log/alert only on change).
    """
    missing_qty = float(missing_qty)
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT qty, discrepancy_qty FROM positions WHERE strategy = ? AND symbol = ?',
                           (strategy, symbol)).fetchone()
        if row is None or missing_qty <= 0:
            return False
        missing_qty = min(missing_qty, float(row[0]))
        old = float(row[1]) if row[1] is not None else 0.0
        if old > 0 and abs(old - missing_qty) <= 1e-9 * max(1.0, old):
            return False
        now = utc_now_iso()
        if old > 0:
            conn.execute('''UPDATE positions SET discrepancy_qty = ?, discrepancy_reason = ?, discrepancy_alerted = 0
                            WHERE strategy = ? AND symbol = ?''', (missing_qty, reason, strategy, symbol))
        else:
            conn.execute('''UPDATE positions SET discrepancy_qty = ?, discrepancy_reason = ?, discrepancy_since = ?,
                            discrepancy_alerted = 0 WHERE strategy = ? AND symbol = ?''',
                         (missing_qty, reason, now, strategy, symbol))
    return True


def clear_discrepancy(strategy, symbol, db_path=None):
    """Removes a flag (the broker holds the quantity again, or a human resolved it). Returns True if one existed."""
    with db_connection(db_path) as conn:
        cur = conn.execute('''UPDATE positions SET discrepancy_qty = NULL, discrepancy_reason = NULL,
                               discrepancy_since = NULL, discrepancy_alerted = 0
                               WHERE strategy = ? AND symbol = ? AND discrepancy_qty IS NOT NULL''', (strategy, symbol))
        return cur.rowcount > 0


def mark_discrepancy_alerted(strategy, symbol, db_path=None):
    with db_connection(db_path) as conn:
        conn.execute('UPDATE positions SET discrepancy_alerted = 1 WHERE strategy = ? AND symbol = ?', (strategy, symbol))


def conservative_price(*prices):
    """Lowest valid (finite, positive) price among the candidates, or None."""
    valid = []
    for p in prices:
        try:
            p = float(p)
        except (TypeError, ValueError):
            continue
        if math.isfinite(p) and p > 0:
            valid.append(p)
    return min(valid) if valid else None


def _insert_event(conn, strategy, symbol, event_type, qty_delta, cash_delta, price, avg_entry_before,
                  qty_before, qty_after, reason, broker_order_id=None, extra=None):
    conn.execute('''INSERT INTO ledger_events (timestamp, strategy, symbol, event_type, qty_delta, cash_delta, price,
                                               avg_entry_before, qty_before, qty_after, reason, broker_order_id, extra)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                 (utc_now_iso(), strategy, symbol, event_type, float(qty_delta), float(cash_delta),
                  None if price is None else float(price), float(avg_entry_before), float(qty_before),
                  float(qty_after), reason, broker_order_id, extra))


def apply_fill(strategy, symbol, side, filled_qty, fill_price, db_path=None, order_update=None,
               broker_order_id=None, reason=None):
    """
    Applies a BROKER fill to the strategy ledger and its cash budget.
    side: "buy" | "sell". filled_qty is a share/coin quantity (never dollars).
    Returns realized pnl in dollars for sells (0.0 for buys).
    Raises ValueError if a sell exceeds the owned quantity: the ledger never
    goes negative, because this system does not short.

    This function is for broker fills ONLY. Internal reconciliation must use
    write_off(), which cannot credit proceeds above mark or cost (third pass T-01).

    order_update: optional (order_row_id, status, total_filled_qty, filled_avg_price)
    written in the SAME transaction, so a crash can never leave a fill applied
    to the ledger but not recorded on the order (or vice versa): second-pass S-05.
    """
    filled_qty = float(filled_qty)
    fill_price = float(fill_price)
    if filled_qty <= 0 or fill_price <= 0:
        return 0.0
    ensure_strategy_state(strategy, db_path)
    realized = 0.0
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        if order_update is not None:
            oid, ostatus, ofq, ofp = order_update
            conn.execute('''UPDATE orders SET status = ?, filled_qty = ?, filled_avg_price = COALESCE(?, filled_avg_price),
                            updated_at = ? WHERE id = ?''', (ostatus, float(ofq), ofp, now, oid))
        row = conn.execute('SELECT qty, avg_entry_price, opened_at FROM positions WHERE strategy = ? AND symbol = ?',
                           (strategy, symbol)).fetchone()
        qty, avg, opened_at = (float(row[0]), float(row[1]), row[2]) if row else (0.0, 0.0, None)
        cash = conn.execute('SELECT cash_budget FROM strategy_state WHERE strategy = ?', (strategy,)).fetchone()[0]

        if side == "buy":
            new_qty = qty + filled_qty
            new_avg = (qty * avg + filled_qty * fill_price) / new_qty
            cash_delta = -filled_qty * fill_price
            cash += cash_delta
            if row is None:
                conn.execute('INSERT INTO positions (strategy, symbol, qty, avg_entry_price, opened_at, updated_at) VALUES (?,?,?,?,?,?)',
                             (strategy, symbol, new_qty, new_avg, now, now))
            else:
                conn.execute('UPDATE positions SET qty = ?, avg_entry_price = ?, updated_at = ? WHERE strategy = ? AND symbol = ?',
                             (new_qty, new_avg, now, strategy, symbol))
            _insert_event(conn, strategy, symbol, EVENT_FILL, +filled_qty, cash_delta, fill_price, avg, qty, new_qty,
                          reason or "broker buy fill", broker_order_id)
        elif side == "sell":
            # Tolerate float noise from broker rounding, never a real oversell.
            if filled_qty > qty * (1 + 1e-6) + 1e-9:
                raise ValueError(f"{strategy} cannot sell {filled_qty} {symbol}: owns only {qty}")
            filled_qty = min(filled_qty, qty)
            realized = filled_qty * (fill_price - avg)
            cash_delta = filled_qty * fill_price
            cash += cash_delta
            new_qty = qty - filled_qty
            if new_qty <= 1e-9:
                new_qty = 0.0
                conn.execute('DELETE FROM positions WHERE strategy = ? AND symbol = ?', (strategy, symbol))
            else:
                # A sell only ever sells what the broker holds, never the flagged
                # remainder; the flag can at most equal what is left (R-1).
                conn.execute('''UPDATE positions SET qty = ?, updated_at = ?,
                                discrepancy_qty = CASE WHEN discrepancy_qty IS NOT NULL AND discrepancy_qty > ?
                                                       THEN ? ELSE discrepancy_qty END
                                WHERE strategy = ? AND symbol = ?''',
                             (new_qty, now, new_qty, new_qty, strategy, symbol))
            _insert_event(conn, strategy, symbol, EVENT_FILL, -filled_qty, cash_delta, fill_price, avg, qty, new_qty,
                          reason or "broker sell fill", broker_order_id, extra=f"realized={realized:.6f}")
        else:
            raise ValueError(f"unknown side {side!r}")

        conn.execute('UPDATE strategy_state SET cash_budget = ?, updated_at = ? WHERE strategy = ?',
                     (cash, now, strategy))
    return realized


def adopt_position(strategy, symbol, qty, avg_entry_price, db_path=None):
    """
    Records a position that already exists at the broker as owned by a
    strategy (used once when migrating an old database). Debits the cash
    budget at the entry price so equity stays consistent. Journaled as ADOPT.
    """
    qty = float(qty)
    avg_entry_price = float(avg_entry_price)
    if qty <= 0 or avg_entry_price <= 0:
        return
    ensure_strategy_state(strategy, db_path)
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT qty, avg_entry_price FROM positions WHERE strategy = ? AND symbol = ?',
                           (strategy, symbol)).fetchone()
        old_qty, old_avg = (float(row[0]), float(row[1])) if row else (0.0, 0.0)
        new_qty = old_qty + qty
        new_avg = (old_qty * old_avg + qty * avg_entry_price) / new_qty
        cash = conn.execute('SELECT cash_budget FROM strategy_state WHERE strategy = ?', (strategy,)).fetchone()[0]
        cash_delta = -qty * avg_entry_price
        if row is None:
            conn.execute('INSERT INTO positions (strategy, symbol, qty, avg_entry_price, opened_at, updated_at) VALUES (?,?,?,?,?,?)',
                         (strategy, symbol, new_qty, new_avg, now, now))
        else:
            conn.execute('UPDATE positions SET qty = ?, avg_entry_price = ?, updated_at = ? WHERE strategy = ? AND symbol = ?',
                         (new_qty, new_avg, now, strategy, symbol))
        conn.execute('UPDATE strategy_state SET cash_budget = ?, updated_at = ? WHERE strategy = ?',
                     (cash + cash_delta, now, strategy))
        _insert_event(conn, strategy, symbol, EVENT_ADOPT, +qty, cash_delta, avg_entry_price, old_avg, old_qty, new_qty,
                      "adopted pre-existing broker position into the ledger")


def write_off(strategy, symbol, qty, mark_price, reason, account_qty_observed=None, others_claim=None, db_path=None):
    """
    INTERNAL reconciliation: removes `qty` of `symbol` from the strategy's
    ledger because the shares are no longer at the broker under this
    strategy's claim (sold outside the system, legacy close_all, manual trade).
    Remediation R-1 (owner decision D1-B): the engine never calls this. It is
    reached only through execution.manual_write_off, i.e. a human running
    tools/reconcile_position.py, and it reduces the position's flag.

    No broker fill occurred, so this must never masquerade as one:
      * cash is credited at min(mark, average entry): equity can never INCREASE
        and no positive realized P&L can ever be created by an internal event
        (third pass T-01). Any unrealized gain on the written-off shares is
        forfeited; any unrealized loss is realized.
      * the mutation is journaled in ledger_events as WRITE_OFF_DRIFT with the
        observed broker quantity and other strategies' claims (T-05), with no
        broker_order_id, so it can never be mistaken for a fill.
    Returns (cash_credited, realized_estimate).
    """
    qty = float(qty)
    if qty <= 0:
        return 0.0, 0.0
    ensure_strategy_state(strategy, db_path)
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT qty, avg_entry_price FROM positions WHERE strategy = ? AND symbol = ?',
                           (strategy, symbol)).fetchone()
        if row is None:
            return 0.0, 0.0
        owned, avg = float(row[0]), float(row[1])
        qty = min(qty, owned)
        try:
            mark = float(mark_price)
        except (TypeError, ValueError):
            mark = float("nan")
        basis = avg if not (math.isfinite(mark) and mark > 0) else min(mark, avg)
        cash_delta = qty * basis
        realized = qty * (basis - avg)           # <= 0 by construction
        new_qty = owned - qty
        cash = conn.execute('SELECT cash_budget FROM strategy_state WHERE strategy = ?', (strategy,)).fetchone()[0]
        if new_qty <= 1e-9:
            new_qty = 0.0
            conn.execute('DELETE FROM positions WHERE strategy = ? AND symbol = ?', (strategy, symbol))
        else:
            # The written-off quantity comes out of the flagged quantity first (R-1).
            conn.execute('''UPDATE positions SET qty = ?, updated_at = ?,
                            discrepancy_qty = CASE WHEN discrepancy_qty IS NULL OR discrepancy_qty - ? <= 1e-12
                                                   THEN NULL ELSE discrepancy_qty - ? END
                            WHERE strategy = ? AND symbol = ?''',
                         (new_qty, now, qty, qty, strategy, symbol))
        conn.execute('UPDATE strategy_state SET cash_budget = ?, updated_at = ? WHERE strategy = ?',
                     (cash + cash_delta, now, strategy))
        _insert_event(conn, strategy, symbol, EVENT_WRITE_OFF_DRIFT, -qty, cash_delta, basis, avg, owned, new_qty,
                      reason, None,
                      extra=f"mark={mark_price} account_qty={account_qty_observed} others_claim={others_claim} realized_estimate={realized:.6f}")
    return cash_delta, realized


def list_ledger_events(strategy=None, symbol=None, event_type=None, db_path=None):
    """Journal rows (dicts) in insertion order, optionally filtered."""
    clauses, params = [], []
    if strategy:
        clauses.append("strategy = ?"); params.append(strategy)
    if symbol:
        clauses.append("symbol = ?"); params.append(symbol)
    if event_type:
        clauses.append("event_type = ?"); params.append(event_type)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    cols = ["id", "timestamp", "strategy", "symbol", "event_type", "qty_delta", "cash_delta", "price",
            "avg_entry_before", "qty_before", "qty_after", "reason", "broker_order_id", "extra"]
    with db_connection(db_path) as conn:
        rows = conn.execute(f'SELECT {", ".join(cols)} FROM ledger_events {where} ORDER BY id', params).fetchall()
    return [dict(zip(cols, r)) for r in rows]


# ---------------------------------------------------------------------------
# Equity and drawdown
# ---------------------------------------------------------------------------

def strategy_equity(strategy, mark_prices, db_path=None):
    """
    Strategy equity = own cash budget + own positions at `mark_prices`.
    mark_prices: {symbol: price}. Symbols without a mark are valued at their
    last accepted mark, or at average entry if never marked (never at zero,
    which would fake a drawdown; see audit issue C-03).
    Returns (equity, positions_value, missing_marks).
    """
    state = get_strategy_state(strategy, db_path)
    with db_connection(db_path) as conn:
        rows = conn.execute('''SELECT symbol, qty, avg_entry_price, last_mark, discrepancy_qty
                                FROM positions WHERE strategy = ? AND qty > 0''', (strategy,)).fetchall()
    value = 0.0
    missing = []
    for symbol, qty, avg, last_mark, flagged in rows:
        qty, avg = float(qty), float(avg)
        flagged = min(qty, float(flagged)) if flagged else 0.0
        px = mark_prices.get(symbol)
        if px is None or not (px > 0):
            missing.append(symbol)
            px = last_mark or avg
        # Flagged quantity (not at the broker) is valued at the LOWEST of the
        # current mark, the last accepted mark and the average entry: an
        # unrealized gain on shares we cannot see is never counted, a loss is.
        # No loss is realized and cash is unchanged (remediation R-1).
        flag_px = conservative_price(px, last_mark, avg) if flagged else 0.0
        value += (qty - flagged) * px + flagged * flag_px
    return state["cash_budget"] + value, value, missing


# ---------------------------------------------------------------------------
# Signal / anti-churn state
# ---------------------------------------------------------------------------

def get_signal_state(strategy, symbol, db_path=None):
    with db_connection(db_path) as conn:
        row = conn.execute('SELECT last_bar, last_action, last_action_at FROM signal_state WHERE strategy = ? AND symbol = ?',
                           (strategy, symbol)).fetchone()
    if row is None:
        return {"last_bar": None, "last_action": None, "last_action_at": None}
    return {"last_bar": row[0], "last_action": row[1], "last_action_at": row[2]}


def set_signal_state(strategy, symbol, last_bar, last_action, db_path=None):
    with db_connection(db_path) as conn:
        conn.execute('''INSERT INTO signal_state (strategy, symbol, last_bar, last_action, last_action_at)
                        VALUES (?,?,?,?,?)
                        ON CONFLICT(strategy, symbol) DO UPDATE SET
                            last_bar = excluded.last_bar,
                            last_action = excluded.last_action,
                            last_action_at = excluded.last_action_at''',
                     (strategy, symbol, None if last_bar is None else str(last_bar), last_action, utc_now_iso()))


# ---------------------------------------------------------------------------
# Orders (BROKER orders only; internal accounting never appears here)
# ---------------------------------------------------------------------------

# Remediation R-4: an order is OPEN unless its status is known to be FINAL.
# The old allow-list missed broker states such as pending_cancel,
# pending_replace, calculated, done_for_day, stopped and suspended; a fill
# arriving after them was never booked. Anything not listed here (including
# "submitting", "unresolved" and "awaiting_fill_price", second-pass S-01) is
# reconciled every cycle, keeps blocking duplicate orders for the symbol and
# keeps reserving budget.
TERMINAL_ORDER_STATUSES = ("filled", "canceled", "cancelled", "expired", "rejected", "replaced",
                           "unknown_at_broker", "reconcile_error", "unresolved_stale")


def is_open_status(status):
    return str(status).lower() not in TERMINAL_ORDER_STATUSES


def record_order(strategy, symbol, side, requested_qty, signal_price, status, broker_order_id=None,
                 filled_qty=0.0, filled_avg_price=None, reason=None, client_order_id=None, db_path=None):
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        cur = conn.execute('''INSERT INTO orders (strategy, symbol, side, requested_qty, signal_price, broker_order_id,
                                                  status, filled_qty, filled_avg_price, reason, submitted_at, updated_at, client_order_id)
                              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                           (strategy, symbol, side, float(requested_qty), signal_price, broker_order_id,
                            status, float(filled_qty or 0.0), filled_avg_price, reason, now, now, client_order_id))
        return cur.lastrowid


# ---------------------------------------------------------------------------
# Strategy budget (remediation R-3)
#
# A strategy may only commit cash its own budget still has:
#     available = cash_budget - notional of this strategy's OPEN buy orders
# (unfilled quantity x signal price). The check and the order row that
# reserves the cash are written in ONE `BEGIN IMMEDIATE` transaction, so two
# threads or two processes on the same database cannot both pass the check
# against the same dollars.
# ---------------------------------------------------------------------------

BUDGET_TOLERANCE = 1e-6


def _reserved_buy_notional(conn, strategy):
    placeholders = ",".join("?" for _ in TERMINAL_ORDER_STATUSES)
    row = conn.execute(f'''SELECT COALESCE(SUM(MAX(requested_qty - filled_qty, 0) *
                                           COALESCE(signal_price, filled_avg_price, 0)), 0)
                            FROM orders WHERE strategy = ? AND side = 'buy'
                              AND LOWER(status) NOT IN ({placeholders})''',
                       (strategy, *TERMINAL_ORDER_STATUSES)).fetchone()
    return float(row[0] or 0.0)


def available_budget(strategy, db_path=None):
    """Cash this strategy may still commit: cash_budget minus open buy reservations."""
    ensure_strategy_state(strategy, db_path)
    with db_connection(db_path) as conn:
        cash = conn.execute('SELECT cash_budget FROM strategy_state WHERE strategy = ?', (strategy,)).fetchone()[0]
        return float(cash) - _reserved_buy_notional(conn, strategy)


def reserve_buy_order(strategy, symbol, qty, signal_price, reason=None, client_order_id=None, db_path=None):
    """
    Atomically checks the budget and records a "submitting" buy order that
    reserves qty x signal_price. Returns (order_row_id, available_before);
    order_row_id is None when the budget is insufficient (nothing written).
    """
    ensure_strategy_state(strategy, db_path)
    qty = float(qty)
    try:
        price = float(signal_price)
    except (TypeError, ValueError):
        return None, 0.0
    if not (math.isfinite(price) and price > 0 and math.isfinite(qty) and qty > 0):
        return None, 0.0
    now = utc_now_iso()
    with db_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cash = conn.execute('SELECT cash_budget FROM strategy_state WHERE strategy = ?', (strategy,)).fetchone()[0]
        available = float(cash) - _reserved_buy_notional(conn, strategy)
        if qty * price > available + BUDGET_TOLERANCE:
            return None, available
        cur = conn.execute('''INSERT INTO orders (strategy, symbol, side, requested_qty, signal_price, broker_order_id,
                                                  status, filled_qty, filled_avg_price, reason, submitted_at, updated_at,
                                                  client_order_id)
                              VALUES (?,?,?,?,?,NULL,'submitting',0,NULL,?,?,?,?)''',
                           (strategy, symbol, "buy", qty, price, reason, now, now, client_order_id))
        return cur.lastrowid, available


def update_order(order_row_id, status, filled_qty=None, filled_avg_price=None, broker_order_id=None, db_path=None):
    with db_connection(db_path) as conn:
        conn.execute('''UPDATE orders SET status = ?,
                            filled_qty = COALESCE(?, filled_qty),
                            filled_avg_price = COALESCE(?, filled_avg_price),
                            broker_order_id = COALESCE(?, broker_order_id),
                            updated_at = ?
                        WHERE id = ?''',
                     (status, filled_qty, filled_avg_price, broker_order_id, utc_now_iso(), order_row_id))


def open_orders(strategy, symbol=None, db_path=None):
    """Orders this strategy submitted that are not yet in a terminal state."""
    placeholders = ",".join("?" for _ in TERMINAL_ORDER_STATUSES)
    cols = 'id, symbol, side, requested_qty, broker_order_id, status, filled_qty, client_order_id, submitted_at'
    open_clause = f'LOWER(status) NOT IN ({placeholders})'
    with db_connection(db_path) as conn:
        if symbol is None:
            rows = conn.execute(f'SELECT {cols} FROM orders WHERE strategy = ? AND {open_clause}',
                                (strategy, *TERMINAL_ORDER_STATUSES)).fetchall()
        else:
            rows = conn.execute(f'SELECT {cols} FROM orders WHERE strategy = ? AND symbol = ? AND {open_clause}',
                                (strategy, symbol, *TERMINAL_ORDER_STATUSES)).fetchall()
    return [{"id": r[0], "symbol": r[1], "side": r[2], "requested_qty": r[3], "broker_order_id": r[4],
             "status": r[5], "filled_qty": r[6], "client_order_id": r[7], "submitted_at": r[8]} for r in rows]


def record_equity_snapshot(strategy, equity, cash_budget, positions_value, peak_equity, drawdown, risk_level, db_path=None):
    # A suspect reading has no drawdown; SQLite stores NaN as NULL, so pass None explicitly.
    if drawdown is None or drawdown != drawdown:
        drawdown = None
    with db_connection(db_path) as conn:
        conn.execute('''INSERT INTO equity_snapshots (timestamp, strategy, equity, cash_budget, positions_value, peak_equity, drawdown, risk_level)
                        VALUES (?,?,?,?,?,?,?,?)''',
                     (utc_now_iso(), strategy, equity, cash_budget, positions_value, peak_equity, drawdown, risk_level))
