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
  * positions      : qty and average entry price per symbol that THIS strategy owns
  * strategy_state : the strategy's own cash budget, peak equity, kill-switch state
  * signal_state   : last bar/action per symbol (anti-churn state machine)

Strategy equity = cash_budget + sum(qty * mark_price). All quantities are in
shares (or coins). Dollars never enter the `qty` fields.
"""
from datetime import datetime, timezone

from core.config import CAPITAL
from core.logger import db_connection, utc_now_iso


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Strategy state (cash budget, peak equity, kill switch)
# ---------------------------------------------------------------------------

def ensure_strategy_state(strategy, db_path=None):
    """Creates the state row for a strategy on first use (cash = CAPITAL)."""
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
        row = conn.execute('''SELECT cash_budget, peak_equity, last_equity, critical_streak, halted, halted_reason
                              FROM strategy_state WHERE strategy = ?''', (strategy,)).fetchone()
    return {
        "cash_budget": row[0], "peak_equity": row[1], "last_equity": row[2],
        "critical_streak": row[3], "halted": bool(row[4]), "halted_reason": row[5],
    }


def update_strategy_state(strategy, db_path=None, **fields):
    ensure_strategy_state(strategy, db_path)
    allowed = {"cash_budget", "peak_equity", "last_equity", "critical_streak", "halted", "halted_reason"}
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


def total_ledger_qty(symbol, exclude_strategy=None, db_path=None):
    """
    Total quantity of `symbol` claimed by every strategy's ledger (optionally
    excluding one). Used by the execution layer to make sure the sum of all
    ledgers never exceeds what the shared account holds (second-pass S-03).
    """
    with db_connection(db_path) as conn:
        if exclude_strategy is None:
            row = conn.execute('SELECT COALESCE(SUM(qty), 0) FROM positions WHERE symbol = ?', (symbol,)).fetchone()
        else:
            row = conn.execute('SELECT COALESCE(SUM(qty), 0) FROM positions WHERE symbol = ? AND strategy != ?',
                               (symbol, exclude_strategy)).fetchone()
    return float(row[0] or 0.0)


def apply_fill(strategy, symbol, side, filled_qty, fill_price, db_path=None, order_update=None):
    """
    Applies a broker fill to the strategy ledger and its cash budget.
    side: "buy" | "sell". filled_qty is a share/coin quantity (never dollars).
    Returns realized pnl in dollars for sells (0.0 for buys).
    Raises ValueError if a sell exceeds the owned quantity: the ledger never
    goes negative, because this system does not short.

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
            cash -= filled_qty * fill_price
            if row is None:
                conn.execute('INSERT INTO positions (strategy, symbol, qty, avg_entry_price, opened_at, updated_at) VALUES (?,?,?,?,?,?)',
                             (strategy, symbol, new_qty, new_avg, now, now))
            else:
                conn.execute('UPDATE positions SET qty = ?, avg_entry_price = ?, updated_at = ? WHERE strategy = ? AND symbol = ?',
                             (new_qty, new_avg, now, strategy, symbol))
        elif side == "sell":
            # Tolerate float noise from broker rounding, never a real oversell.
            if filled_qty > qty * (1 + 1e-6) + 1e-9:
                raise ValueError(f"{strategy} cannot sell {filled_qty} {symbol}: owns only {qty}")
            filled_qty = min(filled_qty, qty)
            realized = filled_qty * (fill_price - avg)
            cash += filled_qty * fill_price
            new_qty = qty - filled_qty
            if new_qty <= 1e-9:
                conn.execute('DELETE FROM positions WHERE strategy = ? AND symbol = ?', (strategy, symbol))
            else:
                conn.execute('UPDATE positions SET qty = ?, updated_at = ? WHERE strategy = ? AND symbol = ?',
                             (new_qty, now, strategy, symbol))
        else:
            raise ValueError(f"unknown side {side!r}")

        conn.execute('UPDATE strategy_state SET cash_budget = ?, updated_at = ? WHERE strategy = ?',
                     (cash, now, strategy))
    return realized


def adopt_position(strategy, symbol, qty, avg_entry_price, db_path=None):
    """
    Records a position that already exists at the broker as owned by a
    strategy (used once when migrating an old database). Debits the cash
    budget at the entry price so equity stays consistent.
    """
    apply_fill(strategy, symbol, "buy", qty, avg_entry_price, db_path)


# ---------------------------------------------------------------------------
# Equity and drawdown
# ---------------------------------------------------------------------------

def strategy_equity(strategy, mark_prices, db_path=None):
    """
    Strategy equity = own cash budget + own positions at `mark_prices`.
    mark_prices: {symbol: price}. Symbols without a mark are valued at their
    average entry price (conservative when data is missing rather than 0,
    which would fake a drawdown; see audit issue C-03).
    Returns (equity, positions_value, missing_marks).
    """
    state = get_strategy_state(strategy, db_path)
    positions = list_positions(strategy, db_path)
    value = 0.0
    missing = []
    for symbol, (qty, avg, _) in positions.items():
        px = mark_prices.get(symbol)
        if px is None or not (px > 0):
            missing.append(symbol)
            px = avg
        value += qty * px
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
# Orders
# ---------------------------------------------------------------------------

# "submitting" is included: an order row in that state means the process may
# have died between sending the order and recording the broker id, so it must
# be reconciled (by client_order_id), never ignored (second-pass S-01).
OPEN_ORDER_STATUSES = ("submitting", "new", "accepted", "pending_new", "partially_filled",
                       "accepted_for_bidding", "pending", "held", "unresolved", "awaiting_fill_price")


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
    placeholders = ",".join("?" for _ in OPEN_ORDER_STATUSES)
    cols = 'id, symbol, side, requested_qty, broker_order_id, status, filled_qty, client_order_id, submitted_at'
    with db_connection(db_path) as conn:
        if symbol is None:
            rows = conn.execute(f'SELECT {cols} FROM orders WHERE strategy = ? AND status IN ({placeholders})',
                                (strategy, *OPEN_ORDER_STATUSES)).fetchall()
        else:
            rows = conn.execute(f'SELECT {cols} FROM orders WHERE strategy = ? AND symbol = ? AND status IN ({placeholders})',
                                (strategy, symbol, *OPEN_ORDER_STATUSES)).fetchall()
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
