import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone
from core.config import (CAPITAL, STRATEGY_COLORS, HEARTBEAT_STALE_SECONDS,MAX_DRAWDOWN, WARNING_DRAWDOWN)
from core.logger import get_trades, get_heartbeat
from metrics.risk import compute_all_metrics
from paper_trading.alpaca_paper import get_api
#https://emojidb.org/dashboard-emojis

st.set_page_config(page_title="TradingSystem Control Panel", page_icon= "🌐", layout="wide",initial_sidebar_state="collapsed")

STRATEGIES=["stable","risky1","risky2"]

def get_heartbeat_status(strategy):
    """
    This will return the running, paused or disconnected status of the bots based on most recent 'heartbeat'
    """
    row=get_heartbeat(strategy)
    if row is None:
        return "DISCONNECTED"
    last_seen_str, status=row
    last_seen=datetime.fromisoformat(last_seen_str).replace(tzinfo=timezone.utc)
    now=datetime.now(timezone.utc)
    seconds_ago=(now-last_seen).total_seconds()
    if seconds_ago>HEARTBEAT_STALE_SECONDS:
        return "DISCONNECTED"
    return status

def heartbeat_indicator (status):
    """This will return a colored image for heartbeat status"""
    if status=="RUNNING":
        return "🟢 Running Good"
    elif status == "PAUSED":
        return "🟡 Bot Paused"
    else:
        return "🔴 Bot Disconnected"
    
def get_all_trades():
    """Returns all trades through df (dataframe)"""
    all_trades=[]
    for s in STRATEGIES:
        trades=get_trades(s)
        if trades:
            all_trades.extend(trades)
    if not all_trades:
        return pd.DataFrame()
    df=pd.DataFrame(all_trades,columns=['id','timestamp','strategy','asset','action','price','quantity','pnl','reason'])
    df['timestamp']=pd.to_datetime(df['timestamp'])
    df=df.sort_values('timestamp',ascending=False)
    return df


#tabs of dashboard 

tab1,tab2,tab3,tab4=st.tabs(["🌐 Overview","📈Performance","🛡️ Risk & Positions","📋Trades & Audit"])

with tab1:
    st.title("TradingSystem Control Panel")
    st.caption("This system was built by Shriyans Singh | Polygon.io+Alpaca+ XGBoost+ PPO |btw you should hire me :D")

    st.subheader("Portfolio")
    try:
        api=get_api("stable")
        account=api.get_account()
        portfolio_value=float(account.portfolio_value)
        cash=float(account.cash)
        invested=portfolio_value-cash
        daily_pnl=float(account.equity)-float(account.last_equity)
        col1,col2,col3,col4=st.columns(4)
        with col1:
            st.metric("Portfolio Value", f"${portfolio_value:.2f}",delta=f"${daily_pnl:+,.2f} today")
        with col2:
            st.metric("Cash Available", f"${cash:,.2f}")
        with col3:
            st.metric("Invested",f"${invested:,.2f}")
        with col4:
            if portfolio_value>0:
                p=((invested/portfolio_value)*100)
            else:
                p=0
            st.metric("Exposure", f"{p:.1f}%")
    except Exception as e:
        st.warning(f"The portfolio data was unable to be fetched: {e}")
    
    st.subheader("System Status")
    cols=st.columns(3)
    for i, strategy in enumerate(STRATEGIES):
        status=get_heartbeat_status(strategy)
        color=STRATEGY_COLORS[strategy]
        with cols[i]:
            st.markdown(f"**{strategy.upper()}**")
            st.markdown(heartbeat_indicator(status))




                    #leaderboard time to compare strategies
    st.subheader("Strategy Comparision LeaderBoard")
    rows=[]
    for strategy in STRATEGIES:
        trades=get_trades(strategy)
        status=get_heartbeat_status(strategy)
        if not trades:
            rows.append({"Strategy":strategy.upper(),"Total Trades": 0,"Sharpe Ratio":"N/A","Sortino Ratio":"N/A","Max Drawdown":"N/A","Win Rate":"N/A","Status":status})

        else:
            df_t=pd.DataFrame(trades, columns=['id','timestamp','strategy','asset','action','price','quantity','pnl','reason'])
            closed = df_t[df_t['action'] == 'SELL'].copy()
            closed['pnl'] = pd.to_numeric(closed['pnl'], errors='coerce')
            closed = closed[closed['pnl'].notna()]
            # per-trade returns as a fraction of strategy capital (metrics expect fractions, not dollars)
            returns = (closed['pnl'] / CAPITAL[strategy]).reset_index(drop=True)
            if returns.empty:
                returns = pd.Series([0.0])
            metrics =compute_all_metrics(returns)
            rows.append({ 
                "Strategy": strategy.upper(),
                "Total Trades":len(df_t),
                "Sharpe Ratio":f"{metrics['sharpe_ratio']:.2f}",
                "Sortino Ratio":f"{metrics['sortino_ratio']:.2f}",
                "Max Drawdown":f"{metrics['max_drawdown']:.2%}",
                "Win Rate":f"{metrics['win_loss_ratio']:.2%}",
                "Status":status
            })
    leaderboard = pd.DataFrame(rows).set_index("Strategy")
    st.dataframe(leaderboard, use_container_width=True)



    #tab1 complete, tab2 startingn
with tab2:
    st.title("Performance")
    df_all = get_all_trades()

    if df_all.empty:
        st.info("No performance history available yet — waiting for first trades")
    else:
        # Equity Curve
        st.subheader("Equity Curve")
        fig_equity = go.Figure()
        for strategy in STRATEGIES:
            df_s = df_all[df_all['strategy'] == strategy].copy()
            if df_s.empty:
                continue
            df_s = df_s.sort_values('timestamp')
            df_s['pnl'] = pd.to_numeric(df_s['pnl'], errors='coerce').fillna(0)
            df_s['cumulative_pnl'] = df_s['pnl'].cumsum()
            fig_equity.add_trace(go.Scatter(
                x=df_s['timestamp'],
                y=df_s['cumulative_pnl'],
                name=strategy.upper(),
                line=dict(color=STRATEGY_COLORS[strategy], width=2)
            ))
        fig_equity.update_layout(
            xaxis_title="Date",
            yaxis_title="Cumulative PnL ($)",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_equity, use_container_width=True)

        # Drawdown Chart
        st.subheader("Drawdown Chart")
        fig_dd = go.Figure()
        for strategy in STRATEGIES:
            df_s = df_all[df_all['strategy'] == strategy].copy()
            if df_s.empty:
                continue
            df_s = df_s.sort_values('timestamp')
            df_s['pnl'] = pd.to_numeric(df_s['pnl'], errors='coerce').fillna(0)
            df_s['cumulative_pnl'] = df_s['pnl'].cumsum()
            peak = df_s['cumulative_pnl'].cummax()
            df_s['drawdown'] = (df_s['cumulative_pnl'] - peak) / (peak.abs() + 1e-10)
            fig_dd.add_trace(go.Scatter(
                x=df_s['timestamp'],
                y=df_s['drawdown'],
                name=strategy.upper(),
                fill='tozeroy',
                line=dict(color=STRATEGY_COLORS[strategy], width=1)
            ))
        fig_dd.update_layout(
            xaxis_title="Date",
            yaxis_title="Drawdown %",
            yaxis_tickformat=".1%",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_dd, use_container_width=True)

with tab3:
    st.title("Risk & Positions")
    try:
        from core import portfolio
        api = get_api("stable")
        # Mark each strategy's OWN ledger with the broker's current prices.
        broker_px = {}
        try:
            for p in api.list_positions():
                broker_px[p.symbol] = float(p.current_price)
        except Exception:
            pass
        # kill switches per strategy: drawdown from the STRATEGY's persisted peak
        # (the old panel compared ACCOUNT equity with each strategy's capital)
        st.subheader("Kill Switch Status (per strategy)")
        cols = st.columns(3)
        for i, strategy in enumerate(STRATEGIES):
            with cols[i]:
                state = portfolio.get_strategy_state(strategy)
                marks = {sym: broker_px.get(sym.replace("X:", ""), None) for sym in portfolio.list_positions(strategy)}
                equity, pos_value, _ = portfolio.strategy_equity(strategy, {k: v for k, v in marks.items() if v})
                peak = max(state["peak_equity"], equity)
                drawdown_pct = (equity - peak) / peak if peak else 0.0
                warning = WARNING_DRAWDOWN[strategy]
                critical = MAX_DRAWDOWN[strategy]
                progress = min(drawdown_pct / critical, 1.0) if critical != 0 else 0
                progress = max(progress, 0.0)

                if state["halted"]:
                    label = "⛔ HALTED (kill switch)"
                elif drawdown_pct <= critical:
                    label = "🔴CRITICAL"
                elif drawdown_pct <= warning:
                    label = "🟡WARNING"
                else:
                    label = "🟢 SAFE"

                st.markdown(f"**{strategy.upper()}** — {label}")
                st.markdown(f"Equity: `${equity:,.2f}` (cash `${state['cash_budget']:,.2f}` + positions `${pos_value:,.2f}`)")
                st.markdown(f"Drawdown from peak `${peak:,.2f}`: `{drawdown_pct:.2%}`")
                st.progress(progress)

        st.subheader("Strategy Ledger (who owns what)")
        ledger_rows = []
        for strategy in STRATEGIES:
            for sym, (qty, avg, opened) in portfolio.list_positions(strategy).items():
                px = broker_px.get(sym.replace("X:", ""))
                ledger_rows.append({"Strategy": strategy.upper(), "Symbol": sym, "Qty": qty,
                                    "Avg Entry": f"${avg:,.2f}", "Mark": f"${px:,.2f}" if px else "n/a",
                                    "Unrealized": f"${qty * (px - avg):+,.2f}" if px else "n/a", "Opened": opened})
        if ledger_rows:
            st.dataframe(pd.DataFrame(ledger_rows), use_container_width=True)
        else:
            st.info("No strategy-owned positions")

        #Live pos (account-wide, as Alpaca sees it)
        st.subheader("Broker Positions (account-wide)")
        positions = api.list_positions()
        if not positions:
            st.info("No active positions")
        else:
            pos_rows = []
            for p in positions:
                pos_rows.append({
                    "Symbol"        : p.symbol,
                    "Qty"           : float(p.qty),
                    "Market Value"  : f"${float(p.market_value):,.2f}",
                    "Unrealized PnL": f"${float(p.unrealized_pl):+,.2f}",
                    "PnL %"         : f"{float(p.unrealized_plpc)*100:+.2f}%",
                    "Entry Price"   : f"${float(p.avg_entry_price):,.2f}",
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True)

    except Exception as e:
        st.warning(f"Could not fetch risk data: {e}")



with tab4:
    st.title("Trades & Audit")
    df_all = get_all_trades()

    if df_all.empty:
        st.info("No trades executed yet")
    else:
        #filters for user
        col1, col2, col3 = st.columns(3)
        with col1:
            strategy_filter = st.selectbox(
                "Filter by Strategy",
                ["All"] + [s.upper() for s in STRATEGIES]
            )
        with col2:
            action_filter = st.selectbox(
                "Filter by Action",
                ["All", "BUY", "SELL", "KILL_SWITCH"]
            )
        with col3:
            asset_filter = st.text_input("Filter by Asset", "")

        # Apply filters
        df_filtered = df_all.copy()
        if strategy_filter != "All":
            df_filtered = df_filtered[
                df_filtered['strategy'] == strategy_filter.lower()
            ]
        if action_filter != "All":
            df_filtered = df_filtered[df_filtered['action'] == action_filter]
        if asset_filter:
            df_filtered = df_filtered[
                df_filtered['asset'].str.contains(asset_filter.upper())
            ]

        # trade logs
        st.subheader(f"Trade Log ({len(df_filtered)} trades)")
        display_df = df_filtered.drop(columns=['id']).reset_index(drop=True)
        st.dataframe(display_df, use_container_width=True)

        # win/loss
        st.subheader("Win/Loss Distribution")
        col1, col2 = st.columns(2)
        with col1:
            action_counts = df_all['action'].value_counts()
            fig_pie = go.Figure(go.Pie(
                labels=action_counts.index,
                values=action_counts.values,
                hole=0.4
            ))
            fig_pie.update_layout(title="Trade Actions")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            pnl_data = df_all[df_all['pnl'].notna()]
            wins = (pnl_data['pnl'] > 0).sum()
            losses = (pnl_data['pnl'] < 0).sum()
            fig_wl = go.Figure(go.Bar(
                x=["Wins", "Losses"],
                y=[wins, losses],
                marker_color=["#2ecc71", "#e74c3c"]
            ))
            fig_wl.update_layout(title="Win/Loss Count")
            st.plotly_chart(fig_wl, use_container_width=True)

# ending notes
st.divider()
st.caption("TradingSystem | Built by Shriyans Singh | github.com/Shri-2006 | (p.s. you should hire me and talk to me)")