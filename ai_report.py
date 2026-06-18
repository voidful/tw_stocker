#!/usr/bin/env python3
"""
AI 台股實戰區間交易系統 v2 (Event-Driven Quantitative Trading Pipeline)

完整管線：資料下載 → 動態 Universe → AI 特徵排名 → 事件驅動回測 → 風險分析 → HTML 報表

v2 改進：
- Entry 改為 t+1 open（對齊實盤）
- 動態 Liquid Universe（全 TWSE，按流動性篩選 Top-N）
- Top-K 選股取代固定 threshold
- Equity-based position sizing
- ATR 自適應 TP/SL
- 台股交易成本（手續費 + 證交稅）
- 完整風險指標（Sharpe/Sortino/MaxDD/Calmar）
- 0050 Benchmark 對比
- exchange_calendars 精確交易日

使用方式：
    python ai_report.py
    python ai_report.py --tickers 2330 2317 2454 --static-pool
    python ai_report.py --universe-size 100 --top-k 5
"""

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

# 確保 strategy/ 可被 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy.ai_strategy import fetch_panel_data, engineer_features, build_liquid_universe
from strategy.event_backtest import EventDrivenBacktester
from strategy.evaluation import slice_evaluation_window
from strategy.risk_metrics import compute_risk_metrics, format_metrics_summary
from strategy.benchmark import fetch_benchmark, equal_weight_benchmark, compute_excess_return
from strategy.institutional_flow import build_inst_flow_df, get_inst_flow_for_signals, fetch_inst_rankings
from strategy.news_sentiment import get_news_sentiment_for_signals

# 嘗試載入 exchange_calendars
try:
    import exchange_calendars as xcals
    TW_CALENDAR = xcals.get_calendar('XTAI')
    HAS_EXCHANGE_CAL = True
except ImportError:
    TW_CALENDAR = None
    HAS_EXCHANGE_CAL = False
    print("⚠️ exchange_calendars 未安裝，最晚出場日將使用近似計算")


# ==========================================
# 預設股池：熱門權值、AI、航運、金融股（靜態池模式用）
# ==========================================
DEFAULT_TICKERS = [
    '2330', '2317', '2454', '2308', '2881',
    '2603', '3231', '3481', '2382', '2609',
    '2891', '1519', '2379', '2303',
]

# 擴展股池：全 TWSE 主要個股（動態 universe 模式用）
# 包含上市 ETF、權值股、中型股，約 200 檔候選池
EXTENDED_TICKERS = [
    # 半導體
    '2330', '2454', '2303', '3711', '2379', '6770', '3034', '2449',
    '5274', '3529', '2408', '3443', '3035', '6415', '6525', '3661',
    '3037', '2344', '6547',
    # 電子
    '2317', '2382', '2308', '2301', '2357', '2376', '2395', '3231',
    '2474', '2353', '3481', '3017', '2345', '2383', '2356', '3044',
    '2327', '3036', '2324', '2377', '2385', '2360', '2404',
    '2412', '2459', '2458', '3045', '6505', '3023',
    '3706', '3533', '2368', '4904', '4938', '6669',
    # 金融
    '2881', '2882', '2884', '2886', '2887', '2891', '2892',
    '2880', '2883', '2885', '2888', '2889', '2890', '5880', '5876',
    '2801', '2834', '2838', '2845', '2855', '2867', '2897',
    # 傳產
    '1301', '1303', '1326', '2002', '1101', '1102', '2912',
    '1216', '2207', '9904', '1402', '9910', '1605', '2603',
    '2609', '2615', '1519', '2606', '6005',
    # 航運/觀光
    '2618', '2610', '2605', '2634', '2637',
    # 生技
    '4142', '1760', '6446', '1707', '4743',
    # 其他
    '9945', '8454', '1504', '2105', '2201', '2204',
    '5871', '6116', '6285', '3149', '6239',
]

# 去重
EXTENDED_TICKERS = list(dict.fromkeys(EXTENDED_TICKERS))


def get_next_n_trading_days(from_date, n_days):
    """
    使用 exchange_calendars 計算從 from_date 起的第 n 個交易日。

    Parameters
    ----------
    from_date : datetime-like
        起始日期
    n_days : int
        往後幾個交易日

    Returns
    -------
    target_date : str
        目標日期 (YYYY-MM-DD)
    """
    if HAS_EXCHANGE_CAL and TW_CALENDAR is not None:
        try:
            from_ts = pd.Timestamp(from_date)
            # 取得足夠長的交易日列表
            end_search = from_ts + pd.Timedelta(days=n_days * 2 + 30)
            sessions = TW_CALENDAR.sessions_in_range(from_ts, end_search)
            if len(sessions) > n_days:
                return sessions[n_days].strftime('%Y-%m-%d')
        except Exception:
            pass

    # Fallback: 用 1.4 倍近似
    approx_date = pd.Timestamp(from_date) + timedelta(days=int(n_days * 1.4))
    return approx_date.strftime('%Y-%m-%d')


def _build_inst_section():
    """
    建立三大法人籌碼動態 HTML section。
    從 tw-institutional-stocker 抓取 20 日持股變化排名，呈現買超/賣超 Top-15。
    """
    try:
        up_list = fetch_inst_rankings(20, 'up') or []
        down_list = fetch_inst_rankings(20, 'down') or []
    except Exception:
        return '<h2>🏛️ 三大法人籌碼動態</h2><p class="section-note">⚠️ 籌碼數據暫時無法取得</p>'

    if not up_list and not down_list:
        return '<h2>🏛️ 三大法人籌碼動態</h2><p class="section-note">⚠️ 無籌碼數據</p>'

    # 過濾 ETF（code 5 碼以上通常為 ETF）
    up_stocks = [x for x in up_list if len(x.get('code', '')) == 4][:15]
    down_stocks = [x for x in down_list if len(x.get('code', '')) == 4][:15]

    # 買超表
    buy_rows = ""
    for i, item in enumerate(up_stocks, 1):
        code = item.get('code', '')
        name = item.get('name', '')
        change = item.get('change', 0.0)
        ratio = item.get('three_inst_ratio', 0.0)
        bar_width = min(change * 8, 100)
        buy_rows += (
            f'<tr>'
            f'<td style="text-align:center;color:#888;">{i}</td>'
            f'<td><b>{code}</b> {name}</td>'
            f'<td style="text-align:right;color:#00ff00;font-weight:bold;">+{change:.1f}%</td>'
            f'<td style="text-align:right;">{ratio:.1f}%</td>'
            f'<td><div style="background:linear-gradient(90deg,#00ff0044 {bar_width}%,transparent {bar_width}%);'
            f'height:18px;border-radius:3px;"></div></td>'
            f'</tr>\n'
        )

    # 賣超表
    sell_rows = ""
    for i, item in enumerate(down_stocks, 1):
        code = item.get('code', '')
        name = item.get('name', '')
        change = item.get('change', 0.0)
        ratio = item.get('three_inst_ratio', 0.0)
        bar_width = min(abs(change) * 8, 100)
        sell_rows += (
            f'<tr>'
            f'<td style="text-align:center;color:#888;">{i}</td>'
            f'<td><b>{code}</b> {name}</td>'
            f'<td style="text-align:right;color:#ff4444;font-weight:bold;">-{abs(change):.1f}%</td>'
            f'<td style="text-align:right;">{ratio:.1f}%</td>'
            f'<td><div style="background:linear-gradient(90deg,#ff444444 {bar_width}%,transparent {bar_width}%);'
            f'height:18px;border-radius:3px;"></div></td>'
            f'</tr>\n'
        )

    html = f"""
    <h2>🏛️ 三大法人籌碼動態</h2>
    <p class="section-note">
        近 20 日三大法人（外資+投信+自營商）持股比重變化排名。Data: <a href="https://github.com/voidful/tw-institutional-stocker" style="color:#4FC3F7;">tw-institutional-stocker</a>
    </p>
    <div class="split-grid">
        <div>
            <h3 style="color:#00ff00;margin-bottom:8px;">🟢 法人買超 Top-15（20日）</h3>
            <table style="width:100%;">
                <thead><tr>
                    <th style="width:30px;">#</th>
                    <th>股票</th>
                    <th style="text-align:right;">變化</th>
                    <th style="text-align:right;">持股</th>
                    <th style="width:80px;">幅度</th>
                </tr></thead>
                <tbody>{buy_rows}</tbody>
            </table>
        </div>
        <div>
            <h3 style="color:#ff4444;margin-bottom:8px;">🔴 法人賣超 Top-15（20日）</h3>
            <table style="width:100%;">
                <thead><tr>
                    <th style="width:30px;">#</th>
                    <th>股票</th>
                    <th style="text-align:right;">變化</th>
                    <th style="text-align:right;">持股</th>
                    <th style="width:80px;">幅度</th>
                </tr></thead>
                <tbody>{sell_rows}</tbody>
            </table>
        </div>
    </div>
"""
    return html



def generate_report(trades_df, equity_df, total_score, close_df, config,
                    metrics, benchmark_equity=None, ew_equity=None,
                    benchmark2_equity=None,
                    high_df=None, low_df=None, show_inst=True):
    """
    產出 AI 交易計畫 HTML 報表與資金曲線圖（v2 完整版）。

    Parameters
    ----------
    high_df, low_df : pd.DataFrame, optional
        最高/最低價矩陣，用於精確 ATR 計算（對齊回測引擎）。
    """
    print("📊 產出 AI 交易計畫與績效報表...")

    tp_pct = config['tp_pct']
    sl_pct = config['sl_pct']
    max_hold_days = config['max_hold_days']
    initial_capital = config['initial_capital']
    tp_sl_mode = config.get('tp_sl_mode', 'atr')
    top_k = config.get('top_k', 3)

    total_ret = metrics['total_return'] * 100

    # === 計算精確 ATR（與回測引擎同公式） ===
    def _compute_display_atr(close_s, high_s=None, low_s=None, period=20):
        """計算單檔股票的 ATR，對齊 EventDrivenBacktester._compute_atr"""
        if high_s is not None and low_s is not None:
            prev_close = close_s.shift(1)
            tr1 = high_s - low_s
            tr2 = (high_s - prev_close).abs()
            tr3 = (low_s - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        else:
            # fallback: 用收盤價百分比變動
            tr = close_s.pct_change().abs() * close_s
        return tr.rolling(period).mean().iloc[-1]

    # === 繪製資金曲線（含 Benchmark） ===
    plt.style.use('default')
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                              gridspec_kw={'hspace': 0.25})
    fig.patch.set_facecolor('#f5f5f7')

    ax1, ax2 = axes
    for ax in axes:
        ax.set_facecolor('#ffffff')

    # 主圖：策略 + Benchmark
    ax1.plot(equity_df.index, equity_df['Equity'], color='#007aff', lw=2, label='Strategy')
    ax1.axhline(initial_capital, color='#8e8e93', linestyle='--', alpha=0.6, label='Initial Capital')
    ax1.fill_between(equity_df.index, initial_capital, equity_df['Equity'],
                     where=equity_df['Equity'] >= initial_capital, alpha=0.10, color='#007aff')
    ax1.fill_between(equity_df.index, initial_capital, equity_df['Equity'],
                     where=equity_df['Equity'] < initial_capital, alpha=0.12, color='#ff3b30')

    if benchmark_equity is not None and len(benchmark_equity) > 0:
        # 將 benchmark 縮放到同一起始資金
        common_idx = equity_df.index.intersection(benchmark_equity.index)
        if len(common_idx) > 0:
            bench_scaled = benchmark_equity.loc[common_idx] * initial_capital
            ax1.plot(common_idx, bench_scaled, color='#ff9500', lw=1.5, alpha=0.85,
                     label='0050 Buy & Hold', linestyle='--')

    if ew_equity is not None and len(ew_equity) > 0:
        common_idx = equity_df.index.intersection(ew_equity.index)
        if len(common_idx) > 0:
            ew_scaled = ew_equity.loc[common_idx] * initial_capital
            ax1.plot(common_idx, ew_scaled, color='#af52de', lw=1.2, alpha=0.65,
                     label='Equal-Weight', linestyle=':')

    if benchmark2_equity is not None and len(benchmark2_equity) > 0:
        common_idx = equity_df.index.intersection(benchmark2_equity.index)
        if len(common_idx) > 0:
            bench2_scaled = benchmark2_equity.loc[common_idx] * initial_capital
            ax1.plot(common_idx, bench2_scaled, color='#34c759', lw=1.5, alpha=0.85,
                     label='00981A Buy & Hold', linestyle='-.')

    mode_label = f"ATR×{config.get('tp_atr_mult', 3)}/{config.get('sl_atr_mult', 1.5)}" \
        if tp_sl_mode == 'atr' else f"TP +{tp_pct*100:.0f}% / SL -{sl_pct*100:.0f}%"
    ax1.set_title(f'AI Quant v8  |  {mode_label}  |  Top-{top_k}  |  Hold ≤{max_hold_days}D',
                  fontweight='bold', fontsize=14, color='#1d1d1f')
    ax1.set_ylabel('Portfolio Value (TWD)', fontsize=11, color='#1d1d1f')
    ax1.tick_params(colors='#1d1d1f')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(alpha=0.08, color='#000000')
    for spine in ax1.spines.values():
        spine.set_color('#d2d2d7')

    # 子圖：Drawdown
    equity = equity_df['Equity']
    cummax = equity.cummax()
    drawdown = (equity / cummax - 1) * 100
    ax2.fill_between(drawdown.index, 0, drawdown, color='#ff3b30', alpha=0.35)
    ax2.plot(drawdown.index, drawdown, color='#ff3b30', lw=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=10, color='#1d1d1f')
    ax2.set_xlabel('')
    ax2.tick_params(colors='#1d1d1f')
    ax2.grid(alpha=0.08, color='#000000')
    ax2.set_ylim(drawdown.min() * 1.2, 2)
    for spine in ax2.spines.values():
        spine.set_color('#d2d2d7')

    fig.tight_layout()
    fig.savefig('backtest_chart.png', dpi=150, bbox_inches='tight', facecolor='#f5f5f7')
    plt.close(fig)
    print("   📈 資金曲線已存為 backtest_chart.png")

    # === 建立 per-stock 歷史績效統計 ===
    stock_stats = {}
    if not trades_df.empty:
        for ticker in trades_df['Ticker'].unique():
            t = trades_df[trades_df['Ticker'] == ticker]
            wins = len(t[t['Return_Pct'] > 0])
            total = len(t)
            stock_stats[ticker] = {
                'trades': total,
                'win_rate': wins / total * 100 if total > 0 else 0,
                'avg_return': t['Return_Pct'].mean() * 100,
                'total_return': t['Return_Pct'].sum() * 100,
            }

    # === 今日交易計畫（嚴格對齊回測引擎的選股邏輯） ===
    latest_date = total_score.index[-1]
    today_scores = total_score.loc[latest_date].dropna().sort_values(ascending=False)
    threshold = config.get('threshold', 2.0)
    ma_period = 60  # 對齊回測中的 MA 期數

    # Step 1: 篩選候選（score >= threshold + close > MA）
    candidates = []
    filtered_out = []
    for ticker, score in today_scores.items():
        price = close_df[ticker].iloc[-1] if ticker in close_df.columns else np.nan
        if pd.isna(price) or price <= 0:
            continue

        ma_val = close_df[ticker].rolling(ma_period).mean().iloc[-1] if ticker in close_df.columns else np.nan
        above_ma = (not pd.isna(ma_val)) and (price > ma_val)

        if score >= threshold and above_ma:
            candidates.append((ticker, score, price))
        elif score >= threshold:
            filtered_out.append((ticker, score, price, '低於 MA'))
        else:
            filtered_out.append((ticker, score, price, '評分不足'))

    # Step 2: 嚴格取 Top-K（對齊回測邏輯）
    selected = candidates[:top_k]
    not_selected = candidates[top_k:]

    trading_plan_rows = ""

    # 籌碼 + 新聞標注（always on）
    all_tickers = [t for t, _, _ in selected] + [t for t, _, _ in not_selected[:5]]
    inst_data = {}
    news_data = {}
    try:
        inst_data = get_inst_flow_for_signals(all_tickers)
    except Exception:
        inst_data = {}
    try:
        news_data = get_news_sentiment_for_signals(all_tickers)
    except Exception:
        news_data = {}

    # 籌碼動態 HTML section
    inst_section_html = _build_inst_section()

    # 顯示 Top-K 建議買進
    orders = []
    for rank, (ticker, score, price) in enumerate(selected, 1):
        order_valid = True
        order_atr = None  # 供 paper trading 複製回測 gap filter
        time_exit = get_next_n_trading_days(latest_date, max_hold_days)
        # 使用精確 ATR（與回測引擎同公式）
        if tp_sl_mode == 'atr':
            high_s = high_df[ticker] if (high_df is not None and ticker in high_df.columns) else None
            low_s = low_df[ticker] if (low_df is not None and ticker in low_df.columns) else None
            atr_val = _compute_display_atr(close_df[ticker], high_s, low_s)

            if not pd.isna(atr_val) and atr_val > 0:
                order_atr = float(atr_val)
                tp_price = price + atr_val * config.get('tp_atr_mult', 3.0)
                sl_price = price - atr_val * config.get('sl_atr_mult', 2.0)
                # Sanity checks
                if sl_price <= 0:
                    sl_price = price * 0.85  # fallback: -15%
                tp_pct_display = (tp_price / price - 1) * 100
                sl_pct_display = (1 - sl_price / price) * 100
                # 合理性檢查
                if tp_pct_display > 50 or sl_pct_display > 50:
                    plan = '<span style="color:#ff4444">⚠️ ATR 異常，信號無效</span>'
                    order_valid = False
                else:
                    plan = (f'<b>停利:</b> <span style="color:#00ff00">{tp_price:.1f}</span>'
                            f' (+{tp_pct_display:.1f}%) '
                            f'<br><b>停損:</b> <span style="color:#ff4444">{sl_price:.1f}</span>'
                            f' (-{sl_pct_display:.1f}%) '
                            f'<br><b>最晚出場:</b> {time_exit}')
            else:
                tp_price = price * (1 + tp_pct)
                sl_price = price * (1 - sl_pct)
                plan = (f'<b>停利:</b> <span style="color:#00ff00">{tp_price:.1f}</span>'
                        f' (+{tp_pct*100:.1f}%) '
                        f'<br><b>停損:</b> <span style="color:#ff4444">{sl_price:.1f}</span>'
                        f' (-{sl_pct*100:.1f}%) '
                        f'<br><b>最晚出場:</b> {time_exit}')
        else:
            tp_price = price * (1 + tp_pct)
            sl_price = price * (1 - sl_pct)
            plan = (f'<b>停利:</b> <span style="color:#00ff00">{tp_price:.1f}</span>'
                    f' (+{tp_pct*100:.1f}%) '
                    f'<br><b>停損:</b> <span style="color:#ff4444">{sl_price:.1f}</span>'
                    f' (-{sl_pct*100:.1f}%) '
                    f'<br><b>最晚出場:</b> {time_exit}')

        status = f'<span style="color:#00ff00; font-weight:bold;">🟢 建議買進 #{rank}</span>'

        # Per-stock 歷史績效
        ss = stock_stats.get(ticker, None)
        if ss and ss['trades'] >= 2:
            wr_color = '#00ff00' if ss['win_rate'] >= 50 else '#ff4444'
            ar_color = '#00ff00' if ss['avg_return'] > 0 else '#ff4444'
            hist_badge = (
                f'<span style="font-size:0.72rem; color:#888;">'
                f'歷史 <b>{ss["trades"]}</b>筆 | '
                f'勝率 <b style="color:{wr_color}">{ss["win_rate"]:.0f}%</b> | '
                f'均報酬 <b style="color:{ar_color}">{ss["avg_return"]:+.1f}%</b>'
                f'</span>'
            )
        else:
            hist_badge = '<span style="font-size:0.72rem; color:#555;">歷史資料不足</span>'

        # 籌碼 + 新聞標注
        idata = inst_data.get(ticker, {})
        ndata = news_data.get(ticker, {})
        inst_change = idata.get('change', 0.0)
        inst_label = idata.get('label', '⚪ 無資料')
        news_label = ndata.get('label', '⚪ 中性')
        inst_color = '#00ff00' if inst_change > 2 else '#ff4444' if inst_change < -2 else '#ffab00' if abs(inst_change) > 0.5 else '#888'
        inst_badge = (
            f'<td><span style="font-size:0.78rem;">{inst_label}'
            f'<br><span style="color:{inst_color}; font-weight:bold;">{inst_change:+.1f}%</span></span></td>'
            f'<td><span style="font-size:0.78rem;">{news_label}</span></td>'
        )

        trading_plan_rows += (
            f'<tr><td>{ticker}</td><td>{score:.2f}</td>'
            f'<td>{price:.1f}</td><td>{status}</td><td>{plan}</td>'
            f'<td>{hist_badge}</td>{inst_badge}</tr>\n'
        )
        if order_valid:
            orders.append({
                'signal_date': latest_date.strftime('%Y-%m-%d'),
                'execution_date': get_next_n_trading_days(latest_date, 1),
                'ticker': ticker,
                'side': 'buy',
                'rank': rank,
                'score': round(float(score), 4),
                'model_entry_ref': 'next_open',
                'reference_close': round(float(price), 4),
                'limit_price': round(float(price), 4),
                'tp_price': round(float(tp_price), 4),
                'sl_price': round(float(sl_price), 4),
                'atr': round(order_atr, 4) if order_atr else None,
                'gap_limit_atr': float(config.get('gap_filter_effective',
                                                  config.get('gap_filter', 1.5))),
                'max_hold_days': int(max_hold_days),
                'time_exit': time_exit,
                'model_version': 'v8.5',
            })

    # 顯示未被選入的候選（排名 > Top-K）
    for ticker, score, price in not_selected[:5]:
        status = '<span style="color:#ffab00">🟡 候選 (超出 Top-K)</span>'
        ss = stock_stats.get(ticker, None)
        hist_badge = '<span style="font-size:0.72rem; color:#555;">-</span>'
        if ss and ss['trades'] >= 2:
            wr_color = '#00ff00' if ss['win_rate'] >= 50 else '#ff4444'
            hist_badge = f'<span style="font-size:0.72rem; color:#888;">勝率 <b style="color:{wr_color}">{ss["win_rate"]:.0f}%</b></span>'

        idata = inst_data.get(ticker, {})
        ndata = news_data.get(ticker, {})
        inst_change = idata.get('change', 0.0)
        inst_color = '#00ff00' if inst_change > 2 else '#ff4444' if inst_change < -2 else '#888'
        inst_badge = (
            f'<td><span style="font-size:0.72rem;">{idata.get("label", "⚪")}'
            f'<br><span style="color:{inst_color}">{inst_change:+.1f}%</span></span></td>'
            f'<td><span style="font-size:0.72rem;">{ndata.get("label", "⚪")}</span></td>'
        )

        trading_plan_rows += (
            f'<tr style="opacity:0.6"><td>{ticker}</td><td>{score:.2f}</td>'
            f'<td>{price:.1f}</td><td>{status}</td><td>-</td>'
            f'<td>{hist_badge}</td>{inst_badge}</tr>\n'
        )

    # 顯示被過濾掉的（前 5 筆）
    for ticker, score, price, reason in filtered_out[:5]:
        status = f'<span style="color:#aaaaaa">⚪ 觀望 ({reason})</span>'
        price_str = f'{price:.1f}' if not pd.isna(price) else '-'
        inst_badge = '<td style="color:#555">-</td><td style="color:#555">-</td>'
        trading_plan_rows += (
            f'<tr style="opacity:0.4"><td>{ticker}</td><td>{score:.2f}</td>'
            f'<td>{price_str}</td><td>{status}</td><td>-</td>'
            f'<td>-</td>{inst_badge}</tr>\n'
        )

    # === 歷史交易紀錄（最近 20 筆）===
    trade_history_rows = ""
    if not trades_df.empty:
        for _, row in trades_df.tail(20).iloc[::-1].iterrows():
            color = "#00ff00" if row['Return_Pct'] > 0 else "#ff4444"
            trade_history_rows += (
                f'<tr>'
                f'<td>{row["Ticker"]}</td>'
                f'<td>{row["Entry_Date"]}</td>'
                f'<td>{row["Exit_Date"]}</td>'
                f'<td>{row["Entry_Price"]:.1f}</td>'
                f'<td>{row["Exit_Price"]:.1f}</td>'
                f'<td>{row["Reason"]}</td>'
                f'<td>{row["Days_Held"]}天</td>'
                f'<td style="color:{color}; font-weight:bold;">{row["Return_Pct"]*100:+.1f}%</td>'
                f'</tr>\n'
            )

    # === 出場原因統計 ===
    reason_stats_rows = ""
    if not trades_df.empty:
        reason_counts = trades_df['Reason'].value_counts()
        for reason, count in reason_counts.items():
            subset = trades_df[trades_df['Reason'] == reason]
            avg_ret = subset['Return_Pct'].mean() * 100
            reason_stats_rows += (
                f'<tr><td>{reason}</td><td>{count} 筆</td>'
                f'<td style="color:{"#00ff00" if avg_ret > 0 else "#ff4444"}">{avg_ret:+.2f}%</td></tr>\n'
            )

    # === 風險指標卡片 ===
    m = metrics
    total_ret_color = "#00ff00" if total_ret > 0 else "#ff4444"
    sharpe_color = "#00ff00" if m['sharpe'] > 0.5 else ("#ffab00" if m['sharpe'] > 0 else "#ff4444")
    dd_color = "#ff4444" if m['max_drawdown_pct'] < -0.15 else "#ffab00"

    # === Benchmark 數值比較 ===
    benchmark_stats_html = ""
    if benchmark_equity is not None and len(benchmark_equity) > 20:
        try:
            from strategy.risk_metrics import compute_risk_metrics as _crm
            bench_eq = pd.DataFrame({'Equity': benchmark_equity * initial_capital},
                                    index=benchmark_equity.index)
            bench_m = _crm(bench_eq, pd.DataFrame(), initial_capital)
            bm_ann = bench_m['ann_return'] * 100
            bm_vol = bench_m['ann_volatility'] * 100
            bm_mdd = bench_m['max_drawdown_pct'] * 100
            bm_sharpe = bench_m['sharpe']
            excess_ret = m['ann_return'] * 100 - bm_ann
            excess_color = "#00ff00" if excess_ret > 0 else "#ff4444"
            benchmark_stats_html = f"""
    <div class="stats">
        <div class="stat-card benchmark">
            <div class="label">0050 年化報酬</div>
            <div class="value">{bm_ann:+.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">0050 最大回撤</div>
            <div class="value">{bm_mdd:.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">0050 Sharpe</div>
            <div class="value">{bm_sharpe:.2f}</div>
        </div>
        <div class="stat-card" style="border-left-color:{excess_color}">
            <div class="label">超額年化報酬 (α)</div>
            <div class="value" style="color:{excess_color}">{excess_ret:+.1f}%</div>
        </div>
    </div>"""
        except Exception:
            pass

    # === 00981A Benchmark 比較 ===
    benchmark2_stats_html = ""
    if benchmark2_equity is not None and len(benchmark2_equity) > 20:
        try:
            from strategy.risk_metrics import compute_risk_metrics as _crm2
            bench2_eq = pd.DataFrame({'Equity': benchmark2_equity * initial_capital},
                                     index=benchmark2_equity.index)
            bench2_m = _crm2(bench2_eq, pd.DataFrame(), initial_capital)
            bm2_ann = bench2_m['ann_return'] * 100
            bm2_mdd = bench2_m['max_drawdown_pct'] * 100
            bm2_sharpe = bench2_m['sharpe']
            # 計算共存期間超額
            common_dates = equity_df.index.intersection(benchmark2_equity.index)
            if len(common_dates) > 20:
                strat_sub = equity_df.loc[common_dates, 'Equity']
                bench2_sub = benchmark2_equity.loc[common_dates] * initial_capital
                s_ret = (float(strat_sub.iloc[-1]) / float(strat_sub.iloc[0]) - 1) * 100
                b_ret = (float(bench2_sub.iloc[-1]) / float(bench2_sub.iloc[0]) - 1) * 100
                excess2 = s_ret - b_ret
                e2_color = "#00ff00" if excess2 > 0 else "#ff4444"
                benchmark2_stats_html = f"""
    <div class="stats">
        <div class="stat-card benchmark">
            <div class="label">00981A 年化報酬</div>
            <div class="value">{bm2_ann:+.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">00981A 最大回撤</div>
            <div class="value">{bm2_mdd:.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">00981A Sharpe</div>
            <div class="value">{bm2_sharpe:.2f}</div>
        </div>
        <div class="stat-card" style="border-left-color:{e2_color}">
            <div class="label">超額 vs 00981A (共存期)</div>
            <div class="value" style="color:{e2_color}">{excess2:+.1f}%</div>
        </div>
    </div>"""
        except Exception:
            pass

    # === 月度報酬熱圖 ===
    monthly_heatmap_html = ""
    if not trades_df.empty:
        try:
            eq = equity_df['Equity']
            monthly_ret = eq.resample('ME').last().pct_change().dropna()
            rows_html = ""
            for dt, ret in monthly_ret.items():
                ret_pct = ret * 100
                # 顏色映射：紅(-10%) → 黃(0%) → 綠(+10%)
                if ret_pct >= 0:
                    intensity = min(ret_pct / 10, 1.0)
                    bg = f"rgba(0,255,0,{intensity * 0.3:.2f})"
                    color = "#00ff00"
                else:
                    intensity = min(abs(ret_pct) / 10, 1.0)
                    bg = f"rgba(255,68,68,{intensity * 0.3:.2f})"
                    color = "#ff4444"
                rows_html += (
                    f'<tr style="background:{bg}">'
                    f'<td>{dt.strftime("%Y-%m")}</td>'
                    f'<td style="color:{color}; font-weight:bold">{ret_pct:+.1f}%</td>'
                    f'</tr>\n'
                )
            monthly_heatmap_html = f"""
    <table style="max-width:400px">
        <thead><tr><th>月份</th><th>月報酬</th></tr></thead>
        <tbody>
{rows_html}
        </tbody>
    </table>"""
        except Exception:
            pass

    # === 大盤環境診斷 ===
    market_context_html = ""
    try:
        if benchmark_equity is not None and len(benchmark_equity) > 20:
            mkt_latest = benchmark_equity.iloc[-1]
            mkt_5d = benchmark_equity.iloc[-5] if len(benchmark_equity) >= 5 else mkt_latest
            mkt_20d = benchmark_equity.iloc[-20] if len(benchmark_equity) >= 20 else mkt_latest
            mkt_60d = benchmark_equity.iloc[-60] if len(benchmark_equity) >= 60 else mkt_latest
            ret_5d = (mkt_latest / mkt_5d - 1) * 100
            ret_20d = (mkt_latest / mkt_20d - 1) * 100
            ret_60d = (mkt_latest / mkt_60d - 1) * 100

            # 判斷 regime
            mkt_ma60 = benchmark_equity.rolling(60).mean().iloc[-1] if len(benchmark_equity) >= 60 else None
            if mkt_ma60 is not None:
                regime = "🟢 多頭" if mkt_latest > mkt_ma60 else "🔴 空頭"
                regime_color = "#00ff00" if mkt_latest > mkt_ma60 else "#ff4444"
            else:
                regime = "⚪ 未知"
                regime_color = "#888"

            # 波動率
            mkt_vol = benchmark_equity.pct_change().tail(20).std() * (252**0.5) * 100

            market_context_html = f"""
    <div class="stats">
        <div class="stat-card" style="border-left-color:{regime_color}">
            <div class="label">大盤 Regime</div>
            <div class="value" style="color:{regime_color}">{regime}</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">0050 近 5 日</div>
            <div class="value" style="color:{'#00ff00' if ret_5d > 0 else '#ff4444'}">{ret_5d:+.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">0050 近 20 日</div>
            <div class="value" style="color:{'#00ff00' if ret_20d > 0 else '#ff4444'}">{ret_20d:+.1f}%</div>
        </div>
        <div class="stat-card benchmark">
            <div class="label">0050 近 60 日</div>
            <div class="value" style="color:{'#00ff00' if ret_60d > 0 else '#ff4444'}">{ret_60d:+.1f}%</div>
        </div>
        <div class="stat-card risk">
            <div class="label">市場波動率 (20D)</div>
            <div class="value">{mkt_vol:.1f}%</div>
        </div>
    </div>"""
    except Exception:
        pass

    # === 回撤分析 ===
    drawdown_analysis_html = ""
    try:
        eq = equity_df['Equity']
        cummax_eq = eq.cummax()
        dd_series = (eq / cummax_eq - 1) * 100
        current_dd = dd_series.iloc[-1]
        peak_date = cummax_eq.idxmax() if hasattr(cummax_eq, 'idxmax') else None
        peak_val = cummax_eq.max()

        # 回撤時間
        in_dd = dd_series < -1  # 超過 1% 才算回撤
        dd_periods = []
        start = None
        for dt, val in dd_series.items():
            if val < -1 and start is None:
                start = dt
            elif val >= -0.5 and start is not None:
                dd_periods.append((start, dt, dd_series.loc[start:dt].min()))
                start = None
        if start is not None:
            dd_periods.append((start, dd_series.index[-1], dd_series.loc[start:].min()))

        # Top 5 回撤
        dd_periods.sort(key=lambda x: x[2])
        top_dd_rows = ""
        for i, (s, e, depth) in enumerate(dd_periods[:5]):
            duration = (e - s).days
            top_dd_rows += (
                f'<tr><td>#{i+1}</td>'
                f'<td>{s.strftime("%Y-%m-%d")}</td>'
                f'<td>{e.strftime("%Y-%m-%d")}</td>'
                f'<td style="color:#ff4444; font-weight:bold">{depth:.1f}%</td>'
                f'<td>{duration} 天</td></tr>\n'
            )

        dd_color2 = "#00ff00" if current_dd > -5 else ("#ffab00" if current_dd > -10 else "#ff4444")
        drawdown_analysis_html = f"""
    <div class="stats">
        <div class="stat-card risk">
            <div class="label">目前回撤</div>
            <div class="value" style="color:{dd_color2}">{current_dd:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="label">權益最高點</div>
            <div class="value">{peak_val:,.0f}</div>
        </div>
        <div class="stat-card">
            <div class="label">歷史回撤次數</div>
            <div class="value">{len(dd_periods)}</div>
        </div>
    </div>
    <table>
        <thead><tr><th>#</th><th>開始日</th><th>恢復日</th><th>最大深度</th><th>持續天數</th></tr></thead>
        <tbody>
{top_dd_rows}
        </tbody>
    </table>"""
    except Exception:
        pass

    # === 滾動 Sharpe (60 日) ===
    rolling_perf_html = ""
    try:
        eq = equity_df['Equity']
        daily_ret = eq.pct_change().dropna()
        if len(daily_ret) > 60:
            roll_sharpe = daily_ret.rolling(60).mean() / daily_ret.rolling(60).std() * (252**0.5)
            roll_sharpe = roll_sharpe.dropna()
            recent_sharpe = roll_sharpe.iloc[-1]
            avg_sharpe = roll_sharpe.mean()
            min_sharpe = roll_sharpe.min()
            max_sharpe = roll_sharpe.max()

            rs_color = "#00ff00" if recent_sharpe > 2 else ("#ffab00" if recent_sharpe > 1 else "#ff4444")
            rolling_perf_html = f"""
    <div class="stats">
        <div class="stat-card" style="border-left-color:{rs_color}">
            <div class="label">近 60 日 Sharpe</div>
            <div class="value" style="color:{rs_color}">{recent_sharpe:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">歷史平均 Sharpe</div>
            <div class="value">{avg_sharpe:.2f}</div>
        </div>
        <div class="stat-card risk">
            <div class="label">歷史最低 Sharpe</div>
            <div class="value">{min_sharpe:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">歷史最高 Sharpe</div>
            <div class="value">{max_sharpe:.2f}</div>
        </div>
    </div>"""
    except Exception:
        pass

    # === 連續盈虧分析 ===
    streak_html = ""
    try:
        if not trades_df.empty:
            returns = trades_df['Return_Pct'].values
            max_win_streak = max_loss_streak = 0
            cur_win = cur_loss = 0
            for r in returns:
                if r > 0:
                    cur_win += 1
                    cur_loss = 0
                    max_win_streak = max(max_win_streak, cur_win)
                else:
                    cur_loss += 1
                    cur_win = 0
                    max_loss_streak = max(max_loss_streak, cur_loss)

            # 最近 5 筆
            last5 = trades_df.tail(5)
            recent_streak = ""
            for _, r in last5.iterrows():
                c = "🟢" if r['Return_Pct'] > 0 else "🔴"
                recent_streak += f"{c} "

            streak_html = f"""
    <div class="stats">
        <div class="stat-card">
            <div class="label">最長連勝</div>
            <div class="value" style="color:#00ff00">{max_win_streak} 筆</div>
        </div>
        <div class="stat-card risk">
            <div class="label">最長連敗</div>
            <div class="value" style="color:#ff4444">{max_loss_streak} 筆</div>
        </div>
        <div class="stat-card">
            <div class="label">近 5 筆走勢</div>
            <div class="value" style="font-size:1.2rem">{recent_streak}</div>
        </div>
    </div>"""
    except Exception:
        pass

    # === Per-stock 排行榜 ===
    stock_leaderboard_html = ""
    try:
        if stock_stats:
            sorted_stocks = sorted(stock_stats.items(), key=lambda x: x[1]['total_return'], reverse=True)
            # Top 10 + Bottom 5
            top_rows = ""
            for ticker, ss in sorted_stocks[:10]:
                ret_color = "#00ff00" if ss['total_return'] > 0 else "#ff4444"
                wr_color = "#00ff00" if ss['win_rate'] >= 50 else "#ff4444"
                top_rows += (
                    f'<tr>'
                    f'<td><b>{ticker}</b></td>'
                    f'<td>{ss["trades"]}</td>'
                    f'<td style="color:{wr_color}">{ss["win_rate"]:.0f}%</td>'
                    f'<td style="color:{ret_color}">{ss["avg_return"]:+.1f}%</td>'
                    f'<td style="color:{ret_color}; font-weight:bold">{ss["total_return"]:+.1f}%</td>'
                    f'</tr>\n'
                )
            bottom_rows = ""
            for ticker, ss in sorted_stocks[-5:]:
                ret_color = "#00ff00" if ss['total_return'] > 0 else "#ff4444"
                wr_color = "#00ff00" if ss['win_rate'] >= 50 else "#ff4444"
                bottom_rows += (
                    f'<tr style="opacity:0.7">'
                    f'<td><b>{ticker}</b></td>'
                    f'<td>{ss["trades"]}</td>'
                    f'<td style="color:{wr_color}">{ss["win_rate"]:.0f}%</td>'
                    f'<td style="color:{ret_color}">{ss["avg_return"]:+.1f}%</td>'
                    f'<td style="color:{ret_color}; font-weight:bold">{ss["total_return"]:+.1f}%</td>'
                    f'</tr>\n'
                )

            stock_leaderboard_html = f"""
    <h3 style="color:#00e5ff; margin-top:16px">🏆 Top 10 貢獻股</h3>
    <table>
        <thead><tr><th>股票</th><th>交易數</th><th>勝率</th><th>平均報酬</th><th>總貢獻</th></tr></thead>
        <tbody>{top_rows}</tbody>
    </table>
    <h3 style="color:#ff4444; margin-top:16px">📉 Bottom 5 虧損股</h3>
    <table>
        <thead><tr><th>股票</th><th>交易數</th><th>勝率</th><th>平均報酬</th><th>總貢獻</th></tr></thead>
        <tbody>{bottom_rows}</tbody>
    </table>"""
    except Exception:
        pass

    # === 報酬分布 ===
    distribution_html = ""
    try:
        if not trades_df.empty:
            rets = trades_df['Return_Pct'] * 100
            buckets = [
                ('< -15%', -999, -15), ('-15~-10%', -15, -10), ('-10~-5%', -10, -5),
                ('-5~0%', -5, 0), ('0~5%', 0, 5), ('5~10%', 5, 10),
                ('10~20%', 10, 20), ('> 20%', 20, 999),
            ]
            dist_rows = ""
            for label, lo, hi in buckets:
                cnt = len(rets[(rets >= lo) & (rets < hi)])
                pct = cnt / len(rets) * 100
                bar_width = min(pct * 3, 100)
                color = "#ff4444" if lo < 0 else "#00ff00"
                dist_rows += (
                    f'<tr>'
                    f'<td>{label}</td>'
                    f'<td>{cnt}</td>'
                    f'<td>{pct:.1f}%</td>'
                    f'<td><div style="background:{color}; width:{bar_width}%; height:14px; '
                    f'border-radius:3px; opacity:0.6"></div></td>'
                    f'</tr>\n'
                )

            distribution_html = f"""
    <table>
        <thead><tr><th>報酬區間</th><th>筆數</th><th>佔比</th><th>分布</th></tr></thead>
        <tbody>{dist_rows}</tbody>
    </table>"""
    except Exception:
        pass

    # === 產出 HTML ===
    report_date = latest_date.strftime('%Y-%m-%d')
    cost_desc = f"買 {config.get('buy_cost', 0.001425)*100:.3f}% + 賣 {config.get('sell_cost', 0.004425)*100:.3f}%"
    mode_html = f"ATR×{config.get('tp_atr_mult', 3)}/{config.get('sl_atr_mult', 1.5)}" \
        if tp_sl_mode == 'atr' else f"停利 +{tp_pct*100:.0f}% 停損 -{sl_pct*100:.0f}%"
    if config.get('trailing_stop', False):
        mode_html += f" +Trailing({config.get('trailing_atr_mult', 2.0)}×ATR)"

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI 台股量化交易 v8.5 — {report_date}</title>
    <meta name="description" content="AI 驅動的台股量化交易系統 v8.5，完整風險報告、Benchmark 對比、OCO 智慧掛單建議">
    <script>
        (function() {{
            let savedTheme = null;
            try {{ savedTheme = localStorage.getItem('tw-stocker-theme'); }} catch (error) {{}}
            document.documentElement.dataset.theme = savedTheme === 'light' ? 'light' : 'dark';
        }})();
    </script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        html {{
            width: 100%;
            overflow-x: hidden;
            color-scheme: dark;
        }}
        :root {{
            --bg: #0b0f14;
            --surface: #111820;
            --surface-soft: #17212b;
            --text: #eef4fb;
            --muted: #94a3b8;
            --border: #283545;
            --line: #1f2a37;
            --blue: #38bdf8;
            --green: #34d399;
            --red: #fb7185;
            --orange: #fbbf24;
            --purple: #c084fc;
            --shadow: 0 12px 32px rgba(0, 0, 0, 0.28);
            --card-border: rgba(148, 163, 184, 0.16);
            --hover: rgba(148, 163, 184, 0.08);
            --badge-bg: #1f2937;
            --toggle-bg: rgba(17, 24, 39, 0.86);
            --toggle-border: rgba(148, 163, 184, 0.26);
        }}
        html[data-theme="light"] {{
            color-scheme: light;
            --bg: #f5f5f7;
            --surface: #ffffff;
            --surface-soft: #fbfbfd;
            --text: #1d1d1f;
            --muted: #686872;
            --border: #d2d2d7;
            --line: #f2f2f7;
            --blue: #007aff;
            --green: #248a3d;
            --red: #d70015;
            --orange: #bf6a02;
            --purple: #af52de;
            --shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
            --card-border: rgba(0, 0, 0, 0.04);
            --hover: #f7f7fa;
            --badge-bg: #e8e8ed;
            --toggle-bg: rgba(255, 255, 255, 0.88);
            --toggle-border: rgba(0, 0, 0, 0.10);
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            padding: 40px clamp(12px, 4vw, 24px);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            min-width: 0;
            overflow-x: hidden;
        }}
        .container {{
            width: 100%;
            max-width: 1120px;
            min-width: 0;
            margin: 0 auto;
        }}
        .theme-toggle {{
            position: fixed;
            top: 16px;
            right: 16px;
            z-index: 20;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            min-height: 38px;
            padding: 8px 12px;
            border: 1px solid var(--toggle-border);
            border-radius: 999px;
            background: var(--toggle-bg);
            color: var(--text);
            box-shadow: var(--shadow);
            font: inherit;
            font-weight: 650;
            cursor: pointer;
            backdrop-filter: blur(12px);
        }}
        .theme-toggle:hover {{ transform: translateY(-1px); }}
        .theme-icon {{
            display: inline-grid;
            width: 18px;
            height: 18px;
            place-items: center;
            font-size: 0.98rem;
            line-height: 1;
        }}
        .theme-label {{ font-size: 0.82rem; }}
        h1 {{
            font-size: 2.2rem;
            font-weight: 650;
            letter-spacing: 0;
            text-align: center;
            margin-bottom: 8px;
        }}
        h2 {{
            font-size: 1.35rem;
            font-weight: 650;
            letter-spacing: 0;
            margin-top: 40px;
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}
        .subtitle {{
            color: var(--muted);
            font-size: 0.95rem;
            text-align: center;
            margin-bottom: 32px;
            font-weight: 500;
            overflow-wrap: anywhere;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(min(160px, 100%), 1fr));
            gap: 14px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: var(--surface);
            padding: 18px 16px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
            min-width: 0;
        }}
        .stat-card.risk {{ border-top: 3px solid var(--purple); }}
        .stat-card.benchmark {{ border-top: 3px solid var(--orange); }}
        .stat-card .value {{
            font-size: 1.55rem;
            font-weight: 650;
            margin: 6px 0;
            letter-spacing: 0;
        }}
        .stat-card .label {{
            font-size: 0.75rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-weight: 600;
        }}
        table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin-top: 10px;
            background: var(--surface);
            border-radius: 8px;
            box-shadow: var(--shadow);
            overflow: hidden;
            max-width: 100%;
        }}
        th, td {{
            padding: 13px 14px;
            text-align: left;
            border-bottom: 1px solid var(--line);
            font-size: 0.88rem;
            vertical-align: top;
            overflow-wrap: anywhere;
        }}
        th {{
            background: var(--surface-soft);
            color: var(--muted);
            font-weight: 600;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: var(--hover); }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            margin-top: 14px;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
        }}
        .disclaimer {{
            margin-top: 52px;
            padding: 22px;
            background: var(--surface);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            font-size: 0.85rem;
            color: var(--muted);
            line-height: 1.6;
            text-align: center;
            box-shadow: var(--shadow);
        }}
        .config-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: #e8e8ed;
            background: var(--badge-bg);
            color: var(--text);
            margin: 2px;
        }}
        .section-note {{
            font-size: 0.85rem;
            color: var(--muted);
            margin-bottom: 10px;
        }}
        a {{ color: var(--blue); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        span[style*="color:#00ff00"], b[style*="color:#00ff00"], div[style*="color:#00ff00"], td[style*="color:#00ff00"] {{ color: var(--green) !important; }}
        span[style*="color:#ff4444"], b[style*="color:#ff4444"], div[style*="color:#ff4444"], td[style*="color:#ff4444"] {{ color: var(--red) !important; }}
        span[style*="color:#ffab00"], div[style*="color:#ffab00"], td[style*="color:#ffab00"] {{ color: var(--orange) !important; }}
        span[style*="color:#00e5ff"], h3[style*="color:#00e5ff"] {{ color: var(--blue) !important; }}
        span[style*="color:#aaaaaa"], span[style*="color:#888"], td[style*="color:#888"] {{ color: var(--muted) !important; }}
        div[style*="background:#00ff00"] {{ background: var(--green) !important; }}
        div[style*="background:#ff4444"] {{ background: var(--red) !important; }}
        div[style*="background:linear-gradient"] {{ filter: brightness(0.9) saturate(1.2); }}
        .split-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 20px;
        }}
        .split-grid > div {{ min-width: 0; }}
        @media (max-width: 720px) {{
            body {{ padding: 56px 10px 24px; }}
            .theme-toggle {{
                top: 10px;
                right: 10px;
                min-height: 34px;
                padding: 7px 10px;
            }}
            .theme-label {{ font-size: 0.76rem; }}
            h1 {{ font-size: 1.7rem; }}
            h2 {{ font-size: 1.15rem; margin-top: 30px; }}
            .subtitle {{
                display: flex;
                flex-wrap: wrap;
                justify-content: center;
                gap: 6px;
            }}
            .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
            .stat-card {{ padding: 14px 12px; }}
            .stat-card .value {{ font-size: 1.25rem; }}
            .split-grid {{ grid-template-columns: minmax(0, 1fr); gap: 16px; }}
            table {{
                display: block;
                width: 100%;
                max-width: 100%;
                overflow-x: auto;
                white-space: nowrap;
                -webkit-overflow-scrolling: touch;
            }}
            th, td {{ white-space: nowrap; }}
            td:nth-child(5), td:nth-child(6), .disclaimer {{
                white-space: normal;
                min-width: 220px;
            }}
            th, td {{ padding: 11px 12px; font-size: 0.82rem; }}
        }}
        @media (max-width: 420px) {{
            .stats {{ grid-template-columns: minmax(0, 1fr); }}
            .config-badge {{ max-width: 100%; white-space: normal; }}
        }}
    </style>
</head>
<body>
<button class="theme-toggle" id="themeToggle" type="button" aria-label="切換深色與淺色主題" aria-pressed="true">
    <span class="theme-icon" aria-hidden="true">☾</span>
    <span class="theme-label">深色</span>
</button>
<div class="container">

    <h1>🎯 AI 台股量化交易 v8.5</h1>
    <p class="subtitle">
        Event-Driven System &nbsp;|&nbsp; 報表日期: {report_date} &nbsp;|&nbsp;
        <span class="config-badge">🛡️ {mode_html}</span>
        <span class="config-badge">🎯 Top-{top_k}</span>
        <span class="config-badge">⏱️ 最長 {max_hold_days} 天</span>
        <span class="config-badge">💰 成本: {cost_desc}</span>
    </p>

    <h2>📊 績效總覽</h2>
    <div class="stats">
        <div class="stat-card">
            <div class="label">策略總報酬率</div>
            <div class="value" style="color:{total_ret_color};">{total_ret:+.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="label">年化報酬率</div>
            <div class="value" style="color:{total_ret_color};">{m['ann_return']*100:+.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="label">年化波動率</div>
            <div class="value">{m['ann_volatility']*100:.1f}%</div>
        </div>
        <div class="stat-card risk">
            <div class="label">Sharpe Ratio</div>
            <div class="value" style="color:{sharpe_color};">{m['sharpe']:.2f}</div>
        </div>
        <div class="stat-card risk">
            <div class="label">Sortino Ratio</div>
            <div class="value">{m['sortino']:.2f}</div>
        </div>
        <div class="stat-card risk">
            <div class="label">最大回撤</div>
            <div class="value" style="color:{dd_color};">{m['max_drawdown_pct']*100:.1f}%</div>
        </div>
        <div class="stat-card risk">
            <div class="label">Calmar Ratio</div>
            <div class="value">{m['calmar']:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">完成交易數</div>
            <div class="value">{m['total_trades']}</div>
        </div>
        <div class="stat-card">
            <div class="label">勝率</div>
            <div class="value" style="color:#00ff00;">{m['win_rate']*100:.1f}%</div>
        </div>
        <div class="stat-card">
            <div class="label">Profit Factor</div>
            <div class="value">{m['profit_factor']:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="label">平均報酬/筆</div>
            <div class="value">{m['avg_return']*100:+.2f}%</div>
        </div>
        <div class="stat-card">
            <div class="label">平均持有天數</div>
            <div class="value">{m['avg_days_held']:.0f}</div>
        </div>
    </div>

    <h2>🚀 今日 AI 交易執行單</h2>
    <p class="section-note">
        信號基於昨日收盤產生，建議於明日開盤價附近掛單進場。
        {f'最晚出場日使用 XTAI 交易日曆計算' if HAS_EXCHANGE_CAL else '最晚出場日為近似值（未安裝 exchange_calendars）'}
    </p>
    <table>
        <thead>
            <tr>
                <th>股票代號</th>
                <th>AI 評分</th>
                <th>今日收盤</th>
                <th>操作狀態</th>
                <th>🎯 區間執行計畫</th>
                <th>📊 歷史績效</th>
                <th>🏛️ 籌碼</th>
                <th>📰 新聞</th>
            </tr>
        </thead>
        <tbody>
{trading_plan_rows}
        </tbody>
    </table>

{inst_section_html}

    <h2>📈 資金曲線 vs Benchmark</h2>
{benchmark_stats_html}
{benchmark2_stats_html}
    <img src="backtest_chart.png" alt="AI Quantitative Backtest Equity Curve with Benchmark">

    <h2>🌐 大盤環境診斷</h2>
    <p class="section-note">判斷目前市場 regime 與近期趨勢，幫助你決定是否積極或保守操作。</p>
{market_context_html}

    <h2>📉 回撤深度分析</h2>
    <p class="section-note">歷史回撤事件的深度與持續時間，識別策略最脆弱的時期。</p>
{drawdown_analysis_html}

    <h2>📊 滾動績效指標 (60 日)</h2>
    <p class="section-note">近期表現是否仍在健康範圍內，偏離均值時需警惕。</p>
{rolling_perf_html}

    <h2>🔥 連續盈虧分析</h2>
    <p class="section-note">連勝/連敗紀錄與近期走勢，評估策略是否處於熱手或冷卻期。</p>
{streak_html}

    <h2>📋 出場原因分布統計</h2>
    <table>
        <thead><tr><th>出場原因</th><th>次數</th><th>平均報酬</th></tr></thead>
        <tbody>
{reason_stats_rows}
        </tbody>
    </table>

    <h2>📊 報酬分布直方圖</h2>
    <p class="section-note">每筆交易報酬的分布形狀，正偏態代表策略有良好的尾部收益。</p>
{distribution_html}

    <h2>🏅 個股績效排行榜</h2>
    <p class="section-note">哪些股票貢獻最多利潤、哪些持續虧損，幫助判讀信號品質。</p>
{stock_leaderboard_html}

    <h2>📜 最近交易紀錄 (最新 20 筆)</h2>
    <table>
        <thead>
            <tr>
                <th>股票</th><th>進場日</th><th>出場日</th>
                <th>進場價</th><th>出場價</th>
                <th>觸發原因</th><th>持有天數</th><th>報酬率</th>
            </tr>
        </thead>
        <tbody>
{trade_history_rows}
        </tbody>
    </table>

    <h2>📅 月度報酬熱圖</h2>
{monthly_heatmap_html}

    <div class="disclaimer">
        ⚠️ <b>免責聲明：</b>本報表由 AI 量化模型自動產出，僅供學術研究與技術交流之用，
        不構成任何投資建議。歷史回測績效不代表未來實際報酬，投資有風險，決策請自行負責。
        <br><br>
        <b>v8.5 方法論：</b>Entry = t+1 open | TP/SL = {mode_html} | 選股 = Top-{top_k} cross-sectional rank |
        成本 = {cost_desc} | 回測期 = {m['years']:.1f} 年 | 因子 = Mom(20d)×3 + Trend(60MA)×1
    </div>

</div>
<script>
    (function() {{
        const button = document.getElementById('themeToggle');
        if (!button) return;
        const icon = button.querySelector('.theme-icon');
        const label = button.querySelector('.theme-label');

        function setTheme(theme) {{
            const nextTheme = theme === 'light' ? 'light' : 'dark';
            document.documentElement.dataset.theme = nextTheme;
            try {{ localStorage.setItem('tw-stocker-theme', nextTheme); }} catch (error) {{}}
            button.setAttribute('aria-pressed', String(nextTheme === 'dark'));
            if (icon) icon.textContent = nextTheme === 'dark' ? '☾' : '☀';
            if (label) label.textContent = nextTheme === 'dark' ? '深色' : '淺色';
        }}

        setTheme(document.documentElement.dataset.theme || 'dark');
        button.addEventListener('click', function() {{
            setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
        }});
    }})();
</script>
</body>
</html>"""

    with open('stock_report.html', 'w', encoding='utf-8') as f:
        f.write(html)

    # === 輸出 artifacts ===
    os.makedirs('artifacts', exist_ok=True)
    date_str = latest_date.strftime('%Y%m%d')

    if not trades_df.empty:
        trades_df.to_csv(f'artifacts/trades_{date_str}.csv', index=False)

    equity_df.to_csv(f'artifacts/equity_{date_str}.csv')

    # 信號快照
    today_signals = total_score.loc[[latest_date]].T
    today_signals.columns = ['Score']
    today_signals = today_signals.dropna().sort_values('Score', ascending=False)
    today_signals.to_csv(f'artifacts/signals_{date_str}.csv')

    with open(f'artifacts/orders_{date_str}.json', 'w', encoding='utf-8') as f:
        json.dump({'orders': orders}, f, indent=2, ensure_ascii=False)

    try:
        git_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except Exception:
        git_sha = None
    metadata = {
        'created_at': datetime.now().isoformat(),
        'strategy_version': 'v8.5',
        'git_sha': git_sha,
        'report_date': latest_date.strftime('%Y-%m-%d'),
        'config': config,
        'metrics': {
            'total_return': metrics.get('total_return'),
            'ann_return': metrics.get('ann_return'),
            'sharpe': metrics.get('sharpe'),
            'geometric_sharpe': metrics.get('geometric_sharpe'),
            'sortino': metrics.get('sortino'),
            'calmar': metrics.get('calmar'),
            'max_drawdown_pct': metrics.get('max_drawdown_pct'),
            'total_trades': metrics.get('total_trades'),
        },
        'artifacts': {
            'equity': f'artifacts/equity_{date_str}.csv',
            'trades': f'artifacts/trades_{date_str}.csv' if not trades_df.empty else None,
            'signals': f'artifacts/signals_{date_str}.csv',
            'orders': f'artifacts/orders_{date_str}.json',
        },
    }
    with open(f'artifacts/metadata_{date_str}.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    print(f"   ✅ 報表已生成：stock_report.html")
    print(f"   📁 Artifacts 已存入 artifacts/ 目錄")


def parse_args():
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description='AI 台股量化交易系統 v8.5 — 事件驅動回測與交易計畫產生器'
    )
    # 股池
    parser.add_argument(
        '--tickers', nargs='+', default=None,
        help='手動指定股池（靜態池模式）'
    )
    parser.add_argument(
        '--static-pool', action='store_true',
        help='使用靜態池模式（14 檔預設股）而非動態 Universe'
    )
    parser.add_argument(
        '--universe-size', type=int, default=60,
        help='動態 Universe 大小 (預設: 60)'
    )

    # TP/SL
    parser.add_argument(
        '--tp-sl-mode', choices=['fixed', 'atr'], default='atr',
        help='TP/SL 模式: fixed=固定百分比, atr=ATR倍數 (預設: atr)'
    )
    parser.add_argument(
        '--tp', type=float, default=0.15,
        help='固定模式停利百分比 (預設: 0.15 = +15%%)'
    )
    parser.add_argument(
        '--sl', type=float, default=0.08,
        help='固定模式停損百分比 (預設: 0.08 = -8%%)'
    )
    parser.add_argument(
        '--tp-atr', type=float, default=4.0,
        help='ATR 模式停利倍數 (預設: 4.0)'
    )
    parser.add_argument(
        '--sl-atr', type=float, default=3.0,
        help='ATR 模式停損倍數 (預設: 3.0)'
    )

    # Trailing Stop
    parser.add_argument(
        '--trailing', action='store_true',
        help='啟用移動停利 (Trailing Stop)，停用固定 TP，讓強趨勢延伸'
    )
    parser.add_argument(
        '--trailing-atr', type=float, default=2.0,
        help='移動停利 ATR 倍數 (預設: 2.0, 從最高點回落此倍數 ATR 觸發)'
    )

    # 選股
    parser.add_argument(
        '--top-k', type=int, default=7,
        help='每日最多進場股票數 (預設: 7)'
    )
    parser.add_argument(
        '--threshold', type=float, default=2.0,
        help='AI 評分安全下限 (預設: 2.0，低於此分數不進場)'
    )

    # 持倉
    parser.add_argument(
        '--hold-days', type=int, default=20,
        help='最大持倉交易日數 (預設: 20)'
    )

    # 進場過濾器
    parser.add_argument(
        '--regime-filter', action='store_true', default=True,
        help='大盤過濾 (0050 > 60MA 才允許進場, 預設: ON)'
    )
    parser.add_argument(
        '--no-regime-filter', action='store_false', dest='regime_filter',
        help='停用大盤過濾'
    )
    parser.add_argument(
        '--gap-filter', type=float, default=1.5,
        help='跳空過濾 ATR 倍數 (預設: 1.5, 0=停用)'
    )
    parser.add_argument(
        '--volume-confirm', action='store_true',
        help='啟用成交量確認 (進場日成交量 > 20日均量)'
    )
    parser.add_argument(
        '--blacklist', type=int, default=0,
        help='動態黑名單回顧筆數 (預設: 0=停用, 10=最近10筆勝率<25%%則排除)'
    )
    parser.add_argument(
        '--breakeven', type=float, default=0,
        help='獲利保護觸發門檻 (預設: 0=停用, 0.03=+3%%後 SL 移至成本價)'
    )
    parser.add_argument(
        '--slippage', type=float, default=0.001,
        help='滑價模型 (預設: 0.001=10bps; 0=停用)'
    )
    parser.add_argument(
        '--vol-parity', action='store_true',
        help='啟用波動率平價 (Volatility Parity) 部位調整'
    )
    parser.add_argument(
        '--multi-ma', action='store_true',
        help='啟用多均線確認 (20MA > 60MA 才允許進場)'
    )
    parser.add_argument(
        '--ma-period', type=int, default=60,
        help='主趨勢均線天數 (預設: 60)'
    )
    parser.add_argument(
        '--ml-weights', action='store_true',
        help='啟用 LightGBM 因子加權 (取代等權加總)'
    )

    # 資金
    parser.add_argument(
        '--capital', type=float, default=200_000,
        help='初始模擬資金 (預設: 200000)'
    )
    parser.add_argument(
        '--position-size', type=float, default=0.10,
        help='每筆倉位佔當前權益比例 (預設: 0.10 = 10%%)'
    )

    # 成本
    parser.add_argument(
        '--buy-cost', type=float, default=0.001425,
        help='買入手續費率 (預設: 0.001425 = 0.1425%%)'
    )
    parser.add_argument(
        '--sell-cost', type=float, default=0.004425,
        help='賣出成本率 (手續費+證交稅, 預設: 0.004425 = 0.1425%%+0.3%%)'
    )

    # 回測
    parser.add_argument(
        '--days', type=int, default=1200,
        help='歷史回測天數 (預設: 1200)'
    )
    parser.add_argument(
        '--start-date', type=str, default=None,
        help='明確指定回測起始日期 (YYYY-MM-DD，優先於 --days)'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='明確指定回測結束日期 (YYYY-MM-DD，預設為今天)'
    )
    parser.add_argument(
        '--eval-start', type=str, default=None,
        help='績效評估起始日期 (YYYY-MM-DD)。資料仍可從 --start-date 提早抓取作暖機'
    )

    # 結構性功能
    parser.add_argument(
        '--mean-reversion', action='store_true',
        help='啟用均值回歸子策略（熊市時買入超跌反彈股）'
    )
    parser.add_argument(
        '--dynamic-risk', action='store_true',
        help='啟用動態風險預算（根據市場波動調整部位）'
    )
    parser.add_argument(
        '--futures-hedge', action='store_true',
        help='啟用台指期空單對沖（熊市時模擬做空大盤）'
    )

    # 風控竟日卡（預設停用，非動量策略可開啟）
    parser.add_argument(
        '--dd-pause-pct', type=float, default=1.0,
        help='權益回撤暫停門檻 (預設 1.0 = 停用; 建議實盤設 0.15)'
    )
    parser.add_argument(
        '--dd-pause-days', type=int, default=5,
        help='回撤觸發後暫停新倉天數 (預設 5)'
    )
    parser.add_argument(
        '--consec-loss-limit', type=int, default=99,
        help='連續停損次數上限 (預設 99 = 停用; 建議實盤設 3)'
    )
    parser.add_argument(
        '--consec-loss-pause', type=int, default=5,
        help='連續停損後暫停天數 (預設 5)'
    )
    parser.add_argument(
        '--sector-max-pct', type=float, default=0.75,
        help='單一板塊最大持倉比例 (預設 0.75 = 75%%; 1.0 = 停用)'
    )
    parser.add_argument(
        '--corr-filter', type=float, default=0.8,
        help='相關性過濾門檻 (預設 0.8; 0 = 停用; 去除近 20 日相關>門檻的重複持倉)'
    )
    parser.add_argument(
        '--max-heat', type=float, default=1.0,
        help='組合熱度上限 (預設 1.0 = 停用; 實測顯示 2%% 過緊導致交易數量驟減)'
    )
    parser.add_argument(
        '--rank-weight', action='store_true',
        help='啟用排名加權 sizing (預設停用; 實測顯示對動量策略有害)'
    )
    parser.add_argument(
        '--regime-delev', action='store_true',
        help='啟用 Regime 降曝險 (預設停用; 實測顯示會錯過反彈)'
    )
    parser.add_argument(
        '--regime-graduated', action='store_true', default=True,
        help='啟用四段式曝險縮放 (100%%/70%%/40%%/0%%), 取代 binary regime filter (預設: ON)'
    )
    parser.add_argument(
        '--no-regime-graduated', action='store_false', dest='regime_graduated',
        help='停用四段式曝險，改用 binary regime filter'
    )
    parser.add_argument(
        '--regime-floor', type=float, default=0.10,
        help='空頭 regime 最低曝險下限 (0.0=完全停止, 0.2=維持20%%進場能力)'
    )
    parser.add_argument(
        '--inst-flow', type=float, default=0.0,
        help='籌碼因子權重 (預設 0 = 停用; 建議先用 0 觀察，累積數據後再加權)'
    )
    parser.add_argument(
        '--confidence-k', action='store_true',
        help='啟用 Confidence-K：根據分數品質動態調整 Top-K（分數差太大時少買）'
    )
    parser.add_argument(
        '--mid-hold-review', action='store_true',
        help='啟用中期汰弱：持有 10-14 天仍虧損且動量衰退→提早出場'
    )
    # === Phase 2 features ===
    parser.add_argument(
        '--breadth-regime', action='store_true', default=True,
        help='v8.5: Breadth-aware regime：用 universe 內部寬度修正 regime 判斷 (預設開啟)'
    )
    parser.add_argument(
        '--candidate-breadth', action='store_true',
        help='啟用候選寬度：前 15 名候選股 20MA 支撐品質檢查'
    )
    parser.add_argument(
        '--theme-breadth', action='store_true',
        help='啟用主題寬度：前 15 名候選股板塊集中度檢查'
    )
    parser.add_argument(
        '--residual-momentum', action='store_true',
        help='啟用殘差動量：扣除市場 beta 後的個股動量'
    )
    parser.add_argument(
        '--trend-quality', action='store_true',
        help='啟用趨勢品質：slope + 均線排列 + 過熱抑制'
    )
    parser.add_argument(
        '--gap-aware-sizing', action='store_true', default=True,
        help='啟用 Gap-aware sizing：跳空越大，進場倉位越小'
    )
    parser.add_argument(
        '--dynamic-sector-cap', action='store_true',
        help='啟用動態板塊上限：regime 越弱，sector cap 越緊'
    )
    parser.add_argument(
        '--liq-stability', action='store_true',
        help='啟用流動性穩定度：排除突然爆量但平常不穩的標的'
    )
    parser.add_argument(
        '--liq-mode', type=str, default='raw', choices=['raw', 'sector', 'demeaned'],
        help='流動性穩定度模式：raw=全局, sector=行業中性, demeaned=殘差 (預設 raw)'
    )
    parser.add_argument(
        '--cluster-penalty', action='store_true',
        help='啟用 Cluster Penalty：根據候選與持倉的相關性 soft-penalize 分數'
    )
    parser.add_argument(
        '--show-inst', action='store_true', default=True,
        help='在報表信號中顯示三大法人籌碼與新聞情緒標注 (預設開啟)'
    )
    parser.add_argument(
        '--no-show-inst', dest='show_inst', action='store_false',
        help='關閉籌碼與新聞標注'
    )
    parser.add_argument(
        '--macro-regime', action='store_true',
        help='啟用宏觀 Regime 疊加：VIX > 22/25/30 時降低曝險'
    )
    parser.add_argument(
        '--batch-entry', type=int, default=1, choices=[1],
        help='分批建倉天數（目前僅支援 1=單筆；多批需 pending order state machine）'
    )
    parser.add_argument(
        '--dynamic-topk', action='store_true',
        help='動態 Top-K：弱勢 Regime 自動降低持股數'
    )
    parser.add_argument(
        '--dynamic-gap-filter', action='store_true',
        help='動態 Gap Filter：強勢 Regime 放寬跳空限制至 2.0 ATR'
    )
    parser.add_argument(
        '--dynamic-corr-filter', action='store_true',
        help='動態相關性過濾：強勢 Regime 放寬至 0.85'
    )
    parser.add_argument(
        '--sector-flow-tilt', action='store_true',
        help='啟用板塊資金流傾斜：用 10/15/20 天動量追蹤板塊資金流向，動態分配 Top-K 配額'
    )
    parser.add_argument(
        '--tilt-strength', type=float, default=1.0,
        help='板塊傾斜力度 (0.0=均分, 1.0=全力傾斜, 預設 1.0)'
    )
    parser.add_argument(
        '--tilt-windows', type=str, default='10,15,20',
        help='板塊動量計算窗口 (逗號分隔, 預設 10,15,20)'
    )
    # === FinLab 啟發因子（Phase 8: 新因子維度）===
    parser.add_argument(
        '--rsi-weight', type=float, default=0.0,
        help='RSI-20 動量因子權重 (預設 0 = 停用; FinLab 用 RSI 選最強 20 檔)'
    )
    parser.add_argument(
        '--breakout-weight', type=float, default=0.0,
        help='300日創新高突破因子權重 (預設 0 = 停用; FinLab 年化 ~32%%)'
    )
    parser.add_argument(
        '--value-weight', type=float, default=0.0,
        help='PE/PB 價值因子權重 (預設 0 = 停用; FinLab 雙渦輪核心)'
    )
    parser.add_argument(
        '--rev-momentum-weight', type=float, default=0.0,
        help='60日營收動能代理因子權重 (預設 0 = 停用; FinLab MEMORY.md 鐵律)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 決定股池
    if args.static_pool or args.tickers:
        tickers = args.tickers if args.tickers else DEFAULT_TICKERS
        use_dynamic = False
    else:
        tickers = EXTENDED_TICKERS
        use_dynamic = True

    mode_str = f"動態 Universe (Top-{args.universe_size})" if use_dynamic else f"靜態 ({len(tickers)} 檔)"
    tp_sl_str = f"ATR×{args.tp_atr}/{args.sl_atr}" if args.tp_sl_mode == 'atr' \
        else f"+{args.tp*100:.0f}%/-{args.sl*100:.0f}%"
    cost_str = f"買 {args.buy_cost*100:.3f}% 賣 {args.sell_cost*100:.3f}%"

    trailing_str = f" +Trailing({args.trailing_atr}×ATR)" if args.trailing else ""

    print("=" * 60)
    print("🎯 AI 台股量化交易系統 v8.5")
    print("=" * 60)
    print(f"   股池: {mode_str}")
    print(f"   TP/SL: {tp_sl_str}{trailing_str}  Top-K: {args.top_k}  持倉上限: {args.hold_days} 天")
    print(f"   成本: {cost_str}")
    if args.start_date:
        print(f"   回測期間: {args.start_date} → {args.end_date or '今天'}")
    else:
        print(f"   回測天數: {args.days}")
    print("=" * 60)

    # Phase 1: 資料下載
    close_df, open_df, high_df, low_df, vol_df = fetch_panel_data(
        tickers, days=args.days,
        start_date=args.start_date, end_date=args.end_date,
    )

    # Phase 2: 動態 Universe 或靜態池
    if use_dynamic:
        universe_mask = build_liquid_universe(close_df, vol_df, top_n=args.universe_size)
    else:
        universe_mask = None

    # Phase 2.5: 籌碼時序數據（僅用於因子加權；報表顯示使用輕量 API）
    inst_flow_df = None
    if args.inst_flow > 0:
        try:
            inst_flow_df, inst_ratio_df = build_inst_flow_df(
                list(close_df.columns), close_df, verbose=True)
        except Exception as e:
            print(f"   ⚠️ 籌碼數據抓取失敗，跳過: {e}")
            inst_flow_df = None

    # Phase 3.5: 提前下載 0050 用於 regime filter + 殘差動量
    market_close = None
    if args.regime_filter or args.residual_momentum:
        print("\n📊 下載大盤指數 (0050) 用於 regime filter...")
        bench_raw = fetch_benchmark(
            '0050', days=args.days,
            start_date=args.start_date, end_date=args.end_date,
        )
        if len(bench_raw) > 0:
            market_close = bench_raw * bench_raw.iloc[0]

    # Phase 3: 特徵工程
    total_score, ma_60, atr_df, short_ma = engineer_features(
        close_df, vol_df, universe_mask,
        ma_period=args.ma_period,
        multi_ma=args.multi_ma,
        ml_weights=args.ml_weights,
        inst_flow_weight=args.inst_flow,
        inst_flow_df=inst_flow_df,
        residual_momentum=args.residual_momentum,
        trend_quality=args.trend_quality,
        liq_stability=args.liq_stability,
        liq_mode=args.liq_mode,
        market_close=market_close,
        rsi_weight=args.rsi_weight,
        breakout_weight=args.breakout_weight,
        value_weight=args.value_weight,
        rev_momentum_weight=args.rev_momentum_weight,
    )

    # Phase 4: 事件驅動回測
    backtester = EventDrivenBacktester(
        tp_pct=args.tp,
        sl_pct=args.sl,
        max_hold_days=args.hold_days,
        initial_capital=args.capital,
        position_size=args.position_size,
        tp_sl_mode=args.tp_sl_mode,
        tp_atr_mult=args.tp_atr,
        sl_atr_mult=args.sl_atr,
        trailing_stop=args.trailing,
        trailing_atr_mult=args.trailing_atr,
        regime_filter=args.regime_filter,
        regime_graduated=args.regime_graduated,
        regime_floor=args.regime_floor,
        gap_filter_atr=args.gap_filter,
        volume_confirm=args.volume_confirm,
        blacklist_lookback=args.blacklist,
        breakeven_pct=args.breakeven,
        slippage=args.slippage,
        vol_parity=args.vol_parity,
        mean_reversion=args.mean_reversion,
        dynamic_risk=args.dynamic_risk,
        futures_hedge=args.futures_hedge,
        dd_pause_pct=args.dd_pause_pct,
        dd_pause_days=args.dd_pause_days,
        consec_loss_limit=args.consec_loss_limit,
        consec_loss_pause=args.consec_loss_pause,
        sector_max_pct=args.sector_max_pct,
        corr_filter=args.corr_filter,
        max_portfolio_heat=args.max_heat,
        rank_weighted=args.rank_weight,
        regime_deleverage=args.regime_delev,
        confidence_k=args.confidence_k,
        mid_hold_review=args.mid_hold_review,
        breadth_regime=args.breadth_regime,
        candidate_breadth=args.candidate_breadth,
        theme_breadth=args.theme_breadth,
        dynamic_sector_cap=args.dynamic_sector_cap,
        gap_aware_sizing=args.gap_aware_sizing,
        cluster_penalty=args.cluster_penalty,
        macro_regime=args.macro_regime,
        batch_entry=args.batch_entry,
        dynamic_topk=args.dynamic_topk,
        dynamic_gap_filter=args.dynamic_gap_filter,
        dynamic_corr_filter=args.dynamic_corr_filter,
        sector_flow_tilt=args.sector_flow_tilt,
        tilt_strength=args.tilt_strength,
        tilt_windows=[int(w) for w in args.tilt_windows.split(',')],
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
    )
    trades_df, equity_df = backtester.run(
        total_score, close_df, open_df, high_df, low_df, ma_60,
        top_k=args.top_k,
        threshold=args.threshold,
        market_close=market_close,
        vol_df=vol_df,
        universe_mask=universe_mask,
    )

    report_equity_df, report_trades_df = slice_evaluation_window(
        equity_df, trades_df,
        eval_start=args.eval_start,
        initial_capital=args.capital,
    )
    if args.eval_start:
        print(f"\n📏 績效統計區間: {args.eval_start} → {args.end_date or '今天'} "
              f"(資料自 {args.start_date or f'近 {args.days} 天'} 暖機)")

    # Phase 5: 風險指標
    metrics = compute_risk_metrics(report_equity_df, report_trades_df, args.capital)
    print(format_metrics_summary(metrics))

    # Phase 6: Benchmark
    print("\n📊 載入 Benchmark 進行比較...")
    benchmark_start = args.eval_start or args.start_date
    benchmark_equity = fetch_benchmark(
        '0050', days=args.days,
        start_date=benchmark_start, end_date=args.end_date,
    )
    benchmark2_equity = fetch_benchmark(
        '00981A', days=args.days,
        start_date=benchmark_start, end_date=args.end_date,
    )
    ew_equity = equal_weight_benchmark(close_df)
    if benchmark_start:
        ew_equity = ew_equity.loc[pd.to_datetime(ew_equity.index) >= pd.Timestamp(benchmark_start)]
        if not ew_equity.empty:
            ew_equity = ew_equity / ew_equity.iloc[0]

    # Phase 7: 報表產出
    config = {
        'tp_pct': args.tp,
        'sl_pct': args.sl,
        'max_hold_days': args.hold_days,
        'initial_capital': args.capital,
        'threshold': args.threshold,
        'tp_sl_mode': args.tp_sl_mode,
        'tp_atr_mult': args.tp_atr,
        'sl_atr_mult': args.sl_atr,
        'trailing_stop': args.trailing,
        'trailing_atr_mult': args.trailing_atr,
        'top_k': args.top_k,
        'buy_cost': args.buy_cost,
        'sell_cost': args.sell_cost,
        'gap_filter': args.gap_filter,
        'dynamic_gap_filter': args.dynamic_gap_filter,
        # 下一交易日進場實際採用的 gap filter 倍數（含 dynamic regime 放寬），
        # 由回測引擎以 latest_date 大盤資料算出，供 paper trading 精確對齊。
        'gap_filter_effective': backtester.next_session_gap_limit(),
    }
    generate_report(report_trades_df, report_equity_df, total_score, close_df, config,
                    metrics, benchmark_equity, ew_equity,
                    benchmark2_equity=benchmark2_equity,
                    high_df=high_df, low_df=low_df,
                    show_inst=args.show_inst)
    print("\n🚀 全部完成！請打開 stock_report.html 查看結果。")


if __name__ == '__main__':
    main()
