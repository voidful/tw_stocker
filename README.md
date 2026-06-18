# TW Stocker v9.1 — AI 量化交易系統（多策略架構）

中期動量 + 板塊輪動 + 早盤餐費策略的多策略系統。v8.5 個股動量穩健底倉 + Sector Rotation v2 板塊資金流追蹤 + Meal Money v2 早盤小目標策略。
美股前提（SPY/VIX/SOX）→ 板塊資金流選擇 → 板塊內選股。

> 最新重算：2026-05-26。以下數字已套用日期對齊、eval window 裁切、raw OHLCV tradability mask、`.TW/.TWO` fallback、Arithmetic Sharpe 修正。舊版 README 的高 Sharpe / crisis headline 不應再沿用。

📊 **線上報表**：https://voidful.github.io/tw_stocker/stock_report.html
📈 **Paper Trading**：https://voidful.github.io/tw_stocker/paper_trading.html

---

## 多策略架構總覽

| | v8.5 Momentum | Sector Rotation v2 | Meal Money v2 |
|---|:---:|:---:|:---:|
| **邏輯** | 個股 cross-sectional ranking | 先選板塊 → 板塊內排名 | 少交易早盤做多 + 每日雙向量大策略 |
| **Regime** | 台股 0050 vs MA60 | 美股 SPY + VIX + SOX | 台股寬度 + 費半/SOX + 夜盤追蹤；預設不硬 gate |
| **選股因子** | Mom(20d)×3 + Trend(60MA)×1 | 板塊 flow(10/15/20d) + 板塊內動量 | v8.5 score 或 09:00~09:10 高量價壓力；不限制產業別 |
| **角色** | 穩健底倉（低 MDD） | 積極追蹤（高報酬） | 09:40 前結束的小額日內策略 |
| **成本模型** | 買 0.1425% + 賣 0.4425% + 滑價 | 同左 | 一般版賣出稅 0.3%；每日當沖版賣出稅 0.15%；最低手續費 |
| **狀態** | Production baseline | Research sleeve | Experimental / paper first |

---

## Meal Money v2 — 一天賺餐費早盤策略

這是一個獨立於 production 底倉的早盤策略，使用本 repo 的 5 分 K `data/*.csv` 回測。
分類追蹤參考玩股網的上市/上櫃類股結構，repo 內用 `strategy.sector_flow.classify_sector()` 做長期 sector attribution。

核心規則：

```
隔夜篩選:
  1. 沿用 v8.5 分數：rank_momentum(20d) × 3 + rank_trend(60MA) × 1
  2. 前日 close > 60MA
  3. 股價 103 ~ 180
  4. 20 日均量 >= 2,000,000 股；不使用 09:15 bar 成交量作為進場前濾網
  5. 動態流動性 Universe Top-60
  6. 預設不限制產業別；sector 只做長期歸因追蹤
  7. 預設追蹤台股市場寬度、費半/SOX、夜盤；只有明確開 gate 才用來擋單

早盤執行:
  1. 前 15 分鐘不下單；預設 09:15 才允許進場
  2. 09:00~09:10 必須已放量上漲：成交量 >= 2,000,000、漲幅 >= 2.0%、振幅 >= 1.5%
  3. 用 one-tick edge 模型：預設嘗試比參考價低 1 tick 的買價
  4. 開盤 gap 預設需在 +0.5% ~ +4.0%
  5. 此舊版單筆本金預設 100,000，且成交股款不得低於 15,000；用零股 sizing，不用整張
  6. 費用後淨利至少 +500 才算達標；設定目標區間 +500 ~ +800
  7. 目標價用台股 tick size 往上對齊，預設要求最多 3 ticks 達標
  8. 若未達標，09:35 強制出場，確保 09:40 前結束
  9. 同一根 5 分 K 同時碰到停損與停利時，回測採停損優先

長期追蹤:
  1. 產業別績效：`artifacts/meal_money_sector_YYYYMMDD.csv`
  2. 交易時間波動：`artifacts/meal_money_time_focus_YYYYMMDD.csv`
  3. 台股市場寬度 / 5 日市場報酬：`Market_Breadth_20`、`Market_Return_5D`
  4. 費半/SOX：使用 `--track-us-market` 追蹤，`--use-us-market` 才硬 gate
  5. 夜盤：提供 `--night-market-csv` 時寫入 `Night_Return_Pct`，加 `--use-night-filter` 才硬 gate
  6. 預設追蹤時間：09:15、10:00、11:00、11:40、12:00、12:50、13:00、13:20

成本:
  買進 = 買進成交金額 × 0.1425%，每邊最低手續費 20
  賣出 = 賣出成交金額 × 0.1425% + 賣出成交金額 × 0.3%，每邊最低手續費 20
  例如 15,000 成交金額：買手續費 21.375、賣手續費 21.375、證交稅 45，來回成本 87.75
  額外滑價預設 0
```

Historical 100k reference（2024-05-01 → 2026-04-01 extended pool）：

| 指標 | 值 |
|------|:---:|
| Active days | 12 |
| Success days | 8 |
| Daily success rate | 66.7% |
| Total net PnL | +2,249 |
| Avg trade net PnL | +187 |
| Avg trades / success day | 1.00 |
| 09:40 cutoff violations | 0 |

Train/test split：2024-05-01 → 2025-04-30 為 +356；2025-05-01 → 2026-04-01 為 +1,894，測試段 9 筆、6 筆達標、成功率 66.7%。100k focused search 400 組後，此組是在 train/test/full 皆為正的候選中，交易筆數最多的版本；獲利最高版本為 8 筆、全段 +3,640，但交易更少。

前提：此結果使用完整手續費 0.1425% + 證交稅 0.3%，並納入每邊最低手續費 20。100,000 本金要淨賺 500，通常需要約 1% 左右的股價移動；它是舊版大本金參考，不是目前「本金最多 20,000」的主策略。小本金需求以 Daily Meal Money 的 `small-cap-daily` 為準。

Meal Money v2 的目的不是取代 v8.5，而是低相關、短持有時間的 production candidate。它依賴 5 分 K 品質；正式加到每日主報表前，應持續觀察實盤成交率、滑價、產業歸因、夜盤/費半前提與 09:40 前出場紀律。

### Daily Meal Money — 每日雙向量大優先版本

為了符合「本金最多 20,000、盡量每天交易賺午餐錢」的要求，`daily_meal_money_report.py` 現在預設使用 `--preset small-cap-daily`。它不使用前日 v8.5 排名，而是在指定交易時段內從全市場高成交量股票中挑出一檔最強的量價壓力候選；做多與做空皆可，方向由當下量價壓力決定。舊的 100,000 本金版本仍可用 `--preset daily` 明確指定重跑，但不是目前 production candidate。

核心規則：

```
20k 每日候選:
  1. 全市場 data/*.csv，不限制產業別
  2. 股價 103 ~ 180
  3. 20 日均量 >= 1,000,000 股
  4. 允許交易時段：09:00~09:40、10:00~10:20、11:00~11:20、11:40~12:20、12:50~13:30
  5. 各時段先看前段量價；成交量 >= 500,000、振幅 >= 1.2%
  6. 做多 gap: -1.5% ~ +2.5%；做空 gap: -2.5% ~ +1.5%
  7. abs_pressure 分數：成交額 + 相對量 + 絕對波動 + 方向壓力；做空分數 +10 bias
  8. 非 09:00~09:40 候選分數扣 20，只有明顯強訊號才切到後面時段
  9. 做多/做空同時排序；當天只取第一個可在 tick / 本金限制內交易的候選
 10. 單筆本金 15,000 ~ 20,000；目標為費稅後淨利 +420
 11. 當沖證交稅用 0.15%，買賣手續費各 0.1425%，每邊最低 20
 12. 停損 2.0%，目標最多 6 ticks；同根 K 同時碰停損/停利時停損優先
 13. 各時段都在該時段結束前強制出場
```

Backtest（2024-05-01 → 2026-04-01，全市場 1213 檔 5 分 K）：

| 指標 | 值 |
|------|:---:|
| Calendar days | 465 |
| Active days / trades | 429 / 465 |
| Active ratio | 92.3% |
| Target-hit trades | 61 |
| Target rate | 14.2% |
| Win rate | 57.1% |
| Long ratio | 17.9% |
| Total net PnL | +5,974 |
| Avg trade net PnL | +14 |
| Cutoff violations | 0 |
| Train / test PnL | +5,171 / +802 |

年度拆解：2024 為 +5,566、2025 為 -2,090、2026 為 +2,498。這代表五段時窗 20k 版本已比硬塞每日交易穩定，但仍不是無腦 production；2025 年段為負，正式實盤前必須先 paper trading 驗證：可現股當沖標的、先賣後買券源/券差、實際滑價、各時段掛單成交率、以及 0.15% 當沖稅是否實際適用。

Small-cap daily preset（本金 20,000 內、盡量每天交易）：

```
python daily_meal_money_report.py --preset small-cap-daily --start-date 2024-05-01 --end-date 2026-04-01
```

| 指標 | 值 |
|------|:---:|
| Capital cap | 20,000 |
| Target net PnL | +420 |
| Accepted windows | 09:00~09:40, 10:00~10:20, 11:00~11:20, 11:40~12:20, 12:50~13:30 |
| Active days / trades | 429 / 465 |
| Active ratio | 92.3% |
| Target-hit trades | 61 |
| Target rate | 14.2% |
| Win rate | 57.1% |
| Long ratio | 17.9% |
| Total net PnL | +5,974 |
| Avg trade net PnL | +14 |
| Train / test PnL | +5,171 / +802 |

此 preset 專門服務「小本金但盡量每天交易」：20 日均量 >= 1,000,000、各時段前段量 >= 500,000、振幅 >= 1.2%、相對前段量 <= 2.0，做多/做空雙向取最高 abs_pressure 分數，目標最多 6 ticks，停損 2.0%。勝率改善版使用 reselect-after-feasibility：當天分數第一名若因所需 ticks / 本金等條件不合格，不直接放棄當天，而是依分數改選下一個可交易候選。搜尋中 `max_required_ticks=7` 可提高交易數到 459，但勝率降到 49.2%；單用 09:00~09:40、6 ticks 則是 419 筆、勝率 56.6%、+5,338。加入五段時窗與非早盤扣分後，保留 429 筆交易、勝率 57.1%、+5,974，因此目前採此版本。

Small-cap one-shot preset（本金 20,000 內、交易次數少、單筆淨利至少 +500）：

```
python daily_meal_money_report.py --preset small-cap-one-shot --start-date 2024-05-01 --end-date 2026-04-01
```

| 指標 | 值 |
|------|:---:|
| Capital cap | 20,000 |
| Side | Short only |
| Active days / trades | 23 |
| Target-hit trades | 13 |
| Target rate | 56.5% |
| Win rate | 73.9% |
| Total net PnL | +5,519 |
| Avg trade net PnL | +240 |
| Train / test PnL | +2,455 / +3,065 |

此 preset 專門服務「小本金、少交易、一次達標」：09:00~09:10 需出現極端下跌壓力，振幅 >= 6%，才在 09:15 做空。搜尋中，若採用保守一般股票賣出稅 0.3%，沒有找到同時滿足 train/test 皆正且 target-hit rate >= 55% 的候選；能成立的版本使用現股當沖稅 0.15%。

快速重跑：

```bash
python daily_meal_money_report.py --preset small-cap-daily --start-date 2024-05-01 --end-date 2026-04-01
python daily_meal_money_report.py --mode signals
python daily_meal_money_report.py --preset small-cap-one-shot --start-date 2024-05-01 --end-date 2026-04-01
```

---

## Sector Rotation v2 — 板塊輪動策略

### 三層架構

```
Layer 1: 美股 Macro Regime（前提門檻）
  SPY trend + VIX level → 整體曝險 (0.0 ~ 1.0)
  SPY + SOX 雙空 → 幾乎停止 (0.1)
  VIX > 28 → 完全停止 (0.0)

Layer 2: 板塊資金流（主體選擇）
  7 大板塊的 10/15/20d 平均報酬加權排名
  取前 3 板塊，板塊均報酬 < -3% → 不進場

Layer 3: 板塊內選股
  momentum(20d) × 2 + trend(close > MA60) × 1
  每板塊 Top-3，合計 6~9 檔

出場: ATR TP/SL 4.0/3.0 + 20 天持倉上限
成本: 買 0.143% + 賣 0.443% + 滑價 10bps
```

### 關鍵設計：美股前提

| 條件 | 曝險 | 說明 |
|------|:---:|------|
| SPY↑ + VIX < 22 | **100%** | 全面多頭 |
| SPY↑ + VIX 22~25 | 70% | 輕微恐慌 |
| SPY↓ + VIX < 25 | 40% | 溫和空頭 |
| SPY↓ + VIX 25~28 + SPY > MA20 | 50% | 復甦允許 |
| SPY↓ + VIX 25~28 | 20% | 中等恐慌 |
| SPY + SOX 雙空 | **10%** | 最危險 |
| VIX > 28 | **0%** | 完全停止 |

### SOX 科技門檻 (v1.1: 影響所有板塊)

| 條件 | 效果 |
|------|------|
| SOX > MA60 | 全面開放 |
| SOX < MA60, mom > -3% | **所有板塊半倉**（不只科技） |
| SOX < MA60, mom < -3% | 科技禁止 + 其他半倉 |

---

## 11 段歷史危機壓測（2026-05-26 重算）

| 期間 | SR v2 Sharpe | v8.5 Sharpe | 0050 | SR MDD | VIX 均 |
|------|:---:|:---:|:---:|:---:|:---:|
| 💥 金融海嘯 '08-'09 | -1.63 | -0.86 | 0.95 | -38% | 35 |
| 💥 海嘯復甦 '09-'10 | 0.38 | 1.18 | 2.16 | -15% | 28 |
| 🦠 疫情前 '19Q4 | 1.88 | 0.70 | 1.50 | -7% | 14 |
| 🦠 疫情爆發 '20H1 | 0.63 | 1.04 | -0.34 | -15% | 35 |
| 🦠 疫後牛市 '20-'21 | 2.04 | 2.19 | 2.67 | -26% | 24 |
| ⚔️ 烏俄戰爭 '22H1 | -2.30 | -1.31 | -2.21 | -18% | 27 |
| 📉 升息衝擊 '22 | -2.18 | -1.41 | -2.02 | -26% | 26 |
| 🤖 AI 行情 '23-'24 | 1.99 | 2.48 | 2.55 | -14% | 16 |
| 🏛️ 關稅前一月 '26 | -2.01 | -0.32 | -3.50 | -11% | 26 |
| 🏛️ 關稅衝擊 '26 | 2.31 | 1.42 | 2.12 | -12% | 23 |
| 📊 近期 '26 | 2.93 | 2.38 | 2.73 | -12% | 21 |

### 00981A 對標（共存期，年化口徑）

| 期間 | SR v2 | 00981A | 差距 |
|------|:---:|:---:|:---:|
| 🏛️ 關稅前一月 | -72.6% | -1.0% | 🔴 -71.5% |
| 🏛️ 關稅衝擊 | +139.0% | +21.4% | ✅ +117.6% |
| 📊 近期 | +264.5% | +35.0% | ✅ +229.5% |

### 危機測試解讀

修正 eval window 後，SR v2 在 2022 升息、烏俄戰爭、2026 關稅前一月仍有明顯弱段；它不是單向優於 v8.5 或 0050。SR v2 的優勢主要出現在半導體/電子強趨勢與復甦段，弱點則是全球風險升溫但尚未觸發完全停手時容易被 whipsaw。

00981A 僅在 2025-05 之後有可比資料；早期 crisis 不做 00981A 比較。

---

## v8.5 Momentum 策略（保留）

### Research Gate 決策（2026-05-26）

完整 sweep 後挑出 4 組 finalist：`baseline`、`gap=1.0`、`hold=20+k=4`、`tp=3.5/sl=3.0`，再跑 2021-2025 anchored nested walk-forward。

| Candidate | Avg OOS Sharpe | Min OOS Sharpe | Avg MDD | Worst MDD | Train-selected folds |
|------|:---:|:---:|:---:|:---:|:---:|
| `gap=1.0` | **1.082** | -1.419 | -14.6% | -21.1% | 0/5 |
| `baseline` | 1.078 | -1.159 | -14.8% | -19.8% | 1/5 |
| `hold=20+k=4` | 1.034 | **-0.888** | **-14.3%** | -19.8% | **4/5** |
| `tp=3.5/sl=3.0` | 1.033 | -1.375 | -16.6% | -25.9% | 0/5 |

Nested train-selected portfolio: average OOS Sharpe 1.025，min Sharpe -0.888，max fold PBO 0.23。Candidate-set PBO = 0.94，因此本輪 **不升級新參數到 production**。Production 維持 v8.5 baseline：TP/SL ATR 4.0/3.0、Hold 20D、Top-7、Gap 1.5。

### 績效總覽（1200d：2023-02-13 → 2026-05-26）

| 指標 | 值 | 說明 |
|------|:---:|------|
| **Sharpe** | **2.365** | Arithmetic daily-return Sharpe |
| **Geom. Sharpe** | **3.010** | 年化總報酬 / 年化波動 |
| **年化報酬** | **+76.8%** | 包含交易成本 + 10bps 滑價 |
| **MDD** | **-16.4%** | 使用 raw tradable OHLCV |
| **Calmar** | **4.644** | 年化報酬/MDD |
| **Profit Factor** | **1.95** | 575 筆交易，勝率 57.7% |

### 驗證快照（2026-05-26）

| 測試 | 結果 |
|------|------|
| **v8.5 full period 2019-2026** | 年化 +39.0%，Sharpe 1.562，MDD -35.4%，1348 筆交易 |
| **v8.5 walk-forward OOS** | 4/4 正 Sharpe，3/4 Sharpe ≥ 1.0；平均 1.688，最低 0.537 |
| **OOS decay** | 平均 OOS Sharpe / full-period Sharpe = 1.08 |
| **SR v2 full period 2019-2026** | 年化 +36.6%，Sharpe 1.317，MDD -35.4%；總報酬 +748.7% vs 0050 +593.2% |
| **Monte Carlo v3** | equity_20260526，2000 runs，block=20，seed=42；5% 總報酬 +129.5%，5% MDD -24.8%，中位 Sharpe 2.21 |

### 策略公式

```
每日訊號:
  1. Universe = 過去 20 日平均成交額 Top-60
  2. 綜合評分 = rank_momentum(20d) × 3 + rank_trend(60MA) × 1
  3. 進場: score ≥ 2.0 AND close > 60MA AND 大盤 regime ≥ 40%
  4. 跳空 > 1.5×ATR 的進場日跳過
  5. Top-7 選股（相關性 > 0.8 的替換為不相關候選）

出場 (gap-aware): ATR TP 4.0 / SL 3.0 + 20 天持倉
成本: 買 0.1425% + 賣 0.4425% + 滑價 10bps
```

---

## 快速開始

```bash
pip install -r requirements.txt

# ── v8.5 Momentum ──
python ai_report.py --show-inst

# ── Sector Rotation v2 ──
python sector_rotation_report.py                          # 預設 1200 天
python sector_rotation_report.py --start-date 2019-01-01  # 7 年回測
python sector_rotation_report.py --compare                # vs 0050

# ── 深度危機壓測 (11 段) ──
python deep_crisis_test.py

# ── 驗證工具 ──
python walk_forward.py                                    # OOS 穩定性
python walk_forward_nested.py --quick                     # Nested train→select→test gate
python monte_carlo.py --equity artifacts/equity_YYYYMMDD.csv --runs 2000 --block-size 20
python crisis_test.py                                     # 基礎危機壓測

# ── 研究審計 / 多重測試修正 ──
python sweep.py --quick                                   # 預設寫入 artifacts/experiments.sqlite
python factor_grid_search.py --mode ablation              # 寫入同一個 experiment registry
python -m validation.deflated_sharpe --equity artifacts/equity_YYYYMMDD.csv --trials 20
python -m research.experiment_registry --latest 20

# ── Paper Trading ──
python paper_trade.py signals --enrich
python paper_trade.py hardstop

# ── Meal Money v2（早盤餐費策略）──
python meal_money_report.py --mode backtest --start-date 2024-05-01 --end-date 2026-04-01
python meal_money_report.py --mode signals
python meal_money_report.py --mode backtest --track-us-market
python meal_money_report.py --mode backtest --use-us-market  # SOX hard gate
python meal_money_report.py --mode backtest --night-market-csv artifacts/night_market.csv --use-night-filter
python meal_money_sweep.py --pool extended --max-configs 280

# ── Daily Meal Money（每日雙向量大優先）──
python daily_meal_money_report.py --start-date 2024-05-01 --end-date 2026-04-01
python daily_meal_money_report.py --mode signals
python daily_meal_money_report.py --preset small-cap-daily --start-date 2024-05-01 --end-date 2026-04-01
python daily_meal_money_report.py --preset small-cap-one-shot --start-date 2024-05-01 --end-date 2026-04-01
```

## 研究平台化工具

新增的研究層把「跑過哪些參數」變成可審計紀錄，而不是只留下漂亮表格。

| 模組 | 功能 |
|------|------|
| `research/experiment_registry.py` | SQLite registry，預設 `artifacts/experiments.sqlite`；記錄 git commit、資料快照、假說、參數空間、trial metrics、daily returns、decision |
| `validation/deflated_sharpe.py` | Deflated Sharpe Ratio，修正多重測試與非正態報酬造成的 Sharpe 膨脹 |
| `validation/pbo_cscv.py` | CSCV Probability of Backtest Overfitting，估計「train 內選到的最佳參數在 OOS 掉到下半部」的比例 |
| `walk_forward_nested.py` | Anchored train → inner parameter selection → next-year test；所有候選參數與外層 OOS 結果都寫入 registry |

`sweep.py`、`factor_grid_search.py`、`ablation_study.py` 會預設寫入同一個 registry；若只想臨時跑表格，可加 `--no-registry`。

## 專案結構

```
tw_stocker/
├── ai_report.py                  # v8.5 主程式 + CLI + HTML 報表
├── sector_rotation_report.py     # 🆕 板塊輪動 v2 回測 + 報告
├── meal_money_report.py          # 🆕 一天賺餐費早盤做多策略 CLI
├── meal_money_sweep.py           # 🆕 Meal Money 參數搜尋與 production gate
├── daily_meal_money_report.py    # 🆕 每日雙向量大優先餐費策略 CLI
├── deep_crisis_test.py           # 🆕 11 段歷史危機壓測 + 00981A
├── crisis_test.py                # 基礎危機壓力測試
├── walk_forward.py               # Anchored OOS 穩定性驗證 (v2)
├── walk_forward_nested.py        # Nested train→select→test research gate
├── monte_carlo.py                # Equity-Curve Block Bootstrap (v3)
├── sweep.py                      # 季度參數校準 + Telegram 警報
├── paper_trade.py                # Paper Trading v8 + 月報
├── research/
│   └── experiment_registry.py     # SQLite experiment audit log
├── validation/
│   ├── deflated_sharpe.py         # Deflated Sharpe Ratio
│   └── pbo_cscv.py                # CSCV Probability of Backtest Overfitting
├── strategy/
│   ├── ai_strategy.py            # 因子工程 (Mom×3 + Trend×1)
│   ├── event_backtest.py         # v8.5 事件驅動回測引擎
│   ├── meal_money.py             # 🆕 早盤做多餐費策略回測/信號邏輯
│   ├── daily_meal_money.py       # 🆕 每日雙向量大優先餐費策略
│   ├── us_market.py              # 🆕 美股信號 (SPY/VIX/SOX)
│   ├── sector_rotation_backtest.py # 🆕 板塊輪動回測引擎
│   ├── sector_flow.py            # 板塊資金流分析
│   ├── institutional_flow.py     # 三大法人籌碼因子
│   ├── news_sentiment.py         # 新聞情緒因子
│   ├── risk_metrics.py           # 風險指標計算
│   └── benchmark.py              # Benchmark (0050 / EW)
├── artifacts/                    # 每日 CSV + 月報
├── .github/workflows/
│   └── update_ai_report.yml      # 每日自動執行
└── stock_report.html             # 完整交易報表
```

## 壓力測試方法論

> **資料與評估完整性**：回測可以用 `fetch_start` 提前抓資料暖機，但績效統計只從
> `eval_start` 開始；benchmark / regime filter 會使用相同的 start/end 對齊。
> OHLCV 交易價格不再全欄位 forward-fill，交易只能在 raw open/high/low/close/volume
> 完整且 volume > 0 的日期發生。
>
> ⚠️ **Monte Carlo** (`monte_carlo.py`) 對每日組合報酬率做 equity-curve block bootstrap，
> 保留了多檔同持、regime 縮放、gap sizing 等組合效應。但 bootstrap 仍假設日報酬的
> 時序結構可以被隨機重排——在極端 regime 轉換時這不成立。結果應視為
> **分布估計的參考**，不能直接當作實盤安全邊際。
>
> **OOS 穩定性** (`walk_forward.py`) 是固定參數的分段 OOS 測試；**研究閘門**
> (`walk_forward_nested.py`) 則是 nested train→select→test，用來評估參數搜尋流程本身是否過擬合。
>
> **歷史危機壓測** (`deep_crisis_test.py`) 在 11 段歷史危機做完整回測，
> 含金融海嘯、COVID、烏俄戰爭、升息、關稅衝擊，同時比較 v8.5 / SR v2 / 0050 / 00981A。

## 免責聲明

本系統由 AI 量化模型自動產出，僅供學術研究與技術交流之用，不構成任何投資建議。歷史回測績效不代表未來實際報酬，投資有風險，決策請自行負責。
