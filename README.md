# TW Stock Data

台灣市場 OHLCV 歷史資料集與靜態資料瀏覽器。這個 repository 只保存、驗證與呈現原始資料。

資料瀏覽器：[https://voidful.github.io/tw_stocker/](https://voidful.github.io/tw_stocker/)

## 內容

- `data/*.csv`：以股票／ETF 代號分檔的 OHLCV 資料。
- `data/manifest.json`：每個 CSV 的欄位、時間範圍、筆數、檔案大小與 SHA-256。
- `index.html`：可搜尋代號、檢視資料涵蓋範圍及瀏覽近期 K 線資料的靜態頁面。
- `scripts/build_catalog.py`：從 CSV 重建或驗證 manifest。

manifest 內的總檔數、總筆數與資料日期會隨資料內容自動計算，避免 README 出現過期快照。

## CSV 格式

repo 目前包含兩種原始格式：

```text
Date,Open,High,Low,Close,Adj Close,Volume
```

```text
Datetime,Open,High,Low,Close,Volume,Dividends,Stock Splits[,Capital Gains]
```

日期時間保留檔案原有的時區表示。不同代號可能有不同的資料起訖日與欄位組合，使用前請先查閱 [`data/manifest.json`](data/manifest.json)。

## 使用資料

直接下載單一代號：

```bash
curl -LO https://raw.githubusercontent.com/voidful/tw_stocker/main/data/0050.csv
```

Python 標準函式庫讀取：

```python
import csv

with open("data/0050.csv", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
```

重建或檢查資料目錄：

```bash
python scripts/build_catalog.py
python scripts/build_catalog.py --check
```

`--check` 會驗證檔名、支援的欄位結構、首末資料列、筆數與 SHA-256，且不修改檔案。

## 資料聲明

資料可能包含延遲、缺漏、來源端修正或格式差異。此 repo 僅供資料存取與展示；任何使用者都應自行核對資料授權、正確性及適用性。
