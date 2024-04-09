# TW Stocker

每天更新的台股歷史資料庫

## 使用方式，以2330為例，可以換成自己需要的股票

```python
import pandas as pd

url="https://raw.githubusercontent.com/voidful/tw_stocker/main/data/2330.csv"
pd.read_csv(url)
```

## 資料來源
Yahoo finance，每隔5分鐘的六十天內資料，會用github action持續更新。
