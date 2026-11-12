# 77 館水側控制

77 館兩台冰水主機（CH-1、CH-2）的控制建議服務。

依現場監控資料預測當天上班時段各小時冷凍負荷，並建議：

- **台數控制**：開幾台
- **出水溫度控制**：冰水出水溫度設定

系統只提供建議，不會自動操作主機。

## 專案結構

| 路徑 | 說明 |
| --- | --- |
| `app.py` | 服務進入點 |
| `settings.py` | 讀取 `.env` 設定 |
| `data/` | 監控資料、氣象資料 |
| `db/` | 資料庫存取 |
| `model/` | 負荷預測與控制規則 |
| `service/` | 排程與圖表 |
| `analysis/`、`research/` | 數據分析過程 |

## 設定 .env

在專案根目錄建立 `.env`，包含以下設定：

- `POSTGRES_*`：PostgreSQL 連線資訊
- `IOT_*`：現場監控資料 API 與點位
- `CWA_*`、`STATION_*`、`TIMEZONE`：氣象資料與時區
- `CONTROL_*`：控制建議參數
- `JUPYTER_TOKEN`：Jupyter 登入 token

## 執行

需求：Python 3.11、[uv](https://docs.astral.sh/uv/)、PostgreSQL。

```bash
uv sync
uv run uvicorn app:app --host 127.0.0.1 --port 8000
```

開啟 <http://127.0.0.1:8000/docs> 查看 API。
