# 負荷時序圖：畫逐時實際負荷與預測負荷，讓業主看預測與實際的差距；
# 繪圖程式由 /static 提供，案場無外網也能看
import datetime
import time

import pandas as pd
import plotly.graph_objects as go

import settings
from model import recall, train

# 實際淺藍、預測深藍
ACTUAL_COLOR = "#86b6ef"
PRED_COLOR = "#184f95"


# 讀一段期間逐時資料：實際負荷（只到現在以前）與預測負荷 (RT)，時序圖的資料來源；
# index 為台灣時間（無時區）；缺資料為 NaN
async def get_load_frame(start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
    # 實際負荷只到現在這個整點以前
    hour_now = int(time.time()) // 3_600 * 3_600
    actual = await train.get_chiller_hourly(
        start_timestamp,
        min(end_timestamp, hour_now),
    )
    prediction = await recall.get_predictions(start_timestamp, end_timestamp)
    load_frame = pd.DataFrame(
        {
            "actual": actual.set_index("ts_hour")[train.TARGET_COLUMN],
            "pred": prediction,
        }
    ).reindex(range(start_timestamp, end_timestamp, 3_600))
    load_frame.index = settings.to_local_index(load_frame.index, naive=True)
    return load_frame


# 畫 /load-chart：前 24 小時實際與預測、後 24 小時預測
def load_figure(load_frame: pd.DataFrame, now: pd.Timestamp) -> go.Figure:
    prediction = load_frame["pred"]
    past = load_frame[load_frame.index < now.floor("h")]
    # 從前一整點接續
    future = prediction[load_frame.index >= past.index[-1]]

    figure = go.Figure(
        [
            go.Scatter(
                x=past.index,
                y=past["actual"],
                name="實際負荷",
                line=dict(color=ACTUAL_COLOR, width=3),
            ),
            go.Scatter(
                x=past.index,
                y=prediction[past.index],
                name="預測負荷（前 24 小時）",
                line=dict(color=PRED_COLOR, width=2),
            ),
            go.Scatter(
                x=future.index,
                y=future,
                name="預測負荷（後 24 小時）",
                line=dict(color=PRED_COLOR, width=2, dash="dash"),
            ),
        ]
    )

    # 現在的位置：垂直虛線與文字
    figure.add_shape(
        type="line",
        x0=now,
        x1=now,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#52514e", width=1, dash="dot"),
    )
    figure.add_annotation(
        x=now,
        y=1,
        yref="paper",
        text="現在",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        xshift=4,
    )
    return _style(figure, f"更新於 {now:%Y/%m/%d %H:%M}")


# 畫 /load-history：指定期間的實際負荷與預測負荷，回頭檢查預測表現
def history_figure(
    load_frame: pd.DataFrame,
    start_date: datetime.date,
    end_date: datetime.date,
) -> go.Figure:
    figure = go.Figure(
        [
            go.Scatter(
                x=load_frame.index,
                y=load_frame["actual"],
                name="實際負荷",
                line=dict(color=ACTUAL_COLOR, width=3),
            ),
            go.Scatter(
                x=load_frame.index,
                y=load_frame["pred"],
                name="預測負荷",
                line=dict(color=PRED_COLOR, width=2),
            ),
        ]
    )
    subtitle = f"{start_date:%Y/%m/%d} ~ {end_date:%Y/%m/%d}"
    return _style(figure, subtitle)


# 負荷時序圖共用樣式（標題、字型、軸、圖例），兩張圖外觀一致
def _style(figure: go.Figure, subtitle: str) -> go.Figure:
    figure.update_traces(
        mode="lines+markers",
        marker_size=5,
        hovertemplate="%{y:.0f} RT",
    )
    figure.update_layout(
        title=dict(
            text="全館冷凍負荷",
            subtitle=dict(text=subtitle),
            x=0.01,
            xanchor="left",
        ),
        font=dict(
            family="Microsoft JhengHei, Noto Sans TC, PingFang TC, sans-serif",
            size=14,
        ),
        xaxis=dict(tickformat="%m/%d %H:%M", showgrid=True, gridcolor="#e6e5e0"),
        yaxis=dict(
            title="冷凍噸 (RT)",
            rangemode="tozero",
            showgrid=True,
            gridcolor="#e6e5e0",
        ),
        legend=dict(orientation="h", x=0, yref="container", y=0, yanchor="bottom"),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=560,
        margin=dict(t=90, b=150),
    )
    return figure


# 圖轉成網頁：繪圖程式引用本機 /static/plotly.min.js，不連外網
def to_html(figure: go.Figure) -> str:
    return figure.to_html(
        include_plotlyjs="/static/plotly.min.js",
        full_html=True,
        config={"displaylogo": False},
    )
