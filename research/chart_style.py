# 分析 notebook 共用的圖表格式（顏色、字型、圖例位置），讓各圖風格一致；import 即套用版面
import plotly.graph_objects as go
import plotly.io as pio

# 各量固定顏色（色盲可辨）
COLORS = {
    "Q_COOL": "#2a78d6",  # 冷凍噸：藍
    "kW": "#eb6834",  # 耗電：橘
    "FLOW": "#1baf7a",  # 冰水流量：青綠
    "T_CHWS": "#6250d6",  # 出水溫度：紫
    "T_CHWR": "#e34948",  # 回水溫度：紅
    "temperature": "#c98500",  # 外氣乾球：琥珀
    "WET_BULB": "#008300",  # 外氣濕球：綠（虛線，區分乾球）
    "ENTHALPY": "#e87ba4",  # 外氣焓值：粉紅
    "KW_PER_RT": "#2a78d6",  # 效率：藍
}
# 散佈圖依運轉狀態分組的深淺色
GROUP_SHADES = {
    "Q_COOL": {"只有 CH-2 運轉": "#184f95", "CH-1 + CH-2 同時運轉": "#86b6ef"},
    "T_CHWS": {"只有 CH-2 運轉": "#2a1d7c", "CH-1 + CH-2 同時運轉": "#9f94e6"},
    "kW": {"只有 CH-2 運轉": "#8b310e", "CH-1 + CH-2 同時運轉": "#f09670"},
}
LIGHT_BLUE = "#86b6ef"  # 大量原始資料點
REF_COLOR = "#52514e"  # 擬合線等參考線

# 中文名稱與代號
NAMES = {
    "Q_COOL": "冷凍噸 Q_COOL",
    "kW": "耗電 kW",
    "FLOW": "冰水流量 FLOW",
    "T_CHWS": "出水溫度 T_CHWS",
    "T_CHWR": "回水溫度 T_CHWR",
    "DT": "冰水溫差 DT",
    "temperature": "外氣乾球溫度 temperature",
    "WET_BULB": "外氣濕球溫度 WET_BULB",
    "ENTHALPY": "外氣焓值 ENTHALPY",
    "LR": "負載率 LR",
    "KW_PER_RT": "效率 KW_PER_RT",
}

UNITS = {
    "Q_COOL": "RT",
    "kW": "kW",
    "FLOW": "LPM",
    "T_CHWS": "°C",
    "T_CHWR": "°C",
    "DT": "°C",
    "temperature": "°C",
    "WET_BULB": "°C",
    "ENTHALPY": "kJ/kg",
    "LR": "%",
    "KW_PER_RT": "kW/RT",
}


# 欄位代號轉成座標軸標籤「名稱 (單位)」
def label(col: str) -> str:
    return f"{NAMES[col]} ({UNITS[col]})"


FIG_WIDTH = 1200  # 下載 PNG 的寬度（px）
FIG_HEIGHT = 600  # 單張圖的高度（px）

# 座標軸共用格式
_axis_style = dict(
    showgrid=True,
    gridcolor="#e6e5e0",
    gridwidth=1,
    zeroline=False,
    showline=True,
    linecolor="#8a8984",
    ticks="outside",
    tickcolor="#8a8984",
    title=dict(standoff=8),
)

# 報告用版面範本，設為預設
pio.templates["report"] = go.layout.Template(
    layout=dict(
        font=dict(
            family="Microsoft JhengHei, Noto Sans TC, PingFang TC, sans-serif",
            size=14,
            color="#0b0b0b",
        ),
        title=dict(
            font=dict(size=20),
            x=0.01,
            xanchor="left",
            subtitle=dict(font=dict(size=14, color="#52514e")),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=_axis_style,
        yaxis=_axis_style,
        legend=dict(
            orientation="h",
            x=1,
            xanchor="right",
            y=1.02,
            yanchor="bottom",
        ),
        annotationdefaults=dict(font=dict(size=15)),  # 子圖標題
        hoverlabel=dict(font_size=13),
        margin=dict(l=80, r=30, t=110, b=70),
    ),
)
pio.templates.default = "report"


# 日期軸格式：刻度顯示月/日，滑鼠停留顯示到分鐘
def date_axis(fig: go.Figure) -> None:
    fig.update_xaxes(tickformat="%m/%d", hoverformat="%m/%d %H:%M")


# 子圖標題靠左，和主標題對齊
def left_align_subplot_titles(fig: go.Figure) -> None:
    # 子圖中點 → 左邊界
    left_by_center = {}
    for key, axis in fig.layout.to_plotly_json().items():
        if key.startswith("xaxis") and "domain" in axis:
            left, right = axis["domain"]
            left_by_center[round((left + right) / 2, 4)] = left

    for annotation in fig.layout.annotations:
        left = left_by_center.get(round(annotation.x, 4))
        if left is not None:
            annotation.update(x=left, xanchor="left")


# 分組圖例置於圖下方；trace 以 legend="legend2" 指定組別
def grouped_legends(fig: go.Figure, titles: list, height: int = FIG_HEIGHT) -> None:
    keys = ["legend"] + [f"legend{i}" for i in range(2, len(titles) + 1)]

    # 各組顯示的項目數，取最多的那組
    item_counts = []
    for key in keys:
        item_count = 0
        for trace in fig.data:
            trace_legend = trace.legend or "legend"
            if trace_legend == key and trace.showlegend is not False:
                item_count += 1
        item_counts.append(item_count)
    rows = max(item_counts)
    legend_px = 22 * (rows + 1)  # 最多項目的那組：標題 + 項目高度

    for i, (key, text) in enumerate(zip(keys, titles)):
        legend_layout = dict(
            title=dict(text=f"<b>{text}</b>", side="top"),
            orientation="v",
            x=i / len(titles) + 0.01,
            xanchor="left",
            yref="container",
            y=(legend_px + 12) / height,
            yanchor="top",
            traceorder="normal",
            itemsizing="constant",
            bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout({key: legend_layout})
    fig.update_layout(margin=dict(b=90 + legend_px))


# 套用標題與尺寸後顯示，下載 PNG 時檔名與大小一致；legend 為 "top" 或 "bottom"
def show(
    fig: go.Figure,
    title: str,
    subtitle: str | None = None,
    height: int = FIG_HEIGHT,
    legend: str = "top",
    legend_groups: list | None = None,
) -> None:
    left_align_subplot_titles(fig)
    if legend_groups:
        grouped_legends(fig, legend_groups, height)
    elif legend == "bottom":
        # 兩個以上圖例項目才移到下方
        legend_count = 0
        for trace in fig.data:
            if trace.showlegend is not False:
                legend_count += 1
        if fig.layout.showlegend is not False and legend_count > 1:
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    x=0.5,
                    xanchor="center",
                    yref="container",
                    y=0.02,
                    yanchor="bottom",
                ),
                margin=dict(b=100),
            )

    # 沒有副標時用空字串
    if subtitle:
        subtitle_text = subtitle
    else:
        subtitle_text = ""
    fig.update_layout(
        title=dict(text=title, subtitle=dict(text=subtitle_text)),
        height=height,
    )

    # 下載 PNG 的檔名去掉斜線與空白
    filename = title.replace("/", "、").replace(" ", "")
    fig.show(
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "width": FIG_WIDTH,
                "height": height,
                "scale": 2,
            },
        }
    )
