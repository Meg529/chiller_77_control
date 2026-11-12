# 日期特徵：國定假日與時間特徵，負荷模型用（假日、上班時段的負荷差很多）
import holidays
import numpy as np
import pandas as pd


# 是否國定假日（含補假，不含補班）：假日負荷低，模型需要知道
def is_holiday(dates: pd.DatetimeIndex) -> np.ndarray:
    years = sorted(set(dates.year))
    taiwan_holidays = holidays.TW(years=years)
    return np.array([date in taiwan_holidays for date in dates.date], dtype=int)


# 加入時間特徵：小時 sin/cos（讓 23 時與 0 時相鄰）、上班時段、國定假日
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    hour = df.index.hour
    df["HOUR_SIN"] = np.sin(2 * np.pi * hour / 24)
    df["HOUR_COS"] = np.cos(2 * np.pi * hour / 24)
    # 上班時段：08:00 ~ 19:59
    df["IS_WORKING_HOUR"] = ((hour >= 8) & (hour <= 19)).astype(int)
    df["IS_HOLIDAY"] = is_holiday(df.index)
    return df
