# CH-2 耗電模型（分析用）：建模資料、灰箱線性迴歸（參考 DOE-2，耗電 = f(冷凍噸, 出水溫度)）、
# 出水調整情境、候選模型比較、全部小時分析、卡諾理論驗算
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from data import sensor
from db import db_interaction, db_schema

# ---------- 建模資料：每分鐘資料取穩定運轉後轉逐時 ----------

CHILLER_ID = "77館材化所CH2"
WARMUP_MIN = 60  # 開機暫態排除 (min)
STEP = "1h"  # 建模時間粒度
STEP_HOURS = 1  # 每筆時數 (h)
MIN_STEADY_MIN = 45  # 每小時最少穩定運轉 (min)
LOAD_LIMIT_RT = 360  # 接近滿載門檻 (RT)：建模排除、情境不調整


# 讀 CH-2 每分鐘資料，並算開機後已運轉幾分鐘 RUN_MIN（排除開機暫態用）
async def load_chiller_minute(
    start_time: int | None = None,
    end_time: int | None = None,
) -> pd.DataFrame:
    chiller_data = await sensor.load_chiller(CHILLER_ID, start_time, end_time)
    df = chiller_data.resample("1min").mean()

    # 依開關機分段累加運轉分鐘
    running = sensor.is_running(df)
    segment = (running != running.shift()).cumsum()
    df["RUN_MIN"] = running.groupby(segment).cumsum()
    return df


# 標記接近滿載小時（> LOAD_LIMIT_RT）：這些小時建模排除、情境不調整
def mark_limited(df: pd.DataFrame) -> pd.DataFrame:
    df["LIMITED"] = df["Q_COOL"] > LOAD_LIMIT_RT
    return df


# 只取穩定運轉分鐘（開機超過 WARMUP_MIN 分）做逐時平均；穩定分鐘不足 MIN_STEADY_MIN 的小時捨去
def steady_resample(df_min: pd.DataFrame) -> pd.DataFrame:
    is_steady = (
        (df_min["kW"] > 100)
        & (df_min["Q_COOL"] > 50)
        & (df_min["RUN_MIN"] > WARMUP_MIN)
    )
    steady_minutes = df_min[is_steady]
    hourly = steady_minutes.resample(STEP).mean()
    hourly["STEADY_MIN"] = steady_minutes["kW"].resample(STEP).count()
    return hourly[hourly["STEADY_MIN"] >= MIN_STEADY_MIN].dropna()


# 加入出回水溫差 DT 與日期 DAY（交叉驗證依天分組用）
def add_dt_day(df: pd.DataFrame) -> pd.DataFrame:
    df["DT"] = df["T_CHWR"] - df["T_CHWS"]
    df["DAY"] = df.index.date
    return df


# 讀建模資料：回傳 (每分鐘資料, 穩定運轉逐時資料)
async def load_dataset(
    start_time: int | None = None,
    end_time: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_min = await load_chiller_minute(start_time, end_time)
    steady_hourly = steady_resample(df_min)
    steady_hourly = add_dt_day(steady_hourly)
    steady_hourly = mark_limited(steady_hourly)
    return df_min, steady_hourly


# ---------- 灰箱模型 ----------

TARGET_COL = "kW"
MODEL_ID = "ch2_power_graybox"
MODEL_NAME = "灰箱線性（無氣象）"


# 由流量與溫差算冷凍噸 (RT)
def cooling_rt(d: pd.DataFrame) -> pd.Series:
    return d["FLOW"] * d["DT"] * 4.186 / 60 / 3.517


# 灰箱模型特徵：冷凍噸、冷凍噸平方、出水溫度、冷凍噸 × 出水溫度
def graybox_x(d: pd.DataFrame) -> pd.DataFrame:
    cooling_load = cooling_rt(d)
    return pd.DataFrame(
        {
            "Q": cooling_load,
            "Q2": cooling_load**2,
            "T_CHWS": d["T_CHWS"],
            "Q_x_T_CHWS": cooling_load * d["T_CHWS"],
        },
        index=d.index,
    )


# ASHRAE Guideline 14 指標（CV(RMSE)、NMBE 等）：判斷耗電模型是否夠準
def evaluate(y: pd.Series, pred) -> dict:
    residual = y - pred
    return {
        "CV(RMSE) %": np.sqrt((residual**2).mean()) / y.mean() * 100,
        "NMBE %": residual.sum() / len(y) / y.mean() * 100,
        "MAE kW": residual.abs().mean(),
        "R²": 1 - (residual**2).sum() / ((y - y.mean()) ** 2).sum(),
    }


# 依天分組交叉驗證：同一天的小時不會同時在訓練與測試裡，避免高估準確度
def cross_validate(make_model, make_x, d: pd.DataFrame, n_splits: int = 5) -> pd.Series:
    features = make_x(d)
    target = d[TARGET_COL]
    prediction = pd.Series(index=d.index, dtype=float)

    splitter = GroupKFold(n_splits=n_splits)
    for train_index, test_index in splitter.split(features, target, d["DAY"]):
        model = make_model().fit(features.iloc[train_index], target.iloc[train_index])
        prediction.iloc[test_index] = model.predict(features.iloc[test_index])
    return prediction


# CH-2 耗電模型訓練（排除接近滿載小時）
class PowerModelTrainer:

    # 記下資料起訖時間與模型代號
    def __init__(
        self,
        start_time: int | None = None,
        end_time: int | None = None,
        model_id: str = MODEL_ID,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.model_id = model_id

    # 讀取建模資料，排除接近滿載小時
    async def get_data(self) -> None:
        self.df_min, self.df_steady = await load_dataset(self.start_time, self.end_time)
        self.df = self.df_steady[~self.df_steady["LIMITED"]].copy()

    # 固定採用灰箱線性（候選比較見 AnalysisTrainer）
    def select(self) -> None:
        self.model_name = MODEL_NAME
        self.make_model = LinearRegression
        self.make_x = graybox_x

    # 訓練模型、交叉驗證預測與指標
    def train_model(self) -> None:
        features = self.make_x(self.df)
        target = self.df[TARGET_COL]
        self.model = self.make_model().fit(features, target)
        self.df["kW_CV"] = cross_validate(self.make_model, self.make_x, self.df)
        self.df["kW_PRED"] = self.model.predict(features)
        self.metric = evaluate(target, self.df["kW_CV"])

    # 模型係數（含截距），notebook 顯示用
    def coefficients(self) -> pd.Series:
        feature_names = self.make_x(self.df).columns
        coefficient = pd.Series(self.model.coef_, index=feature_names)
        intercept = pd.Series({"截距": self.model.intercept_})
        return pd.concat([intercept, coefficient])

    # 完整訓練流程：讀資料 → 選模 → 訓練
    async def main(self) -> None:
        await self.get_data()
        self.select()
        self.train_model()


# 把 CH-2 耗電模型存入 model_registry
class PowerModelSaver:

    # 記下要存的訓練結果
    def __init__(self, trainer: PowerModelTrainer):
        self.trainer = trainer

    # 存入模型、指標與特徵欄位（同 model_id 覆蓋）
    async def save_to_db(self) -> None:
        metrics = {}
        for metric_name, metric_value in self.trainer.metric.items():
            metrics[metric_name] = float(metric_value)
        feature_cols = list(self.trainer.make_x(self.trainer.df).columns)

        async with db_schema.async_session() as session:
            db_manager = db_interaction.DBManager(session)
            await db_manager.save_model(
                model_id=self.trainer.model_id,
                model_type=type(self.trainer.model).__name__,
                model_obj=self.trainer.model,
                metrics=metrics,
                features={
                    "step": STEP,
                    "model_name": self.trainer.model_name,
                    "feature_cols": feature_cols,
                },
            )


# ---------- 出水調整情境：出水、回水同步調整，情境與基準預測相減為節電量 ----------


# 不調整出水的小時：接近滿載或剛開機
def no_shift(d: pd.DataFrame) -> pd.Series:
    return d["LIMITED"] | d.get("STARTUP", False)


# 產生出水、回水同時加 delta 度的情境資料
def shift_chws(d: pd.DataFrame, delta: float) -> pd.DataFrame:
    shifted = d.copy()
    shifted.loc[~no_shift(shifted), ["T_CHWS", "T_CHWR"]] += delta
    return shifted


# 出水調整 delta 後的逐筆耗電 (kW)
def predict_shift(model, d: pd.DataFrame, delta: float, make_x=graybox_x) -> pd.Series:
    shifted_features = make_x(shift_chws(d, delta))
    return pd.Series(model.predict(shifted_features), index=d.index)


# 出水調整 delta 後的總耗電變化 (%)：估出水調高能省多少電
def change_pct(model, d: pd.DataFrame, delta: float, make_x=graybox_x) -> float:
    base_kw = model.predict(make_x(d)).sum()
    shifted_kw = predict_shift(model, d, delta, make_x).sum()
    return (shifted_kw - base_kw) / base_kw * 100


# ---------- 候選模型比較：交叉驗證準確度與出水調整的耗電方向 ----------

RAW_COLS = ["T_CHWS", "T_CHWR", "FLOW"]
DT_COLS = ["T_CHWS", "DT", "FLOW"]


# 建 LightGBM 候選模型；monotone 時限制各特徵影響方向（出水高耗電低、溫差與流量大耗電高）
def _lgbm(feature_columns: list, monotone: bool = False) -> LGBMRegressor:
    direction_by_feature = {"T_CHWS": -1, "DT": 1, "FLOW": 1}
    if monotone:
        monotone_constraints = [
            direction_by_feature.get(column, 0) for column in feature_columns
        ]
    else:
        monotone_constraints = None
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        monotone_constraints=monotone_constraints,
    )


# ---------- 建立模型 ----------


# 候選模型：LightGBM，用出水、回水溫度、流量
def lgbm_raw() -> LGBMRegressor:
    return _lgbm(RAW_COLS)


# 候選模型：LightGBM，用出水溫度、溫差、流量
def lgbm_dt() -> LGBMRegressor:
    return _lgbm(DT_COLS)


# 候選模型：LightGBM，用出水溫度、溫差、流量並加上單調限制
def lgbm_dt_monotone() -> LGBMRegressor:
    return _lgbm(DT_COLS, monotone=True)


# ---------- 建立特徵 ----------


# 候選特徵：出水、回水溫度、流量
def raw_x(d: pd.DataFrame) -> pd.DataFrame:
    return d[RAW_COLS]


# 候選特徵：出水溫度、溫差、流量
def dt_x(d: pd.DataFrame) -> pd.DataFrame:
    return d[DT_COLS]


# 候選模型：名稱 → (模型, 特徵)
CANDIDATES = {
    "LightGBM": (lgbm_raw, raw_x),
    "LightGBM（溫差）": (lgbm_dt, dt_x),
    "LightGBM（溫差 + 單調限制）": (lgbm_dt_monotone, dt_x),
    MODEL_NAME: (LinearRegression, graybox_x),
}


# 比較各候選模型：交叉驗證指標與出水調整的耗電變化，確認選用模型的方向合理
def compare_models(d: pd.DataFrame, deltas: tuple = (0.5, -0.5)) -> pd.DataFrame:
    rows = []
    for name, (make_model, make_x) in CANDIDATES.items():
        prediction = cross_validate(make_model, make_x, d)
        # 全部資料訓練，用來算情境
        model = make_model().fit(make_x(d), d[TARGET_COL])

        row = {"模型": name}
        row.update(evaluate(d[TARGET_COL], prediction))
        for delta in deltas:
            row[f"出水 {delta:+.1f}°C 耗電變化 %"] = change_pct(model, d, delta, make_x)
        rows.append(row)
    return pd.DataFrame(rows).set_index("模型")


# 固定採用灰箱線性，並檢查出水調高時耗電下降（方向不對就報錯）
def select_model(comparison: pd.DataFrame) -> str:
    if comparison.loc[MODEL_NAME, "出水 +0.5°C 耗電變化 %"] >= 0:
        raise ValueError(
            f"{MODEL_NAME} 出水溫度調高時耗電沒有下降，方向不合理，需檢查資料"
        )
    return MODEL_NAME


# ---------- 分析：選模、全部小時預測、情境區間 ----------


# 全部小時（含停機）逐時資料，含運轉比例、開機小時標記：估全期間耗電用
def hourly_all(df_min: pd.DataFrame) -> pd.DataFrame:
    step = STEP
    running_minutes = df_min[sensor.is_running(df_min)]
    hourly = (
        running_minutes[["FLOW", "T_CHWS", "T_CHWR", "Q_COOL"]].resample(step).mean()
    )
    hourly["RUN_FRAC"] = running_minutes["kW"].resample(step).count() / 60

    # 開機小時（資料起點已運轉者除外）
    startup_minute = (running_minutes["RUN_MIN"] == 1) & (
        running_minutes.index > df_min.index.min()
    )
    hourly["STARTUP"] = startup_minute.resample(step).max()

    # 補齊全部小時（含停機）
    hours = pd.date_range(
        df_min.index.min().floor(step),
        df_min.index.max().floor(step),
        freq=step,
    )
    hourly = hourly.reindex(hours)
    hourly["kW"] = df_min["kW"].resample(step).mean().reindex(hours)
    hourly["RUN_FRAC"] = hourly["RUN_FRAC"].fillna(0)
    hourly["STARTUP"] = hourly["STARTUP"].fillna(False).astype(bool)
    hourly["RUNNING"] = (hourly["RUN_FRAC"] > 0) & (hourly["Q_COOL"] > 50)

    hourly["DT"] = hourly["T_CHWR"] - hourly["T_CHWS"]
    return hourly


# 分析用訓練：比較候選模型後選模，並預測全部小時耗電
class AnalysisTrainer(PowerModelTrainer):

    # 除了建模資料，另外準備全部小時資料
    async def get_data(self) -> None:
        await super().get_data()
        self.df_all = mark_limited(hourly_all(self.df_min))

    # 比較候選模型後選模
    def select(self) -> None:
        self.comparison = compare_models(self.df)
        self.model_name = select_model(self.comparison)
        self.make_model, self.make_x = CANDIDATES[self.model_name]

    # 有運轉且特徵齊全的小時
    def running_hours(self) -> pd.DataFrame:
        running = self.df_all[self.df_all["RUNNING"]]
        has_all_features = self.make_x(running).notna().all(axis=1)
        return running[has_all_features]

    # 預測全部小時耗電：運轉時的預測 × 運轉比例，並算全部小時的指標
    def predict_all(self) -> None:
        hourly = self.df_all
        running = self.running_hours()
        hourly["kW_PRED"] = 0.0
        running_prediction = self.model.predict(self.make_x(running))
        hourly.loc[running.index, "kW_PRED"] = running_prediction * running["RUN_FRAC"]

        # 缺特徵設 NaN
        missing_features = hourly["RUNNING"] & ~hourly.index.isin(running.index)
        hourly.loc[missing_features, "kW_PRED"] = np.nan

        valid = hourly[["kW", "kW_PRED"]].dropna()
        self.metric_all = evaluate(valid["kW"], valid["kW_PRED"])

    # 訓練後再預測全部小時
    async def main(self) -> None:
        await super().main()
        self.predict_all()


# 依天重抽樣重訓，回傳出水調整耗電變化 % 的 95% 區間：看節電估計的不確定程度
def bootstrap_pct(
    trainer: AnalysisTrainer,
    delta: float,
    n: int = 500,
    seed: int = 0,
) -> tuple:
    make_model = trainer.make_model
    make_x = trainer.make_x
    steady_hourly = trainer.df_steady
    training_data = trainer.df

    rng = np.random.default_rng(seed)
    days = training_data["DAY"].unique()
    changes = []
    for _ in range(n):
        sampled_days = rng.choice(days, len(days))
        sample = pd.concat(
            [training_data[training_data["DAY"] == day] for day in sampled_days]
        )
        model = make_model().fit(make_x(sample), sample[TARGET_COL])

        shifted_kw = predict_shift(model, steady_hourly, delta, make_x).sum()
        base_kw = model.predict(make_x(steady_hourly)).sum()
        change = shifted_kw - base_kw
        changes.append(change / steady_hourly[TARGET_COL].sum() * 100)
    return tuple(np.percentile(changes, [2.5, 97.5]))


# 逐時情境耗電：BASE（現況）與各出水調整 delta
def hourly_scenario(trainer: AnalysisTrainer, deltas) -> pd.DataFrame:
    running = trainer.running_hours()
    scenario = pd.DataFrame(index=trainer.df_all.index)
    scenario["BASE"] = trainer.df_all["kW_PRED"]
    for delta in deltas:
        scenario[delta] = scenario["BASE"]
        shifted_kw = predict_shift(trainer.model, running, delta, trainer.make_x)
        scenario.loc[running.index, delta] = shifted_kw * running["RUN_FRAC"]
    return scenario


# ---------- 理論驗算：卡諾冷凍機的出水調整耗電變化 ----------

T_COND = 34.0  # 冷凝溫度 (°C)：冷卻水出水 32~33°C + 1~2°C
EVAP_DT = 2.0  # 蒸發溫度 = 出水溫度 − 2°C


# 卡諾冷凍機的理想耗電（相對值）：理論驗算出水調整的影響
def ideal_kw(d: pd.DataFrame) -> pd.Series:
    evaporating_kelvin = d["T_CHWS"] - EVAP_DT + 273.15
    condensing_kelvin = T_COND + 273.15
    return cooling_rt(d) * (condensing_kelvin - evaporating_kelvin) / evaporating_kelvin


# 出水調整 delta 的理想耗電變化 (%)：和模型估計對照
def ideal_change_pct(d: pd.DataFrame, delta: float) -> float:
    base_kw = ideal_kw(d).sum()
    shifted_kw = ideal_kw(shift_chws(d, delta)).sum()
    return (shifted_kw - base_kw) / base_kw * 100
