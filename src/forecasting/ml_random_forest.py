"""Machine Learning: Random Forest (scikit-learn) dengan fitur lag, plus tuning TimeSeriesSplit."""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .data import SEED, STORES

RF_DEFAULT = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3}
RF_GRID = {"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5, 10]}
RF_FEATURES = ["jam", "hari_ke", *[f"cabang_{s}" for s in STORES], "lag_1", "lag_24", "lag_168", "roll_mean_24"]


def _rf_rows(train: pd.DataFrame) -> pd.DataFrame:
    # minggu pertama Januari belum punya lag 168; urut waktu agar TimeSeriesSplit memotong berdasarkan waktu
    return train.dropna(subset=RF_FEATURES).sort_values(["ts", "store_location"])


def train_rf(train: pd.DataFrame, **params) -> RandomForestRegressor:
    # n_jobs=1: urutan penjumlahan antar-pohon tetap, jadi hasil identik persis antar-run
    tr = _rf_rows(train)
    rf = RandomForestRegressor(**{**RF_DEFAULT, **params}, n_jobs=1, random_state=SEED)
    return rf.fit(tr[RF_FEATURES], tr["revenue"])


def tune_rf(train: pd.DataFrame, n_splits: int = 5) -> GridSearchCV:
    """Grid search RF_GRID dengan TimeSeriesSplit di data train saja (fold validasi selalu sesudah fold latih).

    Model terbaik di-refit ke seluruh data train → `best_estimator_`.
    """
    tr = _rf_rows(train)
    gs = GridSearchCV(RandomForestRegressor(n_jobs=1, random_state=SEED), RF_GRID, cv=TimeSeriesSplit(n_splits),
                      scoring="neg_root_mean_squared_error", n_jobs=-1)  # paralel antar-kombinasi, tiap fit tetap deterministik
    return gs.fit(tr[RF_FEATURES], tr["revenue"])


def predict_rf(rf: RandomForestRegressor, test: pd.DataFrame):
    return rf.predict(test[RF_FEATURES])
