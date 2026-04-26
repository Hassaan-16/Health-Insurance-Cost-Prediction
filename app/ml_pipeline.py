from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "age",
    "bmi",
    "children",
    "sex",
    "smoker",
    "region_northeast",
    "region_northwest",
    "region_southeast",
    "region_southwest",
    "bmi_smoker_int",
]
NUMERIC_COLUMNS = ["age", "bmi", "children", "bmi_smoker_int"]
REGION_COLUMNS = FEATURE_COLUMNS[5:9]
MODEL_GRID = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 4],
}


def to_risk_band(charge: float) -> str:
    if charge < 10000:
        return "Low"
    if charge < 30000:
        return "Medium"
    return "High"


def load_dataset(data_dir: Path, project_root: Path) -> tuple[pd.DataFrame, str]:
    data_dir.mkdir(parents=True, exist_ok=True)
    local_csv = data_dir / "insurance.csv"
    root_csv = project_root / "insurance.csv"
    if root_csv.exists():
        return pd.read_csv(root_csv), str(root_csv)
    if local_csv.exists():
        return pd.read_csv(local_csv), str(local_csv)

    try:
        import kagglehub  # type: ignore

        download_path = Path(kagglehub.dataset_download("mosapabdelghany/medical-insurance-cost-dataset"))
        csv_file = next(download_path.glob("*.csv"))
        df = pd.read_csv(csv_file)
        df.to_csv(local_csv, index=False)
        return df, str(local_csv)
    except Exception as exc:
        raise RuntimeError(
            "Dataset not found. Place the real insurance dataset at "
            "'data/insurance.csv' or project root 'insurance.csv', "
            "or configure Kaggle credentials for kagglehub download."
        ) from exc


def preprocess_training(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean = df.copy()
    clean["sex"] = clean["sex"].map({"male": 1, "female": 0})
    clean["smoker"] = clean["smoker"].map({"yes": 1, "no": 0})

    region_dummies = pd.get_dummies(clean["region"], prefix="region", drop_first=False)
    clean = pd.concat([clean, region_dummies], axis=1)

    for col in REGION_COLUMNS:
        if col not in clean.columns:
            clean[col] = 0

    clean["bmi_smoker_int"] = clean["bmi"] * clean["smoker"]
    x = clean[FEATURE_COLUMNS].copy()
    y = clean["charges"].copy()
    return x, y


def train_and_save(model_path: Path, scaler_path: Path, data_dir: Path, project_root: Path) -> dict[str, Any]:
    df, dataset_path = load_dataset(data_dir, project_root)
    x, y = preprocess_training(df)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_train = x_train.astype({col: "float64" for col in NUMERIC_COLUMNS})
    x_test = x_test.astype({col: "float64" for col in NUMERIC_COLUMNS})

    scaler = StandardScaler()
    x_train.loc[:, NUMERIC_COLUMNS] = scaler.fit_transform(x_train[NUMERIC_COLUMNS])
    x_test.loc[:, NUMERIC_COLUMNS] = scaler.transform(x_test[NUMERIC_COLUMNS])

    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        MODEL_GRID,
        cv=5,
        scoring="r2",
        n_jobs=1,
    )
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    preds = model.predict(x_test)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return {
        "rows": int(df.shape[0]),
        "dataset_path": dataset_path,
        "best_params": grid.best_params_,
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
    }


@dataclass
class ModelArtifacts:
    model: GradientBoostingRegressor
    scaler: StandardScaler


class InsuranceService:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        legacy_model = project_root / "best_gb_model.joblib"
        legacy_scaler = project_root / "scaler.joblib"
        if legacy_model.exists() and legacy_scaler.exists():
            self.model_path = legacy_model
            self.scaler_path = legacy_scaler
        else:
            self.model_path = project_root / "models" / "best_gb_model.joblib"
            self.scaler_path = project_root / "models" / "scaler.joblib"
        self.data_dir = project_root / "data"
        self._artifacts: ModelArtifacts | None = None
        self._training_summary: dict[str, Any] | None = None

    @property
    def training_summary(self) -> dict[str, Any] | None:
        return self._training_summary

    def dataset_info(self) -> dict[str, Any] | None:
        candidates = [self.project_root / "insurance.csv", self.data_dir / "insurance.csv"]
        for csv_path in candidates:
            if csv_path.exists():
                try:
                    rows = int(pd.read_csv(csv_path).shape[0])
                except Exception:
                    rows = None
                return {"path": str(csv_path), "rows": rows}
        return None

    def ensure_loaded(self) -> None:
        if self._artifacts is not None:
            return

        if not self.model_path.exists() or not self.scaler_path.exists():
            self._training_summary = train_and_save(
                self.model_path,
                self.scaler_path,
                self.data_dir,
                self.project_root,
            )

        self._artifacts = ModelArtifacts(
            model=joblib.load(self.model_path),
            scaler=joblib.load(self.scaler_path),
        )

    def retrain(self) -> dict[str, Any]:
        self._training_summary = train_and_save(
            self.model_path,
            self.scaler_path,
            self.data_dir,
            self.project_root,
        )
        self._artifacts = None
        self.ensure_loaded()
        return self._training_summary

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        region = str(payload["region"]).strip().lower()
        if region not in {"northeast", "northwest", "southeast", "southwest"}:
            raise ValueError("Region must be one of northeast, northwest, southeast, southwest.")

        sex = str(payload["sex"]).strip().lower()
        if sex not in {"male", "female"}:
            raise ValueError("Sex must be 'male' or 'female'.")

        smoker = str(payload["smoker"]).strip().lower()
        if smoker not in {"yes", "no"}:
            raise ValueError("Smoker must be 'yes' or 'no'.")

        age = int(payload["age"])
        bmi = float(payload["bmi"])
        children = int(payload["children"])
        actual_charge = payload.get("actual_charge")
        if actual_charge in ("", None):
            actual_charge = None
        elif actual_charge is not None:
            actual_charge = float(actual_charge)

        return {
            "age": age,
            "bmi": bmi,
            "children": children,
            "sex": sex,
            "smoker": smoker,
            "region": region,
            "actual_charge": actual_charge,
        }

    def _payload_to_features(self, normalized: dict[str, Any]) -> pd.DataFrame:
        smoker_flag = 1 if normalized["smoker"] == "yes" else 0

        row = {
            "age": normalized["age"],
            "bmi": normalized["bmi"],
            "children": normalized["children"],
            "sex": 1 if normalized["sex"] == "male" else 0,
            "smoker": smoker_flag,
            "region_northeast": 1 if normalized["region"] == "northeast" else 0,
            "region_northwest": 1 if normalized["region"] == "northwest" else 0,
            "region_southeast": 1 if normalized["region"] == "southeast" else 0,
            "region_southwest": 1 if normalized["region"] == "southwest" else 0,
            "bmi_smoker_int": normalized["bmi"] * smoker_flag,
        }

        frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        return frame

    def _lookup_actual_charge(self, normalized: dict[str, Any]) -> tuple[float | None, int]:
        df, _ = load_dataset(self.data_dir, self.project_root)
        clean = df.copy()
        clean["sex"] = clean["sex"].astype(str).str.strip().str.lower()
        clean["smoker"] = clean["smoker"].astype(str).str.strip().str.lower()
        clean["region"] = clean["region"].astype(str).str.strip().str.lower()

        matches = clean[
            (clean["age"].astype(int) == normalized["age"])
            & (clean["children"].astype(int) == normalized["children"])
            & (clean["sex"] == normalized["sex"])
            & (clean["smoker"] == normalized["smoker"])
            & (clean["region"] == normalized["region"])
            & ((clean["bmi"].astype(float) - normalized["bmi"]).abs() <= 0.005)
        ]
        if matches.empty:
            return None, 0
        return float(matches.iloc[0]["charges"]), int(matches.shape[0])

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.ensure_loaded()
        assert self._artifacts is not None

        normalized = self._normalize_payload(payload)
        features = self._payload_to_features(normalized)
        features = features.astype({col: "float64" for col in NUMERIC_COLUMNS})
        features.loc[:, NUMERIC_COLUMNS] = self._artifacts.scaler.transform(features[NUMERIC_COLUMNS])

        prediction = float(self._artifacts.model.predict(features)[0])
        response = {
            "predicted_charge": round(prediction, 2),
            "risk_band": to_risk_band(prediction),
        }
        actual = normalized["actual_charge"]
        source = "input" if actual is not None else None
        match_count = 0

        if actual is None:
            actual, match_count = self._lookup_actual_charge(normalized)
            if actual is not None:
                source = "dataset"

        if actual is not None:
            diff = prediction - actual
            response.update(
                {
                    "actual_charge": round(float(actual), 2),
                    "actual_charge_source": source,
                    "difference": round(diff, 2),
                    "absolute_difference": round(abs(diff), 2),
                    "dataset_match_count": match_count,
                }
            )
        else:
            response.update(
                {
                    "actual_charge": None,
                    "actual_charge_source": None,
                    "difference": None,
                    "absolute_difference": None,
                    "dataset_match_count": 0,
                }
            )
        return response
