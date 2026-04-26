from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, render_template, request

try:
    from .ml_pipeline import InsuranceService
except ImportError:
    from ml_pipeline import InsuranceService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
service = InsuranceService(PROJECT_ROOT)

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/model-info")
def model_info():
    try:
        service.ensure_loaded()
        return jsonify(
            {
                "features": [
                    "age",
                    "sex",
                    "bmi",
                    "children",
                    "smoker",
                    "region",
                    "bmi_smoker_int (derived)",
                ],
                "model": "GradientBoostingRegressor",
                "training_summary": service.training_summary,
                "dataset_info": service.dataset_info(),
            }
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    try:
        result = service.predict(payload)
        return jsonify(result)
    except KeyError as exc:
        return jsonify({"error": f"Missing required field: {exc.args[0]}"}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:
        return jsonify({"error": "Prediction failed due to invalid input."}), 400


@app.post("/api/retrain")
def retrain():
    try:
        summary = service.retrain()
        return jsonify({"status": "retrained", "summary": summary})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
