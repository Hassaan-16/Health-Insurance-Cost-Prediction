# Health-Insurance-Cost-Prediction

Web app conversion of `Team_7.ipynb` with:
- Backend: Flask API + Gradient Boosting model pipeline
- Frontend: HTML/CSS/JS prediction form

## Run locally

1. Install dependencies:
   `pip install -r requirements.txt`
2. Put the real dataset CSV at either:
   - `insurance.csv` (project root), or
   - `data/insurance.csv`
   (same columns as the notebook dataset)
3. Start server:
   `python app/app.py`
4. Open:
   `http://localhost:5000`

## API endpoints

- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict`
- `POST /api/retrain`

### Predict payload example

```json
{
  "age": 35,
  "bmi": 29.4,
  "children": 2,
  "sex": "male",
  "smoker": "no",
  "region": "southeast"
}
```
