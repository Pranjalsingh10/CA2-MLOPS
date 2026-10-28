# Red Wine Quality API

A FastAPI service that serves the red-wine quality model trained in Colab. The
saved artifact is expected at `models/model.joblib` and must be a dictionary
with `model` and `scaler` entries. The API applies that saved scaler before
calling the classifier.

The request uses the 11 input features in the same order used by the UCI/Kaggle
red wine quality dataset: fixed acidity, volatile acidity, citric acid,
residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density,
pH, sulphates, and alcohol.

## Run locally

From the repository root, create a virtual environment and install dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Endpoints

- `GET /` — service links
- `GET /health` — health check
- `POST /predict` — predicted wine quality and probability per class

Example request:

```json
{
  "fixed_acidity": 7.4,
  "volatile_acidity": 0.7,
  "citric_acid": 0.0,
  "residual_sugar": 1.9,
  "chlorides": 0.076,
  "free_sulfur_dioxide": 11.0,
  "total_sulfur_dioxide": 34.0,
  "density": 0.9978,
  "pH": 3.51,
  "sulphates": 0.56,
  "alcohol": 9.4
}
```

## Run with Docker

Build and start the API from the repository root:

```powershell
docker build -t wine-quality-api .
docker run --rm -p 8000:8000 wine-quality-api
```

Test it in another terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Use the JSON request above with `POST http://localhost:8000/predict`, or open
`http://localhost:8000/docs`.

## Tests and GitHub Actions

Run API tests locally using `python -m pytest -q`. The workflow in
`.github/workflows/ci.yml` runs those tests and builds the Docker image for
pushes and pull requests targeting `main` or `master`.

To publish this project, create an empty GitHub repository, then from this
directory run:

```powershell
git init
git add .
git commit -m "Add red wine quality prediction API"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repository>.git
git push -u origin main
```

The GitHub Actions run will appear in the repository's **Actions** tab after
the push. The repository must include `models/model.joblib`; do not commit
secrets or unrelated datasets.
