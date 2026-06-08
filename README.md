# senior_project_webtool

A Streamlit web tool for estimating expected users and financial viability for
indie turn-based strategy games.

## Planned Features

- Expected users prediction using a 3-class Low/Mid/High model.
- Downside risk prediction using a binary Low vs MidHigh model.
- Upside potential prediction using a binary Mid vs High model.
- Financial viability calculations using pricing, cost, revenue, and user
  assumptions.

## Project Structure

```text
senior_project_webtool/
|-- data/              # Input datasets and prepared training data.
|-- models/            # Trained scikit-learn models saved with joblib.
|-- outputs/           # Reports, charts, prediction exports, and analysis files.
|-- app.py             # Streamlit web app entry point.
|-- train_models.py    # Model training pipeline placeholder.
|-- model_utils.py     # Shared model loading, saving, and prediction utilities.
|-- requirements.txt   # Python dependencies.
`-- README.md          # Project overview and setup notes.
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Train Models

```bash
python train_models.py
```

The current files are placeholders. Model training, prediction logic, and
financial viability calculations will be added in later development steps.
