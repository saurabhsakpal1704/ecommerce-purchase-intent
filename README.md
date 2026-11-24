# E-Commerce Purchase Intent Prediction

A machine-learning demo that estimates whether a browsing session will result in a purchase, with an interpretable feature-importance view.

> **Demo-data note:** generated session records are synthetic and are not customer analytics from a real store.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py
python train.py
streamlit run app.py
```

## Features
- Synthetic session data generator
- Random Forest classification pipeline
- Accuracy, ROC-AUC and classification report
- Streamlit form for session-level demo predictions
- Permutation feature importance for model interpretation

## Columns
`page_views`, `session_duration_min`, `product_views`, `cart_additions`, `returning_visitor`, `device_type`, `traffic_source`, `purchased`.

## Responsible use
This project is an educational demonstration. Do not use synthetic-data model outputs to target, price, rank or make decisions about real customers. Validate on consented, representative real-world data before any operational use.
