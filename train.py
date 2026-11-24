from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

path = Path("data/ecommerce_sessions_demo.csv")
if not path.exists():
    raise SystemExit("Run `python generate_data.py` first.")
df = pd.read_csv(path)
X, y = df.drop(columns="purchased"), df.purchased
numeric = ["page_views","session_duration_min","product_views","cart_additions","returning_visitor"]
categorical = ["device_type","traffic_source"]
prep = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
])
pipe = Pipeline([("prep", prep), ("model", RandomForestClassifier(
    n_estimators=250, min_samples_leaf=3, class_weight="balanced", random_state=42))])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
pipe.fit(Xtr, ytr)
pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:,1]
print(f"Accuracy: {accuracy_score(yte,pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(yte,proba):.3f}")
print(classification_report(yte,pred,zero_division=0))
imp = permutation_importance(pipe, Xte, yte, n_repeats=8, random_state=42, scoring="roc_auc")
importance = pd.DataFrame({"feature":X.columns, "importance":imp.importances_mean}).sort_values("importance",ascending=False)
Path("models").mkdir(exist_ok=True)
joblib.dump(pipe,"models/purchase_intent_pipeline.joblib")
importance.to_csv("models/permutation_importance.csv",index=False)
print("Saved model and feature-importance results in models/")
