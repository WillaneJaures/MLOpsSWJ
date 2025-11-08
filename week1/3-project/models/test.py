import joblib
import json
import pandas as pd
import numpy as np
import os

# ---- Chargement des artefacts ----
MODEL_DIR = "./"

forest_model = joblib.load(os.path.join(MODEL_DIR, "forest_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.pkl"))

with open(os.path.join(MODEL_DIR, "feature_list.json"), "r") as f:
    FEATURE_LIST = json.load(f)

with open(os.path.join(MODEL_DIR, "categorical_cols.json"), "r") as f:
    CATEGORICAL_COLS = json.load(f)

print("✅ Modèle et artefacts chargés avec succès !")

# ---- Exemple d'entrée ----
# Remplace les valeurs ci-dessous par des valeurs réalistes de ton dataset
sample_input = {
    "gender": "female",
    "SeniorCitizen": 0,
    "Partner": "yes",
    "Dependents": "no",
    "tenure": 12,
    "PhoneService": "yes",
    "MultipleLines": "no",
    "InternetService": "fiber_optic",
    "OnlineSecurity": "no",
    "OnlineBackup": "no",
    "DeviceProtection": "no",
    "TechSupport": "no",
    "StreamingTV": "yes",
    "StreamingMovies": "yes",
    "Contract": "month_to_month",
    "PaperlessBilling": "yes",
    "PaymentMethod": "electronic_check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 843.5
}

# ---- Transformation identique à celle du notebook ----
def preprocess_input(sample_dict):
    cleaned = {k: (v.lower().strip().replace(" ", "_") if isinstance(v, str) else v)
               for k, v in sample_dict.items()}

    row = {col: cleaned.get(col, np.nan) for col in FEATURE_LIST}
    df = pd.DataFrame([row])

    # ⛔️ on ne refait pas d'encodage : on suppose que FEATURE_LIST est déjà encodé
    df_final = df.fillna(0).astype(float)

    X_scaled = scaler.transform(df_final)
    return X_scaled

# ---- Faire la prédiction ----
X_proc = preprocess_input(sample_input)

proba = forest_model.predict_proba(X_proc)
pred = forest_model.predict(X_proc)

print(f"🧩 Probabilité churn=1 : {proba[0][1]:.4f}")
print(f"🏷️  Classe prédite : {pred[0]}")
