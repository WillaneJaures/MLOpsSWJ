from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import joblib
import json
import pandas as pd
import uvicorn
import os
import numpy as np

app = FastAPI(title="Churn Predictor API")

# ---- Chargement des artefacts au démarrage ----
MODEL_DIR = './models'
model = None
scaler = None
encoder = None
metrics = None
FEATURE_LIST = None
CATEGORICAL_COLS = None

try:
    model = joblib.load(os.path.join(MODEL_DIR, 'logr_model.pkl'))
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    print("✅ Scaler chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement du scaler: {e}")

try:
    encoder = joblib.load(os.path.join(MODEL_DIR, 'encoder.pkl'))
    print("✅ Encoder chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement de l'encoder: {e}")

try:
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    print("✅ Métriques chargées avec succès")
except UnicodeDecodeError:
    print(f"❌ metrics.json n'est pas au format UTF-8")
    print("   Utilisez json.dump() au lieu de joblib.dump()")
    metrics = None
except Exception as e:
    print(f"❌ Erreur lors du chargement des métriques: {e}")
    metrics = None

try:
    with open(os.path.join(MODEL_DIR, 'feature_list.json'), 'r') as f:
        FEATURE_LIST = json.load(f)
    print("✅ Feature list chargée avec succès")
except Exception as e:
    print(f"⚠️  Fichier feature_list.json non trouvé (optionnel): {e}")

try:
    with open(os.path.join(MODEL_DIR, 'categorical_cols.json'), 'r') as f:
        CATEGORICAL_COLS = json.load(f)
    print("✅ Categorical cols chargées avec succès")
except Exception as e:
    print(f"⚠️  Fichier categorical_cols.json non trouvé (optionnel): {e}")

# ---- Schéma d'entrée avec validation stricte ----
# ---- Schéma d'entrée avec validation stricte ----
class CustomerData(BaseModel):
    gender: Literal["Male", "Female", "male", "female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No", "yes", "no"]
    Dependents: Literal["Yes", "No", "yes", "no"]
    tenure: int
    PhoneService: Literal["Yes", "No", "yes", "no"]
    MultipleLines: Literal["Yes", "No", "No phone service", "yes", "no", "no phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No", "dsl", "fiber optic", "no"]
    OnlineSecurity: Literal["Yes", "No", "No internet service", "yes", "no", "no internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service", "yes", "no", "no internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service", "yes", "no", "no internet service"]
    TechSupport: Literal["Yes", "No", "No internet service", "yes", "no", "no internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service", "yes", "no", "no internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service", "yes", "no", "no internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year", "month-to-month", "one year", "two year"]
    PaperlessBilling: Literal["Yes", "No", "yes", "no"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)", 
        "electronic check", "mailed check", "bank transfer (automatic)", "credit card (automatic)"
    ]
    MonthlyCharges: float
    TotalCharges: Optional[float] = None
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "Yes",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.50,
                "TotalCharges": 786.00
            }
        }

@app.get("/")
def read_root():
    return {
        "message": "API de prédiction de churn opérationnelle",
        "status": "healthy",
        "endpoints": {
            "/predict": "POST - Prédire le churn d'un client",
            "/health": "GET - Vérifier la santé de l'API",
            "/metrics": "GET - Obtenir les métriques du modèle"
        }
    }

@app.post("/predict")
def predict(customer_data: CustomerData):
    """
    Endpoint pour prédire le churn d'un client
    """
    if model is None or encoder is None or scaler is None:
        raise HTTPException(
            status_code=500,
            detail="Le modèle, l'encoder ou le scaler ne sont pas chargés."
        )
    
    try:
        # Fonction pour normaliser les valeurs (espaces -> underscores, minuscules)
        def normalize_value(value):
            if isinstance(value, str):
                return value.lower().replace(' ', '_')
            return value
        
        # Créer un DataFrame avec les données du client
        # Normaliser pour correspondre au format de l'encoder
        data_dict = {
            'gender': normalize_value(customer_data.gender),
            'seniorcitizen': customer_data.SeniorCitizen,
            'partner': normalize_value(customer_data.Partner),
            'dependents': normalize_value(customer_data.Dependents),
            'tenure': customer_data.tenure,
            'phoneservice': normalize_value(customer_data.PhoneService),
            'multiplelines': normalize_value(customer_data.MultipleLines),
            'internetservice': normalize_value(customer_data.InternetService),
            'onlinesecurity': normalize_value(customer_data.OnlineSecurity),
            'onlinebackup': normalize_value(customer_data.OnlineBackup),
            'deviceprotection': normalize_value(customer_data.DeviceProtection),
            'techsupport': normalize_value(customer_data.TechSupport),
            'streamingtv': normalize_value(customer_data.StreamingTV),
            'streamingmovies': normalize_value(customer_data.StreamingMovies),
            'contract': normalize_value(customer_data.Contract),
            'paperlessbilling': normalize_value(customer_data.PaperlessBilling),
            'paymentmethod': normalize_value(customer_data.PaymentMethod),
            'monthlycharges': customer_data.MonthlyCharges,
            'totalcharges': customer_data.TotalCharges if customer_data.TotalCharges is not None else 0.0
        }
        
        # Créer un DataFrame
        df_input = pd.DataFrame([data_dict])
        
        # Séparer les colonnes catégorielles et numériques
        object_cols = df_input.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df_input.select_dtypes(exclude=['object']).columns.tolist()
        
        # Encoder les colonnes catégorielles
        df_categorical = df_input[object_cols]
        
        try:
            encoded_array = encoder.transform(df_categorical)
        except ValueError as e:
            # Debug en cas d'erreur d'encodage
            print(f"❌ Erreur d'encodage: {str(e)}")
            print(f"Valeurs reçues: {df_categorical.to_dict('records')[0]}")
            
            if hasattr(encoder, 'categories_'):
                print("Catégories attendues:")
                for i, col in enumerate(object_cols):
                    print(f"  {col}: {list(encoder.categories_[i])}")
            
            raise HTTPException(
                status_code=400,
                detail=f"Valeur invalide détectée dans les données catégorielles. {str(e)}"
            )
        
        feature_names = encoder.get_feature_names_out(object_cols)
        df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df_input.index)
        
        # Combiner avec les colonnes numériques
        df_numeric = df_input[numeric_cols]
        df_final = pd.concat([df_numeric, df_encoded], axis=1)
        
        # Appliquer le scaler
        try:
            df_scaled = scaler.transform(df_final)
            df_scaled = pd.DataFrame(df_scaled, columns=df_final.columns, index=df_final.index)
        except ValueError as e:
            print(f"❌ Erreur de scaling: {str(e)}")
            print(f"Features attendues: {scaler.n_features_in_}")
            print(f"Features reçues: {df_final.shape[1]}")
            
            raise HTTPException(
                status_code=500,
                detail=f"Erreur de preprocessing: {str(e)}. Le scaler a peut-être été entraîné avec 'churn' inclus."
            )
        
        # Faire la prédiction
        prediction = model.predict(df_scaled)[0]
        
        # Probabilité
        try:
            probability = model.predict_proba(df_scaled)[0]
            churn_probability = float(probability[1]) if len(probability) > 1 else float(probability[0])
        except:
            churn_probability = None
        
        # Conversion de la prédiction
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            churn_prediction = "Yes" if prediction == 1 or prediction == 1.0 else "No"
        else:
            churn_prediction = "Yes" if str(prediction).lower() in ['yes', '1', 'true'] else "No"
        
        return {
            "churn_prediction": churn_prediction,
            "churn_probability": churn_probability,
            "customer_data": customer_data.dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Erreur inattendue: {str(e)}")
        raise HTTPException(
            status_code=400, 
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@app.get("/health")
def health():
    """Vérification de la santé de l'API et du modèle"""
    try:
        model_ok = model is not None and hasattr(model, 'predict')
        encoder_ok = encoder is not None
        scaler_ok = scaler is not None
        metrics_ok = metrics is not None
        
        all_ok = model_ok and encoder_ok and scaler_ok
        
        status_details = {
            "status": "healthy" if all_ok else "unhealthy",
            "model_loaded": model_ok,
            "encoder_loaded": encoder_ok,
            "scaler_loaded": scaler_ok,
            "metrics_loaded": metrics_ok,
            "model_type": str(type(model).__name__) if model is not None else None,
        }
        
        # Ajouter des infos de debug
        if scaler is not None and hasattr(scaler, 'n_features_in_'):
            status_details["scaler_features"] = scaler.n_features_in_
        
        if model is not None and hasattr(model, 'n_features_in_'):
            status_details["model_features"] = model.n_features_in_
        
        status_details["message"] = (
            "API et tous les composants opérationnels" if all_ok 
            else "Certains composants ne sont pas chargés"
        )
        
        return status_details
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur de santé: {str(e)}"
        }

@app.get("/metrics")
def get_metrics():
    """
    Endpoint pour récupérer les métriques du modèle
    """
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="Les métriques ne sont pas disponibles. Régénérez metrics.json avec json.dump()."
        )
    
    return {
        "metrics": metrics,
        "message": "Métriques chargées depuis models/metrics.json"
    }

@app.get("/debug/categories")
def debug_categories():
    """
    Afficher les catégories exactes attendues par l'encoder
    """
    if encoder is None or not hasattr(encoder, 'categories_'):
        raise HTTPException(status_code=500, detail="Encoder non disponible")
    
    # Récupérer les noms des colonnes catégorielles
    if hasattr(encoder, 'feature_names_in_'):
        col_names = list(encoder.feature_names_in_)
    else:
        col_names = [f"col_{i}" for i in range(len(encoder.categories_))]
    
    categories_dict = {}
    for i, col in enumerate(col_names):
        categories_dict[col] = list(encoder.categories_[i])
    
    return {
        "categories": categories_dict,
        "total_columns": len(col_names),
        "message": "Valeurs EXACTES attendues par l'encoder (respectez la casse)"
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Démarrage de l'API Churn Predictor")
    print("="*50)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)