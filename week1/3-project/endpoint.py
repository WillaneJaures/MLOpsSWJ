from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
import pandas as pd
import json
from pydantic import BaseModel
from typing import Optional

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API pour prédire le churn des clients",
    version="1.0.0"
)

# Modèle Pydantic pour les données d'entrée
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 358.2
            }
        }

# Chargement des objets (modèle, encoder, scaler, métriques)
def load_model():
    """Charge le modèle depuis le fichier model.pkl"""
    try:
        model_path = "./models/model.pkl"
        loaded_obj = joblib.load(model_path)
        
        # Vérifier si c'est un numpy array (prédictions sauvegardées par erreur)
        if isinstance(loaded_obj, np.ndarray):
            raise ValueError(
                "Le fichier model.pkl contient des prédictions (numpy array) au lieu du modèle. "
                "Vous devez sauvegarder le modèle RandomForestClassifier, pas les prédictions. "
                "Utilisez: joblib.dump(Forest, 'models/model.pkl') au lieu de joblib.dump(Forest_pred, 'models/model.pkl')"
            )
        
        # Vérifier si l'objet a une méthode predict (c'est un modèle)
        if not hasattr(loaded_obj, 'predict'):
            raise ValueError(
                f"L'objet chargé n'est pas un modèle valide. Type: {type(loaded_obj)}. "
                "Il doit avoir une méthode 'predict'."
            )
        
        return loaded_obj
    except ValueError as e:
        raise ValueError(str(e))
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle: {str(e)}")

def load_encoder():
    """Charge l'encoder depuis le fichier encoder.pkl"""
    try:
        encoder_path = "./models/encoder.pkl"
        encoder = joblib.load(encoder_path)
        return encoder
    except Exception as e:
        raise Exception(f"Erreur lors du chargement de l'encoder: {str(e)}")

def load_scaler():
    """Charge le scaler depuis le fichier scaler.pkl"""
    try:
        scaler_path = "./models/scaler.pkl"
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du scaler: {str(e)}")

def load_metrics():
    """Charge les métriques depuis le fichier metrics.json"""
    try:
        metrics_path = "./models/metrics.json"
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        return None

# Chargement au démarrage
model = None
encoder = None
scaler = None
metrics = None

try:
    model = load_model()
except (ValueError, Exception) as e:
    print(f"⚠️  Erreur lors du chargement du modèle: {e}")

try:
    encoder = load_encoder()
except Exception as e:
    print(f"⚠️  Erreur lors du chargement de l'encoder: {e}")

try:
    scaler = load_scaler()
except Exception as e:
    print(f"⚠️  Erreur lors du chargement du scaler: {e}")

try:
    metrics = load_metrics()
except Exception as e:
    print(f"⚠️  Erreur lors du chargement des métriques: {e}")

@app.get("/")
async def root():
    """Endpoint de santé de l'API"""
    return {
        "message": "API de prédiction de churn opérationnelle",
        "status": "healthy"
    }

@app.post("/predict")
async def predict_churn(customer_data: CustomerData):
    """
    Endpoint pour prédire le churn d'un client
    
    Args:
        customer_data: Données du client au format CustomerData
    
    Returns:
        dict: Prédiction du churn (Yes/No) et probabilité
    """
    if model is None:
        raise HTTPException(
            status_code=500, 
            detail="Le modèle n'a pas pu être chargé. Vérifiez que le fichier model.pkl contient un modèle valide."
        )
    
    if encoder is None:
        raise HTTPException(
            status_code=500, 
            detail="L'encoder n'a pas pu être chargé. Vérifiez que le fichier encoder.pkl existe."
        )
    
    if scaler is None:
        raise HTTPException(
            status_code=500, 
            detail="Le scaler n'a pas pu être chargé. Vérifiez que le fichier scaler.pkl existe."
        )
    
    try:
        # Créer un DataFrame avec les données du client
        # Note: Les noms de colonnes doivent correspondre à ceux utilisés lors de l'entraînement
        data_dict = {
            'gender': customer_data.gender,
            'seniorcitizen': customer_data.SeniorCitizen,  # Note: minuscule dans le notebook
            'partner': customer_data.Partner.lower() if isinstance(customer_data.Partner, str) else customer_data.Partner,
            'dependents': customer_data.Dependents.lower() if isinstance(customer_data.Dependents, str) else customer_data.Dependents,
            'tenure': customer_data.tenure,
            'phoneservice': customer_data.PhoneService.lower() if isinstance(customer_data.PhoneService, str) else customer_data.PhoneService,
            'multiplelines': customer_data.MultipleLines.lower() if isinstance(customer_data.MultipleLines, str) else customer_data.MultipleLines,
            'internetservice': customer_data.InternetService.lower() if isinstance(customer_data.InternetService, str) else customer_data.InternetService,
            'onlinesecurity': customer_data.OnlineSecurity.lower() if isinstance(customer_data.OnlineSecurity, str) else customer_data.OnlineSecurity,
            'onlinebackup': customer_data.OnlineBackup.lower() if isinstance(customer_data.OnlineBackup, str) else customer_data.OnlineBackup,
            'deviceprotection': customer_data.DeviceProtection.lower() if isinstance(customer_data.DeviceProtection, str) else customer_data.DeviceProtection,
            'techsupport': customer_data.TechSupport.lower() if isinstance(customer_data.TechSupport, str) else customer_data.TechSupport,
            'streamingtv': customer_data.StreamingTV.lower() if isinstance(customer_data.StreamingTV, str) else customer_data.StreamingTV,
            'streamingmovies': customer_data.StreamingMovies.lower() if isinstance(customer_data.StreamingMovies, str) else customer_data.StreamingMovies,
            'contract': customer_data.Contract.lower() if isinstance(customer_data.Contract, str) else customer_data.Contract,
            'paperlessbilling': customer_data.PaperlessBilling.lower() if isinstance(customer_data.PaperlessBilling, str) else customer_data.PaperlessBilling,
            'paymentmethod': customer_data.PaymentMethod.lower() if isinstance(customer_data.PaymentMethod, str) else customer_data.PaymentMethod,
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
        encoded_array = encoder.transform(df_categorical)
        feature_names = encoder.get_feature_names_out(object_cols)
        df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df_input.index)
        
        # Combiner avec les colonnes numériques
        df_numeric = df_input[numeric_cols]
        df_final = pd.concat([df_numeric, df_encoded], axis=1)
        
        # Appliquer le scaler
        df_scaled = scaler.transform(df_final)
        df_scaled = pd.DataFrame(df_scaled, columns=df_final.columns, index=df_final.index)
        
        # Faire la prédiction
        prediction = model.predict(df_scaled)[0]
        
        # Probabilité (si le modèle le supporte)
        try:
            probability = model.predict_proba(df_scaled)[0]
            churn_probability = float(probability[1]) if len(probability) > 1 else float(probability[0])
        except:
            churn_probability = None
        
        # Conversion de la prédiction en format lisible
        if isinstance(prediction, (int, float, np.integer, np.floating)):
            churn_prediction = "Yes" if prediction == 1 or prediction == 1.0 else "No"
        elif isinstance(prediction, (str, np.str_)):
            churn_prediction = "Yes" if str(prediction).lower() in ['yes', '1', 'true'] else "No"
        else:
            churn_prediction = str(prediction)
        
        return {
            "churn_prediction": churn_prediction,
            "churn_probability": churn_probability,
            "customer_data": customer_data.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {str(e)}")

@app.get("/health")
async def health_check():
    """Vérification de la santé de l'API et du modèle"""
    try:
        # Vérifier que tous les objets sont chargés
        model_ok = model is not None and hasattr(model, 'predict')
        encoder_ok = encoder is not None
        scaler_ok = scaler is not None
        metrics_ok = metrics is not None
        
        all_ok = model_ok and encoder_ok and scaler_ok
        
        return {
            "status": "healthy" if all_ok else "unhealthy",
            "model_loaded": model_ok,
            "encoder_loaded": encoder_ok,
            "scaler_loaded": scaler_ok,
            "metrics_loaded": metrics_ok,
            "model_type": str(type(model).__name__) if model is not None else None,
            "message": "API et tous les composants opérationnels" if all_ok else "Certains composants ne sont pas chargés"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de santé: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """
    Endpoint pour récupérer les métriques du modèle
    
    Returns:
        dict: Métriques du modèle (accuracy, precision, recall, f1_score, confusion_matrix, classification_report)
    """
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail="Les métriques ne sont pas disponibles. Vérifiez que le fichier metrics.json existe."
        )
    
    return {
        "metrics": metrics,
        "message": "Métriques chargées depuis models/metrics.json"
    }

