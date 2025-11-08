"""
Script pour sauvegarder correctement le modèle RandomForestClassifier, 
le preprocessing (encoder, scaler) et les métriques.

Ce script doit être exécuté depuis le notebook ou après avoir entraîné le modèle.
"""

import joblib
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def save_model_correctly(model, encoder, scaler, y_test=None, y_pred=None, 
                        save_dir='./models', model_name='model.pkl'):
    """
    Sauvegarde le modèle, le preprocessing et les métriques.
    
    Args:
        model: Le modèle RandomForestClassifier entraîné (variable 'Forest' dans le notebook)
        encoder: L'OneHotEncoder utilisé pour encoder les variables catégorielles
        scaler: Le StandardScaler utilisé pour normaliser les données
        y_test: Les vraies valeurs de test (optionnel, pour calculer les métriques)
        y_pred: Les prédictions sur le test (optionnel, pour calculer les métriques)
        save_dir: Dossier où sauvegarder les fichiers
        model_name: Nom du fichier du modèle
    """
    # Créer le dossier si il n'existe pas
    os.makedirs(save_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    model_path = os.path.join(save_dir, model_name)
    joblib.dump(model, model_path)
    print(f"✅ Modèle sauvegardé dans: {model_path}")
    print(f"   Type du modèle: {type(model).__name__}")
    
    # Sauvegarder l'encoder
    encoder_path = os.path.join(save_dir, 'encoder.pkl')
    joblib.dump(encoder, encoder_path)
    print(f"✅ Encoder sauvegardé dans: {encoder_path}")
    
    # Sauvegarder le scaler
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler sauvegardé dans: {scaler_path}")
    
    # Calculer et sauvegarder les métriques si disponibles
    if y_test is not None and y_pred is not None:
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted')),
            'recall': float(recall_score(y_test, y_pred, average='weighted')),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
        }
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Rapport de classification (sous forme de dictionnaire)
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        # Sauvegarder les métriques en JSON
        metrics_path = os.path.join(save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✅ Métriques sauvegardées dans: {metrics_path}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
    
    # Vérifier que le modèle a bien été sauvegardé
    loaded_model = joblib.load(model_path)
    if hasattr(loaded_model, 'predict'):
        print("✅ Vérification: Le modèle chargé a bien une méthode 'predict'")
    else:
        print("❌ Erreur: Le modèle chargé n'a pas de méthode 'predict'")
    
    print("\n" + "="*60)
    print("✅ Tous les fichiers ont été sauvegardés avec succès!")
    print("="*60)

if __name__ == "__main__":
    print("=" * 60)
    print("Script de sauvegarde du modèle, preprocessing et métriques")
    print("=" * 60)
    print("\n⚠️  Ce script doit être exécuté depuis le notebook après l'entraînement.")
    print("\n   Dans le notebook, utilisez:")
    print("   ```python")
    print("   from save_model import save_model_correctly")
    print("   save_model_correctly(")
    print("       model=Forest,")
    print("       encoder=encoder,")
    print("       scaler=scaler,")
    print("       y_test=y_test,")
    print("       y_pred=Forest_pred,")
    print("       save_dir='./models'")
    print("   )")
    print("   ```")
    print("\n   OU directement dans le notebook:")
    print("   ```python")
    print("   import joblib")
    print("   import json")
    print("   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score")
    print("   ")
    print("   # Sauvegarder le modèle")
    print("   joblib.dump(Forest, './models/model.pkl')")
    print("   ")
    print("   # Sauvegarder le preprocessing")
    print("   joblib.dump(encoder, './models/encoder.pkl')")
    print("   joblib.dump(scaler, './models/scaler.pkl')")
    print("   ")
    print("   # Sauvegarder les métriques")
    print("   metrics = {")
    print("       'accuracy': accuracy_score(y_test, Forest_pred),")
    print("       'precision': precision_score(y_test, Forest_pred, average='weighted'),")
    print("       'recall': recall_score(y_test, Forest_pred, average='weighted'),")
    print("       'f1_score': f1_score(y_test, Forest_pred, average='weighted')")
    print("   }")
    print("   with open('./models/metrics.json', 'w') as f:")
    print("       json.dump(metrics, f, indent=2)")
    print("   ```")
    print("=" * 60)

