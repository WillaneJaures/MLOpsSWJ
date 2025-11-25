# Application Streamlit - Prédiction de Churn

Application web interactive pour prédire le churn (attrition) des clients en utilisant un modèle RandomForestClassifier via une API FastAPI.

## 📋 Prérequis

### 1. Fichiers du modèle dans `models/`

Assurez-vous d'avoir les fichiers suivants dans le dossier `models/`:

- `model.pkl` - Le modèle RandomForestClassifier entraîné
- `encoder.pkl` - L'OneHotEncoder utilisé pour le preprocessing
- `scaler.pkl` - Le StandardScaler utilisé pour la normalisation
- `metrics.json` - Les métriques du modèle (optionnel mais recommandé)

### 2. API FastAPI doit être démarrée

L'application Streamlit communique avec l'API FastAPI (`endpoint.py`). L'API doit être démarrée avant de lancer l'application Streamlit.

## 🚀 Installation

### 1. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
```

### 2. Activer l'environnement virtuel

**Sur Windows:**
```bash
venv\Scripts\activate
```

**Sur Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

## 🎯 Utilisation

### 1. Démarrer l'API FastAPI

Dans un premier terminal, démarrez l'API:

```bash
cd week1/3-project
uvicorn endpoint:app --reload
```

L'API sera accessible à `http://localhost:8000`

Vous pouvez vérifier que l'API fonctionne en visitant:
- `http://localhost:8000/docs` - Documentation interactive Swagger
- `http://localhost:8000/health` - Vérification de santé

### 2. Lancer l'application Streamlit

Dans un second terminal, lancez l'application Streamlit:

```bash
cd week1/3-project
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

**Note:** L'application Streamlit communique avec l'API via des requêtes HTTP. Si l'API n'est pas démarrée, l'application affichera un message d'erreur avec des instructions.

### Interface de l'application

L'application comprend:

1. **Sidebar (Barre latérale)**
   - **Configuration API**: Permet de modifier l'URL de l'API et de vérifier la connexion
   - **Métriques du modèle**: Affiche les métriques récupérées depuis l'API (Accuracy, Precision, Recall, F1-Score)
   - **Matrice de confusion**: Affiche la matrice de confusion

2. **Formulaire de saisie**
   - **Informations démographiques**: Genre, Senior Citizen, Partenaire, Dépendants
   - **Informations sur le service**: Durée de service, Service téléphonique, Lignes multiples
   - **Services Internet**: Type de service, Sécurité, Sauvegarde, Protection, Support technique, Streaming
   - **Contrat et facturation**: Type de contrat, Facturation sans papier, Méthode de paiement
   - **Charges**: Charges mensuelles et totales

3. **Résultats de prédiction**
   - Indication visuelle du risque de churn (Oui/Non)
   - Probabilités de churn et non-churn
   - Graphiques de visualisation
   - Recommandations basées sur la prédiction

## 📊 Fonctionnalités

- ✅ Interface utilisateur intuitive et moderne
- ✅ Communication avec l'API FastAPI via HTTP
- ✅ Formulaire de saisie complet avec validation
- ✅ Prédiction en temps réel via l'API
- ✅ Affichage des probabilités avec graphiques
- ✅ Recommandations personnalisées
- ✅ Affichage des métriques du modèle depuis l'API
- ✅ Visualisation de la matrice de confusion
- ✅ Configuration de l'URL de l'API dans la sidebar
- ✅ Vérification automatique de la connexion à l'API

## 🔧 Préparation des fichiers du modèle

Si vous n'avez pas encore créé les fichiers du modèle, utilisez le script `save_model.py` depuis votre notebook:

```python
from save_model import save_model_correctly

save_model_correctly(
    model=Forest,
    encoder=encoder,
    scaler=scaler,
    y_test=y_test,
    y_pred=Forest_pred,
    save_dir='./models'
)
```

## 📝 Notes

- L'application charge automatiquement les fichiers du modèle au démarrage
- Les données sont automatiquement préprocessées (encodage + normalisation) avant la prédiction
- Les résultats incluent des recommandations basées sur la prédiction

## 🐛 Dépannage

### Erreur: "Impossible de se connecter à l'API"

**Solution:**
1. Vérifiez que l'API FastAPI est démarrée dans un terminal séparé
2. Vérifiez que l'API est accessible à l'URL configurée (par défaut: `http://localhost:8000`)
3. Utilisez le bouton "Vérifier la connexion" dans la sidebar de l'application Streamlit
4. Vérifiez que l'URL de l'API dans la sidebar est correcte

### Erreur: "Certains composants du modèle ne sont pas chargés"

**Solution:**
1. Vérifiez que tous les fichiers requis existent dans le dossier `models/`:
   - `model.pkl`
   - `encoder.pkl`
   - `scaler.pkl`
2. Vérifiez les logs de l'API pour voir quels fichiers manquent
3. Utilisez `save_model.py` pour créer les fichiers correctement

### Erreur lors de la prédiction

- Vérifiez que l'API est toujours en cours d'exécution
- Vérifiez les logs de l'API pour plus de détails
- Assurez-vous que tous les champs du formulaire sont remplis correctement

## 📚 Structure des fichiers

```
week1/3-project/
├── app.py                 # Application Streamlit
├── endpoint.py           # API FastAPI
├── save_model.py         # Script de sauvegarde du modèle
├── models/
│   ├── model.pkl         # Modèle RandomForestClassifier
│   ├── encoder.pkl       # OneHotEncoder
│   ├── scaler.pkl        # StandardScaler
│   └── metrics.json      # Métriques du modèle
└── requirements.txt      # Dépendances Python
```

