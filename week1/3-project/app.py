import streamlit as st
import requests
import json

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Churn",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Style CSS minimal
st.markdown("""
    <style>
    .big-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.5rem;
    }
    .churn-yes {
        background-color: #ff6b6b;
        color: white;
    }
    .churn-no {
        background-color: #51cf66;
        color: white;
    }
    /* Masquer complètement la sidebar - toutes les versions */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    .css-1d391kg {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Titre
st.markdown('<div class="big-title">🔮 Prédiction de Churn</div>', unsafe_allow_html=True)

# URL de l'API
API_URL = "http://127.0.0.1:8000"

# Formulaire simple en une seule colonne
st.subheader("📋 Informations Client")

# Informations de base
gender = st.selectbox("Genre", ["Male", "Female"])
senior_citizen = st.selectbox("Senior ?", ["Non", "Oui"])
partner = st.selectbox("En couple ?", ["Yes", "No"])
dependents = st.selectbox("A des personnes à charge ?", ["Yes", "No"])
tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

# Services
st.markdown("---")
st.subheader("📱 Services")
phone_service = st.selectbox("Téléphone", ["Yes", "No"])
internet_service = st.selectbox("Internet", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Sécurité en ligne", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Support technique", ["Yes", "No", "No internet service"])

# Contrat
st.markdown("---")
st.subheader("📄 Contrat")
contract = st.selectbox("Type de contrat", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Paiement",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
monthly_charges = st.number_input("Charges mensuelles ($)", 0.0, 200.0, 50.0)

# Bouton de prédiction
st.markdown("---")
if st.button("🚀 Prédire", type="primary", use_container_width=True):
    # Préparer les données
    customer_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Oui" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": "No" if phone_service == "No" else "Yes",
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No" if internet_service == "No" else "Yes",
        "DeviceProtection": "No" if internet_service == "No" else "Yes",
        "TechSupport": tech_support,
        "StreamingTV": "No" if internet_service == "No" else "Yes",
        "StreamingMovies": "No" if internet_service == "No" else "Yes",
        "Contract": contract,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": monthly_charges * tenure
    }
    
    # Appel API
    try:
        with st.spinner("⏳ Analyse en cours..."):
            response = requests.post(f"{API_URL}/predict", json=customer_data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            prediction = result.get("churn_prediction")
            probability = result.get("churn_probability")
            
            # Afficher le résultat
            st.markdown("---")
            if prediction == "Yes":
                st.markdown(
                    '<div class="result-box churn-yes">'
                    '⚠️ <strong>RISQUE DE CHURN</strong><br>'
                    f'Probabilité: {probability:.1%}'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.error("Le client risque de partir. Actions recommandées :")
                st.write("• Contacter le client rapidement")
                st.write("• Proposer une offre spéciale")
                st.write("• Améliorer la qualité de service")
            else:
                st.markdown(
                    '<div class="result-box churn-no">'
                    '✅ <strong>PAS DE RISQUE</strong><br>'
                    f'Probabilité de rester: {(1-probability):.1%}'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.success("Client satisfait. Continuez le bon travail !")
        else:
            error = response.json().get("detail", "Erreur inconnue")
            st.error(f"❌ Erreur: {error}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Impossible de se connecter à l'API")
        st.info("Démarrez l'API avec: `uvicorn main:app --reload`")
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Modèle de Machine Learning - Prédiction de Churn</p>",
    unsafe_allow_html=True
)