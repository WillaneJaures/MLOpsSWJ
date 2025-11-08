import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Création du fichier CSV initial (optionnel)
data = {
    "age": [25, 30, 35, 40, 45],
    "salaire": [50000, 60000, 70000, 80000, 90000],
    "experience": [2, 5, 8, 10, 12]
}

df = pd.DataFrame(data)
df.to_csv("./app_data.csv", index=False)

# Application Streamlit
st.title("Visualisation des données avec Streamlit")

upload_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if upload_file is not None:
    # Correction : utiliser pd.read_csv() sur le fichier uploadé
    df = pd.read_csv(upload_file)
    st.write("Aperçu des données")
    st.write(df.head())

    x_col = st.selectbox("Choisissez la colonne pour l'axe X", df.columns)
    y_col = st.selectbox("Choisissez la colonne pour l'axe Y", df.columns)

    st.write(f"Graphique de {x_col} vs {y_col}")
    
    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    
    # Correction : utiliser st.pyplot(fig) au lieu de st.pyplot(plt)
    st.pyplot(fig)
    
    # Affichage des statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(df.describe())
    
    # Option pour afficher un histogramme
    if st.checkbox("Afficher un histogramme"):
        col_hist = st.selectbox("Choisissez une colonne pour l'histogramme", df.columns)
        fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x=col_hist, ax=ax_hist)
        ax_hist.set_title(f"Distribution de {col_hist}")
        st.pyplot(fig_hist)
else:
    st.info("Veuillez uploader un fichier CSV pour commencer l'analyse")