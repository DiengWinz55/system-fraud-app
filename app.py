import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from PIL import Image
import os

# -------- CONFIGURATION ----------
st.set_page_config(page_title="Bank Fraud Detector By Fireshield", page_icon="💳", layout="wide")

# -------- CHARGEMENT DU CSV ----------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset_fraude_40_60.csv")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du dataset : {e}")
        return pd.DataFrame()

data = load_data()

# -------- CHARGEMENT DU MODELE ----------
@st.cache_resource
def load_model():
    try:
        return joblib.load("model_fraude.pkl")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")
        return None

model = load_model()

if model is not None and not data.empty:
    cat_cols = ["pays", "devise", "type_transaction", "canal"]
    num_cols = ["montant", "age", "anciennete_client", "heure", "ip_suspecte", "historique_fraude_client"]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe.fit(data[cat_cols])

# -------- UTILITAIRES ----------
def abbreviate_number(num):
    if pd.isna(num):
        return "N/A"
    num = float(num)
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}G"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.0f}"

# -------- STYLE ----------
st.markdown("""
    <style>
    .main { background: linear-gradient(to bottom right, #f8fafc, #eef2ff); }
    .stButton>button {
        background-color: #2563EB; color: white;
        border-radius: 12px; padding: 12px 24px;
        font-size: 16px; font-weight: bold;
    }
    .welcome-box {
        background: #e0f2fe; padding: 15px; border-radius: 10px;
        margin-bottom: 20px; border-left: 5px solid #2563EB;
    }
    </style>
""", unsafe_allow_html=True)

# -------- SIDEBAR ----------
# Chargement sécurisé du logo
try:
    if os.path.exists("logo.png"):
        img = Image.open("logo.png")
        img.thumbnail((200, 200))
        st.sidebar.image(img)
except Exception as e:
    st.sidebar.warning("⚠️ Logo non chargé.")

st.sidebar.title("Systeme de detection de fraude bancaire")

st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ À propos :**")
st.sidebar.info("""
💳 **Bank Fraud Detector** utilise un modèle d'IA pour :
- 📊 Visualiser les transactions
- 🚨 Détecter les fraudes bancaires
- 🤖 Prédire les risques en temps réel
""")

# -------- FILTRES --------
pays_filter = st.sidebar.multiselect("🌍 Filtrer par Pays", options=sorted(data.get("pays", pd.Series()).unique()))
type_filter = st.sidebar.multiselect("💳 Filtrer par Type", options=sorted(data.get("type_transaction", pd.Series()).unique()))
canal_filter = st.sidebar.multiselect("📡 Filtrer par Canal", options=sorted(data.get("canal", pd.Series()).unique()))
devise_filter = st.sidebar.multiselect("💱 Filtrer par Devise", options=sorted(data.get("devise", pd.Series()).unique()))
fraude_filter = st.sidebar.multiselect("🚨 Filtrer par Fraude", options=[0,1], format_func=lambda x: "Pas Fraude" if x==0 else "Fraude")
ip_filter = st.sidebar.multiselect("🔍 Filtrer par IP Suspecte", options=[0,1], format_func=lambda x: "Non" if x==0 else "Oui")
heure_filter = st.sidebar.slider("⏰ Filtrer par Heure", 0, 23, (0,23))

df_filtered = data.copy()
if not df_filtered.empty:
    if pays_filter: df_filtered = df_filtered[df_filtered["pays"].isin(pays_filter)]
    if type_filter: df_filtered = df_filtered[df_filtered["type_transaction"].isin(type_filter)]
    if canal_filter: df_filtered = df_filtered[df_filtered["canal"].isin(canal_filter)]
    if devise_filter: df_filtered = df_filtered[df_filtered["devise"].isin(devise_filter)]
    if fraude_filter: df_filtered = df_filtered[df_filtered["fraude"].isin(fraude_filter)]
    if ip_filter: df_filtered = df_filtered[df_filtered["ip_suspecte"].isin(ip_filter)]
    df_filtered = df_filtered[(df_filtered["heure"] >= heure_filter[0]) & (df_filtered["heure"] <= heure_filter[1])]

page = st.sidebar.radio("Navigation", ["📊 Dashboard", "📜 Historique", "🤖 Prédiction", "💬 Chatbot"])

# MESSAGE BIENVENUE
st.markdown("""
<div class="welcome-box">
<h3>👋 Bienvenue sur Bank Fraud Detector by Fireshield</h3>
<p>🔹 Analysez et prédisez les fraudes bancaires grâce à l'IA.</p>
<p><b>ℹ️ Variable <code>fraude</code> :</b> 1 = 🚨 FRAUDE | 0 = ✅ PAS DE FRAUDE</p>
</div>
""", unsafe_allow_html=True)

# =============== DASHBOARD ===============
# =============== DASHBOARD ===============
if page == "📊 Dashboard":
    st.subheader("📊 Tableau de Bord")
    st.info("🔍 Sélectionnez des filtres dans la barre latérale pour actualiser les statistiques.")

    if df_filtered.empty:
        st.warning("⚠️ Aucun résultat avec les filtres sélectionnés.")
    else:
        # ===== KPIs GLOBAUX =====
        total_tx = len(df_filtered)
        montant_total = df_filtered["montant"].sum()
        nb_fraudes = df_filtered["fraude"].sum()
        taux_fraude = (nb_fraudes / total_tx * 100) if total_tx > 0 else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nombre total de transactions", total_tx)
        col2.metric("Montant total", f"{montant_total:,.0f}")
        col3.metric("Nombre de fraudes", nb_fraudes)
        col4.metric("Taux global de fraude", f"{taux_fraude:.2f}%")

        st.markdown("---")

        # ===== KPIs PAR DIMENSION =====
        def kpis_par_dimension(df, dim):
            grouped = df.groupby(dim).agg(
                nb_transactions=("montant", "count"),
                montant_total=("montant", "sum"),
                nb_fraudes=("fraude", "sum")
            ).reset_index()
            grouped["taux_fraude"] = (grouped["nb_fraudes"] / grouped["nb_transactions"]) * 100
            return grouped.sort_values("nb_transactions", ascending=False)

        st.subheader("Transaction par Pays")
        st.dataframe(kpis_par_dimension(df_filtered, "pays"))

        st.subheader("KPI par type de Transaction")
        st.dataframe(kpis_par_dimension(df_filtered, "type_transaction"))

        st.subheader("Transaction par Canal")
        st.dataframe(kpis_par_dimension(df_filtered, "canal"))

        st.markdown("---")

        # ===== ANALYSE PAR HEURE =====
        st.subheader("Analyse temporelle par Heure")
        df_hour = df_filtered.groupby("heure").agg(
            nb_transactions=("montant", "count"),
            taux_fraude=("fraude", "mean")
        ).reset_index()
        df_hour["taux_fraude"] *= 100

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(df_hour, x="heure", y="nb_transactions",
                                   title="Nombre de transactions par heure"), use_container_width=True)
        with col2:
            st.plotly_chart(px.line(df_hour, x="heure", y="taux_fraude",
                                    title="Taux de fraude (%) par heure"), use_container_width=True)

        st.markdown("---")

        # ===== MATRICE DE CONFUSION =====
        if model is not None:
            try:
                y_true = df_filtered["fraude"]
                X_num = df_filtered[num_cols].values
                X_cat = ohe.transform(df_filtered[cat_cols])
                X_test = np.hstack([X_num, X_cat])
                y_pred = model.predict(X_test)

                cm = confusion_matrix(y_true, y_pred)
                fig_cm = ff.create_annotated_heatmap(
                    z=cm, x=["Pas Fraude", "Fraude"], y=["Pas Fraude", "Fraude"], colorscale="Blues"
                )
                fig_cm.update_layout(title_text="🔍 Matrice de Confusion")
                st.plotly_chart(fig_cm, use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ Erreur lors du calcul de la matrice de confusion : {e}")

        # ===== CARTE DU TAUX DE FRAUDE =====
        try:
            df_map = df_filtered.groupby("pays").agg(taux_fraude=("fraude", "mean")).reset_index()
            df_map["taux_fraude"] *= 100
            fig_map = px.choropleth(
                df_map, locations="pays", locationmode="country names",
                color="taux_fraude", color_continuous_scale="Reds",
                title="🌍 Taux de Fraude par Pays"
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except:
            st.warning("⚠️ Impossible d'afficher la carte (problème avec les noms de pays).")


# =============== HISTORIQUE ===============
elif page == "📜 Historique":
    st.subheader("📜 Historique des Transactions")
    if df_filtered.empty:
        st.warning("⚠️ Aucun résultat avec les filtres sélectionnés.")
    else:
        st.dataframe(df_filtered.head(50))

# =============== PREDICTION ===============
elif page == "🤖 Prédiction":
    st.subheader("🤖 Prédiction de Fraude")

    with st.form("form_prediction"):
        col1, col2 = st.columns(2)
        with col1:
            montant = st.number_input("Montant", min_value=0.0, step=10.0)
            age = st.number_input("Âge du client", min_value=18, max_value=100, step=1)
            anciennete_client = st.number_input("Ancienneté (années)", min_value=0, step=1)
            heure = st.slider("Heure", 0, 23, 12)
        with col2:
            pays = st.selectbox("Pays", options=sorted(data["pays"].unique()))
            devise = st.selectbox("Devise", options=sorted(data["devise"].unique()))
            type_transaction = st.selectbox("Type", options=sorted(data["type_transaction"].unique()))
            canal = st.selectbox("Canal", options=sorted(data["canal"].unique()))
        
        ip_suspecte = st.radio("IP Suspecte ?", options=[0,1], format_func=lambda x: "Oui" if x==1 else "Non")
        historique_fraude_client = st.radio("Historique Fraude ?", options=[0,1], format_func=lambda x: "Oui" if x==1 else "Non")

        submitted = st.form_submit_button("🔍 Prédire")

    if submitted and model is not None:
        X_num = np.array([[montant, age, anciennete_client, heure, ip_suspecte, historique_fraude_client]])
        X_cat = ohe.transform([[pays, devise, type_transaction, canal]])
        X_input = np.hstack([X_num, X_cat])
        prediction = model.predict(X_input)[0]
        proba = model.predict_proba(X_input)[0][1]

        if prediction == 1:
            st.error(f"🚨 Risque de FRAUDE détecté ({proba*100:.2f}% de probabilité)")
        else:
            st.success(f"✅ Transaction NON FRAUDULEUSE ({proba*100:.2f}% de probabilité)")

# =============== CHATBOT ===============
elif page == "💬 Chatbot":
    st.subheader("💬 Chatbot")
    st.info("🔧 Fonctionnalité en développement.")
