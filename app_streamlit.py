import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

import dagshub
from dotenv import load_dotenv

load_dotenv()

# Conexión al servidor MLflow
# dagshub.init(
#     repo_owner="reroes799",
#     repo_name="ejemplo02",
#     mlflow=True
# )

mlflow.set_tracking_uri("https://dagshub.com/reroes799/ejemplo02.mlflow")

st.set_page_config(
    page_title="Predicción Bank Marketing",
    layout="centered"
)

st.title("Predicción de suscripción bancaria")
st.write(
    "Esta aplicación carga un Pipeline registrado en MLflow. "
    "Por eso se ingresan las columnas originales del dataset, no las variables encodeadas."
)



@st.cache_resource
def cargar_modelo():
   """
    https://dagshub.com/reroes799/ejemplo02.mlflow/#/models/arboles02/versions/1
   """
   return mlflow.sklearn.load_model("models:/arboles02/1")

model = cargar_modelo()

st.sidebar.header("Configuración")

st.subheader("Datos del cliente")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Edad", min_value=18, max_value=100, value=35)
    job = st.selectbox(
        "Trabajo",
        [
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
            "retired", "self-employed", "services", "student", "technician",
            "unemployed", "unknown"
        ]
    )
    marital = st.selectbox("Estado civil", ["married", "single", "divorced"])
    education = st.selectbox("Educación", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("¿Tiene crédito en mora?", ["no", "yes"])
    balance = st.number_input("Balance", value=1200)

with col2:
    housing = st.selectbox("¿Tiene préstamo de vivienda?", ["yes", "no"])
    loan = st.selectbox("¿Tiene préstamo personal?", ["no", "yes"])
    contact = st.selectbox("Tipo de contacto", ["cellular", "telephone", "unknown"])
    day = st.number_input("Día del mes", min_value=1, max_value=31, value=15)
    month = st.selectbox(
        "Mes",
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    campaign = st.number_input("Número de contactos en campaña", min_value=1, value=2)
    pdays = st.number_input("Días desde contacto anterior", value=-1)
    previous = st.number_input("Contactos previos", min_value=0, value=0)
    poutcome = st.selectbox("Resultado campaña anterior", ["unknown", "failure", "other", "success"])

datos = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}])

st.subheader("Datos enviados al modelo")
st.dataframe(datos)

if st.button("Predecir"):
    prediccion = model.predict(datos)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(datos)[0]
        prob_no = proba[0]
        prob_si = proba[1]
    else:
        prob_no = None
        prob_si = None

    if prediccion == 1:
        st.success("Predicción: el cliente SÍ podría suscribirse.")
    else:
        st.warning("Predicción: el cliente NO se suscribiría.")

    if prob_si is not None:
        st.write(f"Probabilidad de NO suscripción: {prob_no:.4f}")
        st.write(f"Probabilidad de SÍ suscripción: {prob_si:.4f}")

st.caption(
    "Nota: el notebook eliminó la columna 'duration'. Por eso esta aplicación tampoco la solicita."
)
