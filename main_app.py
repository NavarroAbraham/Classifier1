import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------
# Configuración de la página
# -------------------------------------------------
st.set_page_config(
    page_title="Iris Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔎 Clasificador interactivo del dataset Iris")
st.caption(
    "Selecciona modelos, métricas y visualizaciones para comparar su desempeño."
)

# -------------------------------------------------
# Cargar datos
# -------------------------------------------------
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="target")
df = pd.concat([X, y], axis=1)

# -------------------------------------------------
# Barra lateral – opciones del usuario
# -------------------------------------------------
st.sidebar.header("⚙️ Configuración")

# 1️⃣ Selección de características
features = st.sidebar.multiselect(
    "Características a usar (máx 2 para frontera)",
    options=list(X.columns),
    default=list(X.columns)[:2],
)

if len(features) != 2:
    st.sidebar.warning("Selecciona **exactamente 2** características para visualizar la frontera.")
    st.stop()

# 2️⃣ Modelos disponibles
model_options = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "SVC": SVC(probability=True),
    "RandomForestClassifier": RandomForestClassifier(),
}
selected_models = st.sidebar.multiselect(
    "Modelos a entrenar",
    options=list(model_options.keys()),
    default=list(model_options.keys()),
)

if not selected_models:
    st.sidebar.warning("Selecciona al menos un modelo.")
    st.stop()

# 3️⃣ Métricas
metric_options = {
    "Exactitud (accuracy)": accuracy_score,
    "Precisión (precision)": precision_score,
    "Recall": recall_score,
    "F1‑score": f1_score,
}
selected_metrics = st.sidebar.multiselect(
    "Métricas a visualizar",
    options=list(metric_options.keys()),
    default=list(metric_options.keys())[:2],
)

# 4️⃣ Hiperparámetros
st.sidebar.subheader("🔧 Hiperparámetros (GridSearch)")
use_grid = st.sidebar.checkbox("Buscar hiperparámetros óptimos", value=False)

# -------------------------------------------------
# Preparar datos
# -------------------------------------------------
X_selected = X[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------
def train_model(name, model):
    if use_grid:
        param_grid = {
            "LogisticRegression": {"C": [0.1, 1, 10]},
            "KNeighborsClassifier": {"n_neighbors": [3, 5, 7]},
            "SVC": {"C": [0.5, 1, 2], "kernel": ["linear", "rbf"]},
            "RandomForestClassifier": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]},
        }
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid.get(name, {}),
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        st.sidebar.info(f"Mejor {name}: {grid.best_params_}")
        return best
    else:
        model.fit(X_train, y_train)
        return model

def plot_decision_boundary(model, X_data, y_data, title):
    x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
    y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    sns.scatterplot(
        x=X_data[:, 0], y=X_data[:, 1], hue=y_data,
        palette="deep", edgecolor="black", s=60, ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    st.pyplot(fig)
    plt.close(fig)

def compute_metrics(y_true, y_pred):
    results = {}
    for metric_name in selected_metrics:
        func = metric_options[metric_name]
        if metric_name in ["Precisión (precision)", "Recall", "F1‑score"]:
            results[metric_name] = func(y_true, y_pred, average="macro")
        else:
            results[metric_name] = func(y_true, y_pred)
    return results

# -------------------------------------------------
# Entrenamiento y visualización
# -------------------------------------------------
st.subheader("📊 Resultados de los modelos")

# Diccionario para guardar modelos ya entrenados y evitar re-entrenar en el bucle de visualización
trained_models_dict = {}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Métricas de desempeño")
    metric_table = []
    for name in selected_models:
        model = train_model(name, model_options[name])
        trained_models_dict[name] = model # Guardamos el modelo
        
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        metric_row = {"Modelo": name, **metrics}
        metric_table.append(metric_row)
    
    df_metrics = pd.DataFrame(metric_table)
    
    # SOLUCIÓN AL ERROR: Aplicar formato solo a columnas numéricas
    numeric_cols = df_metrics.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df_metrics.style.format({col: "{:.3f}" for col in numeric_cols}))

with col2:
    st.markdown("### Matriz de confusión")
    last_name = selected_models[-1]
    last_model = trained_models_dict[last_name]
    y_pred_last = last_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_last)
    
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Verdadero")
    ax.set_title(f"Matriz de confusión – {last_name}")
    st.pyplot(fig)
    plt.close(fig)

st.divider()

st.subheader("🗺️ Fronteras de decisión")
# Usamos columnas para que las gráficas no ocupen toda la pantalla verticalmente
cols = st.columns(len(selected_models))
for i, name in enumerate(selected_models):
    with cols[i]:
        plot_decision_boundary(
            trained_models_dict[name],
            X_test,
            y_test,
            title=f"{name}",
        )
