import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white;
        text-align: center; margin: 0.25rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Data ─────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    df["target"] = iris.target
    return df, iris

df, iris = load_data()

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "Regresión Logística":  LogisticRegression(max_iter=500, random_state=42),
    "Árbol de Decisión":    DecisionTreeClassifier(random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(random_state=42),
    "SVM":                  SVC(probability=True, random_state=42),
    "KNN":                  KNeighborsClassifier(),
    "Naive Bayes":          GaussianNB(),
}

METRICS = {
    "Accuracy":  lambda y, yp: accuracy_score(y, yp),
    "Precision": lambda y, yp: precision_score(y, yp, average="weighted", zero_division=0),
    "Recall":    lambda y, yp: recall_score(y, yp, average="weighted", zero_division=0),
    "F1-Score":  lambda y, yp: f1_score(y, yp, average="weighted", zero_division=0),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
                 use_column_width=True)
st.sidebar.title("⚙️ Configuración")

st.sidebar.header("📊 Datos")
test_size = st.sidebar.slider("Tamaño del conjunto de prueba", 0.10, 0.50, 0.20, 0.05)
random_state = st.sidebar.number_input("Semilla aleatoria", 0, 999, 42)
scale_data = st.sidebar.checkbox("Normalizar datos (StandardScaler)", value=True)

st.sidebar.header("🤖 Modelo")
selected_models = st.sidebar.multiselect(
    "Seleccionar modelos",
    list(MODELS.keys()),
    default=["Regresión Logística", "Random Forest", "SVM"],
)

st.sidebar.header("📐 Métricas")
selected_metrics = st.sidebar.multiselect(
    "Seleccionar métricas",
    list(METRICS.keys()),
    default=list(METRICS.keys()),
)

st.sidebar.header("🗺️ Frontera de Decisión")
feat_x = st.sidebar.selectbox("Característica X", iris.feature_names, index=0)
feat_y = st.sidebar.selectbox("Característica Y", iris.feature_names, index=1)
db_model = st.sidebar.selectbox("Modelo para frontera", list(MODELS.keys()), index=0)
mesh_resolution = st.sidebar.slider("Resolución de malla", 100, 500, 200, 50)

st.sidebar.header("📈 Visualización")
show_cv = st.sidebar.checkbox("Mostrar validación cruzada (5-fold)", value=True)
show_feature_imp = st.sidebar.checkbox("Mostrar importancia de características", value=True)
show_pairplot = st.sidebar.checkbox("Mostrar pairplot del dataset", value=False)

# ── Train / Evaluate ──────────────────────────────────────────────────────────
X = df[iris.feature_names].values
y = df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state), stratify=y
)

@st.cache_data
def train_and_evaluate(model_name, _X_train, _X_test, y_train, y_test, scale, cv, seed, n_features=4):
    model = MODELS[model_name]
    if scale:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    else:
        pipe = Pipeline([("clf", model)])

    pipe.fit(_X_train, y_train)
    y_pred = pipe.predict(_X_test)
    y_prob = pipe.predict_proba(_X_test)

    scores = {name: fn(y_test, y_pred) for name, fn in METRICS.items()}

    cv_scores = None
    if cv:
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")

    return pipe, y_pred, y_prob, scores, cv_scores

if not selected_models:
    st.warning("⚠️ Selecciona al menos un modelo en el panel lateral.")
    st.stop()

results = {}
for name in selected_models:
    pipe, y_pred, y_prob, scores, cv_scores = train_and_evaluate(
        name, X_train, X_test, y_train, y_test, scale_data, show_cv, int(random_state), n_features=X_train.shape[1]
    )
    results[name] = {
        "pipe": pipe, "y_pred": y_pred, "y_prob": y_prob,
        "scores": scores, "cv_scores": cv_scores,
    }

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌸 Clasificación Iris — Panel de Análisis")
st.caption(f"Dataset: {len(df)} muestras · {len(iris.feature_names)} características · "
           f"3 clases · Train: {len(X_train)} | Test: {len(X_test)}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Métricas",
    "🗺️ Frontera de Decisión",
    "📉 Curvas ROC",
    "🔢 Matrices de Confusión",
    "🌳 Importancia",
    "🔍 Dataset",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 · Métricas
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Comparativa de Métricas")

    # Metric cards per model
    for model_name, res in results.items():
        st.markdown(f"#### 🤖 {model_name}")
        cols = st.columns(len(selected_metrics))
        for col, metric in zip(cols, selected_metrics):
            val = res["scores"][metric]
            col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val:.3f}</div>
                <div class="metric-label">{metric}</div>
            </div>""", unsafe_allow_html=True)

        if show_cv and res["cv_scores"] is not None:
            cv = res["cv_scores"]
            st.caption(f"CV Accuracy (5-fold): {cv.mean():.3f} ± {cv.std():.3f}")
        st.markdown("---")

    # Grouped bar chart comparison
    if len(selected_models) > 1 and selected_metrics:
        st.subheader("📊 Comparación Visual")
        comp_data = []
        for mname, res in results.items():
            for metric in selected_metrics:
                comp_data.append({"Modelo": mname, "Métrica": metric, "Valor": res["scores"][metric]})
        comp_df = pd.DataFrame(comp_data)
        fig = px.bar(
            comp_df, x="Métrica", y="Valor", color="Modelo",
            barmode="group", range_y=[0, 1.05],
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Comparación de métricas entre modelos",
        )
        fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                          font_color="white", height=420)
        st.plotly_chart(fig, use_container_width=True)

    # CV scores box plot
    if show_cv:
        cv_data = []
        for mname, res in results.items():
            if res["cv_scores"] is not None:
                for s in res["cv_scores"]:
                    cv_data.append({"Modelo": mname, "Accuracy (CV)": s})
        if cv_data:
            st.subheader("🎯 Distribución de Validación Cruzada")
            cv_df = pd.DataFrame(cv_data)
            fig2 = px.box(
                cv_df, x="Modelo", y="Accuracy (CV)",
                color="Modelo", points="all",
                color_discrete_sequence=px.colors.qualitative.Bold,
                title="Validación Cruzada 5-fold",
            )
            fig2.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                               font_color="white", height=380)
            st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 · Decision Boundary
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader(f"🗺️ Frontera de Decisión — {db_model}")
    st.caption(f"Características: **{feat_x}** vs **{feat_y}**")

    ix = list(iris.feature_names).index(feat_x)
    iy = list(iris.feature_names).index(feat_y)

    X2 = X[:, [ix, iy]]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y, test_size=test_size, random_state=int(random_state), stratify=y
    )

    model2 = MODELS[db_model]
    if scale_data:
        pipe2 = Pipeline([("scaler", StandardScaler()), ("clf", model2)])
    else:
        pipe2 = Pipeline([("clf", model2)])
    pipe2.fit(X2_train, y2_train)

    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, mesh_resolution),
        np.linspace(y_min, y_max, mesh_resolution),
    )
    Z = pipe2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    colors_bg = ["#a8c8fa", "#faa8a8", "#a8fac8"]
    colors_pt = ["#1a6cf5", "#f51a1a", "#1af551"]
    class_names = list(iris.target_names)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=plt.cm.RdYlBu, levels=[-0.5, 0.5, 1.5, 2.5])
    ax.contour(xx, yy, Z, colors="white", linewidths=1.2, levels=[0.5, 1.5])

    for cls_idx, (cls_name, color) in enumerate(zip(class_names, colors_pt)):
        mask = y == cls_idx
        ax.scatter(X2[mask, 0], X2[mask, 1], c=color, label=cls_name,
                   edgecolors="white", linewidths=0.6, s=60, zorder=5)

    ax.set_xlabel(feat_x, fontsize=12)
    ax.set_ylabel(feat_y, fontsize=12)
    ax.set_title(f"Frontera de Decisión — {db_model}", fontsize=14, fontweight="bold")
    ax.legend(title="Especie", framealpha=0.8)
    ax.set_facecolor("#1e1e2e")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    st.pyplot(fig, use_container_width=True)
    plt.close()

    acc2 = accuracy_score(y2_test, pipe2.predict(X2_test))
    st.info(f"Accuracy del modelo con solo estas 2 características: **{acc2:.3f}**")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 · ROC Curves
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("📉 Curvas ROC (One-vs-Rest)")

    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig = make_subplots(
        rows=1, cols=len(selected_models),
        subplot_titles=selected_models,
        shared_yaxes=True,
    )

    colors_roc = ["#00b4d8", "#f72585", "#06d6a0", "#ffd60a", "#8338ec", "#ff6d00", "#3a86ff"]

    for col_idx, (mname, res) in enumerate(results.items(), start=1):
        y_prob = res["y_prob"]
        for cls_idx, cls_name in enumerate(iris.target_names):
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_prob[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode="lines",
                           name=f"{cls_name} (AUC={roc_auc:.2f})",
                           line=dict(color=colors_roc[cls_idx], width=2.5),
                           legendgroup=mname,
                           showlegend=(col_idx == 1)),
                row=1, col=col_idx,
            )
        # Diagonal
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                       line=dict(dash="dash", color="gray", width=1),
                       showlegend=False),
            row=1, col=col_idx,
        )
        fig.update_xaxes(title_text="FPR", row=1, col=col_idx)
        fig.update_yaxes(title_text="TPR", row=1, col=1)

    fig.update_layout(
        height=430, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", title_text="Curvas ROC por clase (One-vs-Rest)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # AUC summary table
    st.subheader("Tabla de AUC por clase")
    auc_rows = []
    for mname, res in results.items():
        y_prob = res["y_prob"]
        row = {"Modelo": mname}
        for cls_idx, cls_name in enumerate(iris.target_names):
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], y_prob[:, cls_idx])
            row[cls_name] = round(auc(fpr, tpr), 4)
        row["Macro AUC"] = round(
            roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"), 4
        )
        auc_rows.append(row)
    st.dataframe(pd.DataFrame(auc_rows).set_index("Modelo"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 · Confusion Matrices
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🔢 Matrices de Confusión")
    n_models = len(selected_models)
    cols = st.columns(min(n_models, 3))

    for idx, (mname, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(mname, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        fig.tight_layout()
        cols[idx % 3].pyplot(fig, use_container_width=True)
        plt.close()

    # Classification report
    st.subheader("📋 Reporte de Clasificación")
    report_model = st.selectbox("Ver reporte de:", list(results.keys()))
    report = classification_report(
        y_test, results[report_model]["y_pred"],
        target_names=iris.target_names, output_dict=True,
    )
    st.dataframe(
        pd.DataFrame(report).T.style.background_gradient(cmap="Blues", subset=["f1-score"]),
        use_container_width=True,
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 · Feature Importance
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("🌳 Importancia de Características")
    feat_models = [m for m in selected_models
                   if m in ("Random Forest", "Gradient Boosting", "Árbol de Decisión")]

    if not feat_models and not show_feature_imp:
        st.info("Selecciona Random Forest, Gradient Boosting o Árbol de Decisión para ver importancias.")
    else:
        for mname in (feat_models if feat_models else []):
            pipe = results[mname]["pipe"]
            clf = pipe.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                imp = clf.feature_importances_
                feat_df = pd.DataFrame({
                    "Característica": iris.feature_names,
                    "Importancia": imp,
                }).sort_values("Importancia", ascending=True)
                fig = px.bar(
                    feat_df, x="Importancia", y="Característica",
                    orientation="h", title=f"Importancia — {mname}",
                    color="Importancia", color_continuous_scale="Blues",
                )
                fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                                  font_color="white", height=350)
                st.plotly_chart(fig, use_container_width=True)

        # Coeficients for Logistic Regression
        if "Regresión Logística" in selected_models:
            pipe = results["Regresión Logística"]["pipe"]
            clf = pipe.named_steps["clf"]
            n_features_model = clf.coef_.shape[1]
            feature_cols = iris.feature_names[:n_features_model]
            coef_df = pd.DataFrame(
                clf.coef_,
                columns=feature_cols,
                index=iris.target_names,
            )
            st.subheader("Coeficientes — Regresión Logística")
            fig = px.imshow(
                coef_df, color_continuous_scale="RdBu_r", aspect="auto",
                title="Heatmap de Coeficientes",
            )
            fig.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("📌 Correlación entre Características")
        corr = df[iris.feature_names].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                    linewidths=0.5, square=True)
        ax.set_title("Matriz de Correlación", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 · Dataset Explorer
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("🔍 Explorador del Dataset Iris")

    col1, col2 = st.columns([2, 1])
    with col1:
        species_filter = st.multiselect(
            "Filtrar por especie", iris.target_names.tolist(), default=iris.target_names.tolist()
        )
        filtered_df = df[df["species"].isin(species_filter)]
        st.dataframe(filtered_df.drop(columns="target"), use_container_width=True, height=300)
        st.caption(f"{len(filtered_df)} muestras seleccionadas")

    with col2:
        st.markdown("**Estadísticas Descriptivas**")
        st.dataframe(filtered_df[iris.feature_names].describe().round(2), use_container_width=True)

    # Scatter matrix / pairplot
    if show_pairplot:
        st.subheader("Pairplot Interactivo")
        fig_pair = px.scatter_matrix(
            df, dimensions=iris.feature_names, color="species",
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="Scatter Matrix — Iris Dataset",
        )
        fig_pair.update_traces(diagonal_visible=True, marker=dict(size=3, opacity=0.7))
        fig_pair.update_layout(height=650, plot_bgcolor="#0e1117",
                               paper_bgcolor="#0e1117", font_color="white")
        st.plotly_chart(fig_pair, use_container_width=True)
    else:
        # Simple 2D scatter
        st.subheader("Gráfico de Dispersión")
        cx = st.selectbox("Eje X", iris.feature_names, key="ex")
        cy = st.selectbox("Eje Y", iris.feature_names, index=1, key="ey")
        fig_s = px.scatter(
            df, x=cx, y=cy, color="species",
            color_discrete_sequence=px.colors.qualitative.Bold,
            marginal_x="histogram", marginal_y="box",
            title=f"{cx} vs {cy}",
        )
        fig_s.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                            font_color="white", height=480)
        st.plotly_chart(fig_s, use_container_width=True)

    # Distribution per feature
    st.subheader("Distribución por Característica y Especie")
    feat_dist = st.selectbox("Característica", iris.feature_names, key="fd")
    fig_d = px.violin(
        df, x="species", y=feat_dist, color="species",
        box=True, points="all",
        color_discrete_sequence=px.colors.qualitative.Bold,
        title=f"Distribución de {feat_dist} por especie",
    )
    fig_d.update_layout(plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                        font_color="white", height=400)
    st.plotly_chart(fig_d, use_container_width=True)
