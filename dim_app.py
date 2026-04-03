
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title  = "DIM Analytics 2026",
    page_icon   = "🔴",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ── Colores del DIM ───────────────────────────────────────────────────────────
DIM_ROJO  = "#C8102E"
DIM_AZUL  = "#003087"
DIM_GRIS  = "#6B7280"
DIM_CLARO = "#F3F4F6"

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #111827;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #C8102E;
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #111827;
        line-height: 1;
    }
    .metric-label {
        font-size: 11px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-sub {
        font-size: 11px;
        color: #6B7280;
        margin-top: 4px;
    }
    .insight-box {
        background: #fff5f5;
        border-left: 3px solid #C8102E;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        font-size: 13px;
        color: #7f1d1d;
        margin-top: 12px;
    }
    h1, h2, h3 { color: #111827 !important; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f3f4f6;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #C8102E !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Cargar datos ──────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    BASE = "/content/drive/MyDrive/dim_analytics"

    df_partidos  = pd.read_excel(f"{BASE}/data/raw/partidos_dim_2026.xlsx")
    df_jugadores = pd.read_csv(f"{BASE}/data/raw/jugadores_dim_2026.csv")

    # Limpiar edad
    df_jugadores["Age"] = (
        df_jugadores["Age"].astype(str).str.split("-").str[0].astype(int)
    )

    # Filtrar solo liga
    df_liga = df_partidos[df_partidos["Comp"] == "Primera A"].copy()
    df_liga = df_liga.sort_values("Date").reset_index(drop=True)
    df_liga["puntos"]      = df_liga["Result"].map({"W":3,"D":1,"L":0})
    df_liga["puntos_acum"] = df_liga["puntos"].cumsum()
    df_liga["dif_goles"]   = df_liga["GF"] - df_liga["GA"]
    df_liga["es_local"]    = (df_liga["Venue"] == "Home").astype(int)

    # Métricas jugadores
    df_jugadores["G+A"]     = df_jugadores["Gls"] + df_jugadores["Ast"]
    df_jugadores["G+A_90"]  = (df_jugadores["G+A"] / df_jugadores["Min"] * 90).round(2)

    def rango_edad(e):
        if e <= 22:   return "Sub-23"
        elif e <= 27: return "23-27"
        elif e <= 32: return "28-32"
        else:         return "33+"

    df_jugadores["rango_edad"] = df_jugadores["Age"].apply(rango_edad)

    return df_liga, df_jugadores

df_liga, df_jugadores = cargar_datos()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔴 DIM Analytics")
    st.markdown("**Independiente Medellín**")
    st.markdown("Liga BetPlay 2026-I")
    st.divider()

    pagina = st.radio(
        "Navegación",
        ["📊 Resumen General",
         "⚽ Partidos",
         "👤 Jugadores",
         "🤖 Modelo ML"]
    )

    st.divider()
    st.markdown("**Filtros**")

    condicion = st.selectbox(
        "Condición de juego",
        ["Todos", "Local (Home)", "Visitante (Away)"]
    )

    st.divider()
    st.caption("Proyecto de Analítica Deportiva")
    st.caption("Ingeniero Industrial | Data Analyst")

# ── Aplicar filtro de condición ───────────────────────────────────────────────
if condicion == "Local (Home)":
    df_filtrado = df_liga[df_liga["Venue"] == "Home"]
elif condicion == "Visitante (Away)":
    df_filtrado = df_liga[df_liga["Venue"] == "Away"]
else:
    df_filtrado = df_liga.copy()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN GENERAL
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "📊 Resumen General":

    st.title("📊 Resumen General — Liga 2026-I")
    st.markdown("---")

    # KPIs
    total      = len(df_filtrado)
    victorias  = (df_filtrado["Result"] == "W").sum()
    empates    = (df_filtrado["Result"] == "D").sum()
    derrotas   = (df_filtrado["Result"] == "L").sum()
    puntos     = df_filtrado["puntos"].sum()
    rendimiento = round(puntos / (total * 3) * 100, 1) if total > 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Partidos",    total)
    col2.metric("Victorias",   victorias)
    col3.metric("Empates",     empates)
    col4.metric("Derrotas",    derrotas)
    col5.metric("Rendimiento", f"{rendimiento}%")

    st.markdown("---")

    col_izq, col_der = st.columns([2, 1])

    with col_izq:
        st.subheader("Evolución de puntos acumulados")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_liga.index + 1, df_liga["puntos_acum"],
                color=DIM_ROJO, linewidth=2.5,
                marker="o", markersize=6,
                markerfacecolor="white",
                markeredgecolor=DIM_ROJO,
                markeredgewidth=2,
                label="Puntos DIM")
        ax.plot(df_liga.index + 1, (df_liga.index + 1) * 3,
                color=DIM_GRIS, linewidth=1.2,
                linestyle="--", alpha=0.5,
                label="Rendimiento perfecto")

        for i, row in df_liga.iterrows():
            color = {"W":"#dcfce7","D":"#fef9c3","L":"#fee2e2"}[row["Result"]]
            ax.axvspan(i + 0.5, i + 1.5, alpha=0.2, color=color)

        for i, row in df_liga.iterrows():
            ax.text(i + 1, row["puntos_acum"] + 0.3,
                    str(row["Opponent"])[:5],
                    ha="center", fontsize=7, color=DIM_GRIS)

        ax.set_xlabel("Jornada")
        ax.set_ylabel("Puntos acumulados")
        ax.set_facecolor(DIM_CLARO)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_der:
        st.subheader("Resultados")

        fig, ax = plt.subplots(figsize=(4, 4))
        sizes  = [victorias, empates, derrotas]
        labels = [f"V ({victorias})", f"E ({empates})", f"D ({derrotas})"]
        colors = ["#16a34a", "#ca8a04", DIM_ROJO]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(width=0.6)
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Goles por partido
    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Goles por partido")
        fig, ax = plt.subplots(figsize=(8, 4))
        x      = range(len(df_filtrado))
        rivales = df_filtrado["Opponent"].str[:8].tolist()
        ancho   = 0.38
        ax.bar([i - ancho/2 for i in x], df_filtrado["GF"],
               width=ancho, color=DIM_ROJO,
               alpha=0.85, label="Goles a favor")
        ax.bar([i + ancho/2 for i in x], df_filtrado["GA"],
               width=ancho, color=DIM_AZUL,
               alpha=0.75, label="Goles en contra")
        ax.set_xticks(list(x))
        ax.set_xticklabels(rivales, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Goles")
        ax.legend(fontsize=9)
        ax.set_facecolor(DIM_CLARO)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.subheader("Local vs Visitante")
        local  = df_liga[df_liga["Venue"] == "Home"]
        visita = df_liga[df_liga["Venue"] == "Away"]
        fig, ax = plt.subplots(figsize=(7, 4))
        categorias = ["Victorias", "Empates", "Derrotas", "GF", "GA"]
        loc_vals = [
            (local["Result"]=="W").sum(),
            (local["Result"]=="D").sum(),
            (local["Result"]=="L").sum(),
            local["GF"].sum(),
            local["GA"].sum()
        ]
        vis_vals = [
            (visita["Result"]=="W").sum(),
            (visita["Result"]=="D").sum(),
            (visita["Result"]=="L").sum(),
            visita["GF"].sum(),
            visita["GA"].sum()
        ]
        xc = range(len(categorias))
        ax.bar([i - 0.2 for i in xc], loc_vals, width=0.38,
               color=DIM_ROJO, label="Local", alpha=0.85)
        ax.bar([i + 0.2 for i in xc], vis_vals, width=0.38,
               color=DIM_AZUL, label="Visitante", alpha=0.75)
        ax.set_xticks(list(xc))
        ax.set_xticklabels(categorias, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_facecolor(DIM_CLARO)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — PARTIDOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "⚽ Partidos":

    st.title("⚽ Partidos — Liga 2026-I")
    st.markdown("---")

    # Forma reciente
    st.subheader("🔥 Forma reciente — últimos 5 partidos")
    ultimos = df_liga.tail(5)
    cols = st.columns(5)
    for i, (_, row) in enumerate(ultimos.iterrows()):
        emoji  = {"W":"✅","D":"🟡","L":"❌"}[row["Result"]]
        venue  = "🏟️" if row["Venue"] == "Home" else "✈️"
        with cols[i]:
            st.markdown(f"**{emoji} {venue}**")
            st.markdown(f"**{row['Opponent'][:12]}**")
            st.markdown(f"**{int(row['GF'])}-{int(row['GA'])}**")
            st.caption(str(row["Date"])[:10])

    st.markdown("---")

    # Tabla completa
    st.subheader("Historial completo")
    df_tabla = df_filtrado[[
        "Date","Opponent","Venue","Result",
        "GF","GA","Poss","Shots","Shots Taregt",
        "puntos","dif_goles"
    ]].copy()
    df_tabla["Date"] = df_tabla["Date"].astype(str).str[:10]
    df_tabla = df_tabla.sort_values("Date", ascending=False)
    df_tabla.columns = [
        "Fecha","Rival","Condición","Resultado",
        "GF","GA","Posesión%","Disparos","Al Arco",
        "Puntos","Dif. Goles"
    ]

    def color_resultado(val):
        if val == "W": return "background-color: #dcfce7; color: #166534"
        if val == "D": return "background-color: #fef9c3; color: #854d0e"
        if val == "L": return "background-color: #fee2e2; color: #991b1b"
        return ""

    st.dataframe(
        df_tabla.style.applymap(color_resultado, subset=["Resultado"]),
        use_container_width=True, height=450
    )

    # Estadísticas de posesión
    st.markdown("---")
    st.subheader("Posesión según resultado")
    fig, ax = plt.subplots(figsize=(9, 4))
    colores_r = {"W":"#16a34a","D":"#ca8a04","L":DIM_ROJO}
    nombres_r = {"W":"Victoria","D":"Empate","L":"Derrota"}
    for r in ["W","D","L"]:
        datos = df_liga[df_liga["Result"]==r]["Poss"]
        if len(datos) > 0:
            ax.scatter([nombres_r[r]]*len(datos), datos,
                      color=colores_r[r], s=100,
                      edgecolors="white", linewidth=1.5,
                      alpha=0.8, zorder=3)
            ax.plot(nombres_r[r], datos.mean(),
                   marker="D", color=colores_r[r],
                   markersize=12, zorder=4)
    ax.axhline(y=50, color=DIM_GRIS, linestyle="--", alpha=0.5)
    ax.set_ylabel("Posesión (%)")
    ax.set_facecolor(DIM_CLARO)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — JUGADORES
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "👤 Jugadores":

    st.title("👤 Análisis de Jugadores")
    st.markdown("---")

    # Filtro por posición
    posiciones = ["Todas"] + sorted(df_jugadores["Pos"].unique().tolist())
    pos_sel    = st.selectbox("Filtrar por posición", posiciones)

    df_jug = df_jugadores.copy()
    if pos_sel != "Todas":
        df_jug = df_jug[df_jug["Pos"] == pos_sel]

    # Tabla jugadores
    st.subheader("Plantilla y estadísticas")
    df_tabla_jug = df_jug[[
        "Player","Pos","Age","Min",
        "Gls","Ast","G+A","CrdY","CrdR","rating_sofascore"
    ]].copy()
    df_tabla_jug = df_tabla_jug.sort_values("Min", ascending=False)
    df_tabla_jug.columns = [
        "Jugador","Pos","Edad","Minutos",
        "Goles","Asist.","G+A","Amarillas","Rojas","Rating"
    ]
    st.dataframe(df_tabla_jug, use_container_width=True, height=350)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top goleadores y asistidores")
        df_ga = df_jug[df_jug["G+A"] > 0].sort_values("G+A", ascending=True)
        fig, ax = plt.subplots(figsize=(7, max(4, len(df_ga)*0.5)))
        ax.barh(df_ga["Player"], df_ga["Gls"],
                color=DIM_ROJO, label="Goles", edgecolor="white")
        ax.barh(df_ga["Player"], df_ga["Ast"],
                left=df_ga["Gls"],
                color=DIM_AZUL, label="Asistencias", edgecolor="white")
        ax.legend(fontsize=9)
        ax.set_xlabel("Participaciones en gol")
        ax.set_facecolor(DIM_CLARO)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Minutos jugados")
        df_min = df_jug.sort_values("Min", ascending=True)
        mediana = df_jug["Min"].median()
        colores = [DIM_ROJO if m >= mediana else DIM_AZUL
                   for m in df_min["Min"]]
        fig, ax = plt.subplots(figsize=(7, max(4, len(df_min)*0.5)))
        ax.barh(df_min["Player"], df_min["Min"],
                color=colores, edgecolor="white")
        ax.axvline(x=mediana, color=DIM_GRIS,
                  linestyle="--", alpha=0.7,
                  label=f"Mediana: {int(mediana)} min")
        ax.legend(fontsize=9)
        ax.set_xlabel("Minutos")
        ax.set_facecolor(DIM_CLARO)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Análisis de edad
    st.markdown("---")
    st.subheader("Distribución por edad")
    col_a, col_b = st.columns(2)

    with col_a:
        resumen_edad = df_jug.groupby("rango_edad").agg(
            jugadores = ("Player","count"),
            goles     = ("Gls","sum"),
            asist     = ("Ast","sum")
        ).reset_index()
        resumen_edad["G+A"] = resumen_edad["goles"] + resumen_edad["asist"]
        fig, ax = plt.subplots(figsize=(6, 4))
        colores_e = ["#3b82f6","#16a34a",DIM_ROJO,"#6b7280"]
        ax.bar(resumen_edad["rango_edad"],
               resumen_edad["jugadores"],
               color=colores_e[:len(resumen_edad)],
               edgecolor="white", width=0.5)
        ax.set_ylabel("Jugadores")
        ax.set_xlabel("Rango de edad")
        ax.set_facecolor(DIM_CLARO)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        colores_pos = {
            "GK":DIM_GRIS,"DF":DIM_AZUL,
            "MF":"#f59e0b","FW":DIM_ROJO,
            "MF,FW":DIM_ROJO,"DF,MF":DIM_AZUL
        }
        for _, row in df_jug.iterrows():
            color = colores_pos.get(row["Pos"], DIM_GRIS)
            ax.scatter(row["Age"], row["Min"],
                      color=color, s=100,
                      edgecolors="white",
                      linewidth=1.5, alpha=0.85, zorder=3)
            ax.annotate(row["Player"].split()[-1],
                       (row["Age"], row["Min"]),
                       textcoords="offset points",
                       xytext=(5, 3), fontsize=8, color=DIM_GRIS)
        ax.axvline(x=df_jug["Age"].mean(),
                  color=DIM_ROJO, linestyle="--",
                  alpha=0.6,
                  label=f"Edad media: {df_jug['Age'].mean():.1f}")
        ax.set_xlabel("Edad")
        ax.set_ylabel("Minutos")
        ax.set_facecolor(DIM_CLARO)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — MODELO ML
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖 Modelo ML":

    st.title("🤖 Modelo Predictivo — Random Forest")
    st.markdown("---")

    # Entrenar modelo
    features = ["Poss","Shots","Shots Taregt","es_local"]
    X = df_liga[features]
    y = df_liga["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        random_state=42, class_weight="balanced"
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    # Métricas
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy del modelo",  f"{round(acc*100,1)}%")
    col2.metric("Partidos entrenamiento", len(X_train))
    col3.metric("Partidos evaluación",    len(X_test))

    st.markdown("---")
    col_izq, col_der = st.columns(2)

    with col_izq:
        # Importancia de variables
        st.subheader("¿Qué variable decide más?")
        importancias = pd.Series(
            modelo.feature_importances_, index=features
        ).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        colores_imp = [DIM_ROJO if v == importancias.max()
                      else DIM_AZUL for v in importancias]
        bars = ax.barh(importancias.index, importancias.values,
                       color=colores_imp, edgecolor="white")
        for bar, val in zip(bars, importancias.values):
            ax.text(bar.get_width() + 0.005,
                   bar.get_y() + bar.get_height()/2,
                   f"{round(val*100,1)}%",
                   va="center", fontsize=10, fontweight="bold")
        ax.set_facecolor(DIM_CLARO)
        ax.set_xlim(0, importancias.max() + 0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_der:
        # Matriz de confusión
        st.subheader("Matriz de confusión")
        cm     = confusion_matrix(y_test, y_pred, labels=["W","D","L"])
        labels = ["Victoria","Empate","Derrota"]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                   xticklabels=labels, yticklabels=labels,
                   linewidths=0.5, ax=ax,
                   annot_kws={"size":13,"weight":"bold"})
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── SIMULADOR INTERACTIVO ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔮 Simulador de partido")
    st.markdown("Ajusta los valores y el modelo predice el resultado en tiempo real.")

    col1, col2 = st.columns(2)
    with col1:
        posesion  = st.slider("Posesión esperada (%)", 30, 70, 50)
        disparos  = st.slider("Disparos esperados",    4,  20, 11)
    with col2:
        al_arco   = st.slider("Disparos al arco",      1,  10,  5)
        es_local  = st.radio("Condición",
                            ["Local 🏟️", "Visitante ✈️"])
        local_val = 1 if "Local" in es_local else 0

    rival_sim = st.text_input("Rival (opcional)", "Atlético Nacional")

    if st.button("🔮 Predecir resultado"):
        partido_nuevo = pd.DataFrame({
            "Poss"         : [posesion],
            "Shots"        : [disparos],
            "Shots Taregt" : [al_arco],
            "es_local"     : [local_val]
        })

        prediccion    = modelo.predict(partido_nuevo)[0]
        probabilidades = modelo.predict_proba(partido_nuevo)[0]
        clases        = modelo.classes_

        resultado_texto = {
            "W": "✅ VICTORIA",
            "D": "🟡 EMPATE",
            "L": "❌ DERROTA"
        }

        st.markdown("---")
        st.markdown(f"### Resultado predicho: {resultado_texto[prediccion]}")
        st.markdown(f"**Rival:** {rival_sim} | **Condición:** {es_local}")
        st.markdown("**Probabilidades:**")

        for clase, prob in sorted(
            zip(clases, probabilidades),
            key=lambda x: x[1], reverse=True
        ):
            nombre = {"W":"Victoria","D":"Empate","L":"Derrota"}[clase]
            st.progress(float(prob), text=f"{nombre}: {round(prob*100,1)}%")
