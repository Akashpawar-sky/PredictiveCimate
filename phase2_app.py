import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Climate AI - Phase 2", page_icon="🌍", layout="wide")

TARGET_CANDIDATES = ["CO2", "co2", "co2_emissions", "co2_emissions_tonnes"]
FEATURE_CANDIDATES = {
    "GDP": ["GDP", "gdp", "gdp_per_capita"],
    "Population": ["Population", "population"],
    "Energy": ["Energy", "energy_consumption", "energy_consumption_twh"],
    "Fossil Fuel": ["Fossil_Fuel", "fossil_fuel", "fossil_fuels"],
    "Renewable": ["Renewable", "renewable", "renewables"],
}

def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data
def load_csv(uploaded_bytes):
    return pd.read_csv(io.BytesIO(uploaded_bytes))

def prepare_data(df):
    work = df.copy()
    year_col = find_column(work, ["Year", "year"])
    target_col = find_column(work, TARGET_CANDIDATES)
    if year_col is None or target_col is None:
        raise ValueError("CSV must contain Year and CO2 columns.")

    rename = {year_col: "Year", target_col: "CO2"}
    for standard, candidates in FEATURE_CANDIDATES.items():
        col = find_column(work, candidates)
        if col:
            rename[col] = standard
    work = work.rename(columns=rename)

    required = ["GDP", "Population", "Energy"]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    numeric_cols = ["Year", "CO2", "GDP", "Population", "Energy", "Fossil Fuel", "Renewable"]
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["Year", "CO2", "GDP", "Population", "Energy"])
    work = work.drop_duplicates()
    return work.sort_values("Year").reset_index(drop=True)

def train_models(df, features):
    X = df[features]
    y = df["CO2"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []
    fitted = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred) ** 0.5,
            "R²": r2_score(y_test, pred),
        })
        fitted[name] = model
        predictions[name] = (y_test.reset_index(drop=True), pd.Series(pred))

    return pd.DataFrame(results).sort_values("RMSE"), fitted, predictions

st.title("🌍 Predictive Climate Modeling — Phase 2")
st.subheader("CO₂ Emissions Forecasting with Larger Historical Data")
st.caption("Improved version: data upload, preprocessing, three ML models, evaluation and feature importance.")

with st.sidebar:
    st.header("Data Source")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    st.markdown("Expected columns: **Year, CO2, GDP, Population, Energy**")
    st.markdown("Optional: **Fossil_Fuel, Renewable**")

if uploaded:
    raw = load_csv(uploaded.getvalue())
else:
    raw = pd.read_csv("data/climate_data.csv")

try:
    df = prepare_data(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

optional = [c for c in ["Fossil Fuel", "Renewable"] if c in df.columns]
features = ["GDP", "Population", "Energy"] + optional

results, models, predictions = train_models(df, features)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Years", f"{int(df.Year.min())}–{int(df.Year.max())}")
c3.metric("Features", str(len(features)))
c4.metric("Missing values", str(int(df[features + ["CO2"]].isna().sum().sum())))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", "🤖 Models", "🔮 Prediction", "⭐ Importance", "📁 Data"
])

with tab1:
    st.subheader("CO₂ trend")
    fig, ax = plt.subplots()
    ax.plot(df["Year"], df["CO2"], marker="o", linewidth=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂")
    ax.set_title("Historical CO₂ Emissions")
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    st.subheader("Feature correlations")
    fig, ax = plt.subplots()
    sns.heatmap(df[["CO2"] + features].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig, clear_figure=True)

with tab2:
    st.subheader("Model comparison")
    st.dataframe(results.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}), use_container_width=True)
    selected = st.selectbox("Show actual vs predicted", results["Model"].tolist())
    actual, pred = predictions[selected]
    fig, ax = plt.subplots()
    ax.plot(actual.values, label="Actual")
    ax.plot(pred.values, label="Predicted")
    ax.set_title(f"{selected}: Actual vs Predicted")
    ax.legend()
    st.pyplot(fig, clear_figure=True)

with tab3:
    st.subheader("CO₂ prediction")
    model_name = st.selectbox("Model", list(models.keys()))
    values = {}
    cols = st.columns(len(features))
    for i, feature in enumerate(features):
        with cols[i]:
            values[feature] = st.number_input(
                feature, min_value=0.0, value=float(df[feature].iloc[-1])
            )
    if st.button("Predict CO₂", type="primary"):
        prediction = models[model_name].predict(pd.DataFrame([values]))[0]
        st.success(f"Estimated CO₂: {prediction:,.2f}")

with tab4:
    st.subheader("Feature importance")
    model_name = st.selectbox(
        "Importance model",
        ["Random Forest", "Gradient Boosting"],
    )
    importance = pd.Series(
        models[model_name].feature_importances_, index=features
    ).sort_values(ascending=True)
    fig, ax = plt.subplots()
    importance.plot.barh(ax=ax)
    ax.set_xlabel("Importance")
    ax.set_title(f"{model_name} Feature Importance")
    st.pyplot(fig, clear_figure=True)

with tab5:
    st.subheader("Processed dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "Download processed CSV",
        df.to_csv(index=False).encode("utf-8"),
        "processed_climate_data.csv",
        "text/csv",
    )

st.divider()
st.caption("Phase 2 is designed for educational/project use. Validate the dataset and units before using results in the dissertation.")
