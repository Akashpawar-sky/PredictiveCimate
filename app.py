import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CO₂ Emissions Forecasting", page_icon="🌍", layout="wide")

DATA_PATH = "data/climate_data.csv"
FEATURES = ["GDP", "Population", "Energy"]
TARGET = "CO2"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = []
    fitted = {}
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
    return pd.DataFrame(results), fitted

df = load_data()
results, models = train_models(df)

st.title("🌍 Predictive Climate Modeling")
st.subheader("CO₂ Emissions Forecasting using Machine Learning")
st.caption("Simple college-level demonstration using Linear Regression and Random Forest.")

with st.sidebar:
    st.header("Project Settings")
    st.write("Dataset: India demo data")
    st.write(f"Years: {int(df['Year'].min())}–{int(df['Year'].max())}")
    st.write(f"Rows: {len(df)}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Model Comparison", "🔮 Prediction", "📁 Data"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest CO₂", f"{df.iloc[-1]['CO2']:.2f}")
    c2.metric("Latest GDP", f"{df.iloc[-1]['GDP']:.0f}")
    c3.metric("Latest Population", f"{df.iloc[-1]['Population']:.0f}")

    fig, ax = plt.subplots()
    ax.plot(df["Year"], df["CO2"], marker="o")
    ax.set_title("CO₂ Emissions Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.scatter(df["GDP"], df["CO2"])
        ax.set_title("GDP vs CO₂")
        ax.set_xlabel("GDP")
        ax.set_ylabel("CO₂")
        st.pyplot(fig, clear_figure=True)

    with c2:
        fig, ax = plt.subplots()
        ax.scatter(df["Energy"], df["CO2"])
        ax.set_title("Energy vs CO₂")
        ax.set_xlabel("Energy")
        ax.set_ylabel("CO₂")
        st.pyplot(fig, clear_figure=True)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df[["CO2", "GDP", "Population", "Energy"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig, clear_figure=True)

with tab2:
    st.write("The app trains two beginner-friendly regression models.")
    st.dataframe(results.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}))
    st.info("Lower MAE/RMSE means lower prediction error. Higher R² means the model explains more variance in the test data.")

with tab3:
    model_name = st.selectbox("Model", list(models.keys()))
    col1, col2, col3 = st.columns(3)
    with col1:
        gdp = st.number_input("GDP", min_value=0.0, value=float(df["GDP"].iloc[-1] * 1.05))
    with col2:
        population = st.number_input("Population", min_value=0.0, value=float(df["Population"].iloc[-1] * 1.01))
    with col3:
        energy = st.number_input("Energy", min_value=0.0, value=float(df["Energy"].iloc[-1] * 1.03))

    if st.button("Predict CO₂"):
        input_df = pd.DataFrame([{"GDP": gdp, "Population": population, "Energy": energy}])
        prediction = models[model_name].predict(input_df)[0]
        st.success(f"Estimated CO₂: {prediction:.2f}")
        st.warning("This is an educational model output, not a scientific forecast.")

with tab4:
    st.dataframe(df, use_container_width=True)
    st.download_button("Download dataset CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="climate_data.csv", mime="text/csv")
