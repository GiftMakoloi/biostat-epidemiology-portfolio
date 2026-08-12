import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from scipy import stats


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Biostatistics & Epidemiology Portfolio",
    page_icon="🧬",
    layout="wide"
)


# ---------------------------------------------------------
# DATA GENERATION
# ---------------------------------------------------------

@st.cache_data
def generate_data(n=600, seed=42):

    rng = np.random.default_rng(seed)

    age = rng.integers(18, 81, n)

    sex = rng.choice(
        ["Female", "Male"],
        n
    )

    exposed = rng.binomial(
        1,
        0.35,
        n
    )

    smoker = rng.binomial(
        1,
        0.22,
        n
    )

    comorbidity = rng.binomial(
        1,
        0.18,
        n
    )

    vaccinated = rng.binomial(
        1,
        0.65,
        n
    )

    # Logistic probability model
    log_odds = (
        -1.85
        + 0.95 * exposed
        + 0.025 * (age - 40)
        + 0.45 * smoker
        + 0.70 * comorbidity
        - 0.80 * vaccinated
    )

    probability = (
        1 / (1 + np.exp(-log_odds))
    )

    infected = rng.binomial(
        1,
        probability
    )

    severe_probability = (
        0.04
        + 0.18 * infected
        + 0.12 * comorbidity
        + 0.05 * smoker
        + 0.002 * np.maximum(age - 50, 0)
    )

    severe_probability = np.clip(
        severe_probability,
        0,
        0.95
    )

    severe = (
        rng.binomial(
            1,
            severe_probability
        ) * infected
    )

    dates = pd.date_range(
        "2025-01-01",
        periods=n,
        freq="D"
    )

    return pd.DataFrame({
        "date": dates,
        "age": age,
        "sex": sex,
        "exposed": exposed,
        "smoker": smoker,
        "comorbidity": comorbidity,
        "vaccinated": vaccinated,
        "infected": infected,
        "severe_disease": severe
    })


@st.cache_data
def generate_weekly_cases(seed=42):

    rng = np.random.default_rng(seed)

    weeks = 52

    dates = pd.date_range(
        "2025-01-05",
        periods=weeks,
        freq="W"
    )

    t = np.arange(weeks)

    baseline = (
        35
        + 15 * np.sin(t / 6)
        + 0.30 * t
    )

    wave_one = (
        65 * np.exp(
            -((t - 17) ** 2) / 32
        )
    )

    wave_two = (
        80 * np.exp(
            -((t - 38) ** 2) / 50
        )
    )

    expected = np.maximum(
        baseline + wave_one + wave_two,
        5
    )

    cases = rng.poisson(expected)

    return pd.DataFrame({
        "date": dates,
        "cases": cases
    })


df = generate_data()
weekly = generate_weekly_cases()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("🧬 Portfolio")

page = st.sidebar.radio(
    "Select Analysis",
    [
        "Overview",
        "Epidemiological Analysis",
        "Statistical Analysis",
        "SIR Disease Model",
        "Time-Series Forecasting"
    ]
)


# ---------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------

if page == "Overview":

    st.title(
        "Biostatistics & Epidemiology Admission Portfolio"
    )

    st.write(
        """
        This interactive portfolio demonstrates practical applications
        of biostatistics and epidemiology using Python.
        """
    )

    st.info(
        "The dataset used in this demonstration is synthetic."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Observations",
        len(df)
    )

    col2.metric(
        "Infections",
        int(df["infected"].sum())
    )

    col3.metric(
        "Infection Rate",
        f"{df['infected'].mean() * 100:.1f}%"
    )

    col4.metric(
        "Severe Cases",
        int(df["severe_disease"].sum())
    )

    st.subheader("Skills Demonstrated")

    st.write(
        """
        • Epidemiological data analysis  
        • Descriptive statistics  
        • Hypothesis testing  
        • Logistic regression  
        • Risk-factor analysis  
        • Disease modelling  
        • Time-series forecasting  
        • Interactive visualization
        """
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ---------------------------------------------------------
# EPIDEMIOLOGICAL ANALYSIS
# ---------------------------------------------------------

elif page == "Epidemiological Analysis":

    st.title("Epidemiological Data Analysis")

    st.subheader("Age Distribution")

    fig = px.histogram(
        df,
        x="age",
        nbins=20,
        title="Age Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Infection Rate by Risk Factor")

    factor = st.selectbox(
        "Choose variable",
        [
            "exposed",
            "smoker",
            "comorbidity",
            "vaccinated"
        ]
    )

    summary = (
        df.groupby(factor)["infected"]
        .mean()
        .reset_index()
    )

    summary["infection_rate"] = (
        summary["infected"] * 100
    )

    fig = px.bar(
        summary,
        x=factor,
        y="infection_rate",
        title=f"Infection Rate by {factor}",
        labels={
            "infection_rate": "Infection Rate (%)"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ---------------------------------------------------------
# STATISTICAL ANALYSIS
# ---------------------------------------------------------

elif page == "Statistical Analysis":

    st.title("Statistical Analysis")

    # Chi-square
    st.subheader(
        "Chi-Square Test: Exposure vs Infection"
    )

    contingency = pd.crosstab(
        df["exposed"],
        df["infected"]
    )

    chi_square, p_value, dof, expected = (
        stats.chi2_contingency(
            contingency
        )
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Chi-Square",
        f"{chi_square:.3f}"
    )

    col2.metric(
        "P-Value",
        f"{p_value:.5f}"
    )

    st.dataframe(
        contingency
    )

    # T-test
    st.subheader(
        "Independent T-Test: Age by Infection Status"
    )

    infected_age = df.loc[
        df["infected"] == 1,
        "age"
    ]

    noninfected_age = df.loc[
        df["infected"] == 0,
        "age"
    ]

    t_stat, t_p = stats.ttest_ind(
        infected_age,
        noninfected_age,
        equal_var=False
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "T-Statistic",
        f"{t_stat:.3f}"
    )

    col2.metric(
        "P-Value",
        f"{t_p:.5f}"
    )

    # Logistic regression
    st.subheader(
        "Logistic Regression"
    )

    model_data = df[
        [
            "infected",
            "age",
            "exposed",
            "smoker",
            "comorbidity",
            "vaccinated"
        ]
    ].copy()

    X = model_data[
        [
            "age",
            "exposed",
            "smoker",
            "comorbidity",
            "vaccinated"
        ]
    ]

    y = model_data["infected"]

    X = sm.add_constant(X)

    model = sm.Logit(
        y,
        X
    )

    result = model.fit(
        disp=False
    )

    regression_table = pd.DataFrame({
        "Coefficient": result.params,
        "Odds Ratio": np.exp(result.params),
        "P-Value": result.pvalues
    })

    st.dataframe(
        regression_table.round(4),
        use_container_width=True
    )

    with st.expander(
        "View full regression summary"
    ):
        st.text(
            result.summary().as_text()
        )


# ---------------------------------------------------------
# SIR MODEL
# ---------------------------------------------------------

elif page == "SIR Disease Model":

    st.title(
        "SIR Infectious Disease Model"
    )

    population = st.slider(
        "Population",
        1000,
        50000,
        10000,
        1000
    )

    initial_infected = st.slider(
        "Initial infected",
        1,
        500,
        10
    )

    beta = st.slider(
        "Transmission rate (β)",
        0.05,
        0.80,
        0.30,
        0.01
    )

    gamma = st.slider(
        "Recovery rate (γ)",
        0.02,
        0.40,
        0.10,
        0.01
    )

    days = st.slider(
        "Simulation days",
        30,
        300,
        160,
        10
    )

    susceptible = population - initial_infected
    infected = initial_infected
    recovered = 0

    rows = []

    for day in range(days):

        rows.append({
            "day": day,
            "Susceptible": susceptible,
            "Infected": infected,
            "Recovered": recovered
        })

        new_infections = (
            beta
            * susceptible
            * infected
            / population
        )

        new_recoveries = (
            gamma * infected
        )

        susceptible -= new_infections

        infected += (
            new_infections
            - new_recoveries
        )

        recovered += new_recoveries

    sir = pd.DataFrame(rows)

    fig = go.Figure()

    for column in [
        "Susceptible",
        "Infected",
        "Recovered"
    ]:

        fig.add_trace(
            go.Scatter(
                x=sir["day"],
                y=sir[column],
                mode="lines",
                name=column
            )
        )

    fig.update_layout(
        title="SIR Model",
        xaxis_title="Day",
        yaxis_title="Population"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    r0 = beta / gamma

    st.metric(
        "Basic Reproduction Number (R₀)",
        f"{r0:.2f}"
    )


# ---------------------------------------------------------
# TIME SERIES
# ---------------------------------------------------------

elif page == "Time-Series Forecasting":

    st.title(
        "Epidemiological Time-Series Forecasting"
    )

    fig = px.line(
        weekly,
        x="date",
        y="cases",
        markers=True,
        title="Weekly Cases"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    periods = st.slider(
        "Forecast weeks",
        4,
        20,
        8
    )

    model = sm.tsa.ARIMA(
        weekly["cases"],
        order=(1, 1, 1)
    )

    fitted = model.fit()

    forecast = fitted.forecast(
        steps=periods
    )

    future_dates = pd.date_range(
        start=weekly["date"].iloc[-1]
        + pd.Timedelta(weeks=1),
        periods=periods,
        freq="W"
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "cases": forecast
    })

    combined = pd.concat(
        [
            weekly,
            forecast_df
        ],
        ignore_index=True
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=weekly["date"],
            y=weekly["cases"],
            mode="lines+markers",
            name="Observed"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=forecast,
            mode="lines+markers",
            name="Forecast"
        )
    )

    fig.update_layout(
        title="Observed Cases and Forecast",
        xaxis_title="Date",
        yaxis_title="Cases"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Forecast Values")

    st.dataframe(
        forecast_df.round(2),
        use_container_width=True
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Biostatistics & Epidemiology Admission Portfolio"
)

st.sidebar.caption(
    "Synthetic data • Educational use"
)
