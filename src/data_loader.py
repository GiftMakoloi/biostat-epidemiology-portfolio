import numpy as np
import pandas as pd


def generate_epidemiological_data(
    n=600,
    seed=42
):
    rng = np.random.default_rng(seed)

    age = rng.integers(
        18,
        81,
        n
    )

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

    severe_disease = (
        rng.binomial(
            1,
            severe_probability
        )
        * infected
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
        "severe_disease": severe_disease
    })
