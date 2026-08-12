import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def descriptive_statistics(df):

    return df.describe().T


def independent_t_test(
    df,
    numeric_variable,
    group_variable
):

    groups = df[
        group_variable
    ].dropna().unique()

    if len(groups) != 2:
        raise ValueError(
            "The grouping variable must have two groups."
        )

    group_1 = df.loc[
        df[group_variable] == groups[0],
        numeric_variable
    ]

    group_2 = df.loc[
        df[group_variable] == groups[1],
        numeric_variable
    ]

    statistic, p_value = stats.ttest_ind(
        group_1,
        group_2,
        equal_var=False
    )

    return statistic, p_value


def chi_square_test(
    df,
    variable_a,
    variable_b
):

    table = pd.crosstab(
        df[variable_a],
        df[variable_b]
    )

    statistic, p_value, degrees_freedom, expected = (
        stats.chi2_contingency(table)
    )

    return {
        "chi_square": statistic,
        "p_value": p_value,
        "degrees_freedom": degrees_freedom,
        "table": table
    }


def logistic_regression(
    df,
    outcome,
    predictors
):

    data = df[
        [outcome] + predictors
    ].dropna()

    X = data[predictors].astype(float)

    y = data[outcome].astype(float)

    X = sm.add_constant(X)

    model = sm.Logit(
        y,
        X
    )

    result = model.fit(
        disp=False
    )

    results = pd.DataFrame({
        "coefficient": result.params,
        "odds_ratio": np.exp(
            result.params
        ),
        "p_value": result.pvalues
    })

    return result, results


def sir_model(
    population=10000,
    initial_infected=10,
    beta=0.30,
    gamma=0.10,
    days=160
):

    susceptible = (
        population
        - initial_infected
    )

    infected = initial_infected

    recovered = 0

    rows = []

    for day in range(days):

        rows.append({
            "day": day,
            "susceptible": susceptible,
            "infected": infected,
            "recovered": recovered
        })

        new_infections = (
            beta
            * susceptible
            * infected
            / population
        )

        new_recoveries = (
            gamma
            * infected
        )

        susceptible -= new_infections

        infected += (
            new_infections
            - new_recoveries
        )

        recovered += new_recoveries

    return pd.DataFrame(rows)
