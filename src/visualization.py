import plotly.express as px
import plotly.graph_objects as go


def age_distribution(df):

    return px.histogram(
        df,
        x="age",
        nbins=20,
        title="Age Distribution"
    )


def infection_rate_by_group(
    df,
    variable
):

    summary = (
        df.groupby(variable)["infected"]
        .mean()
        .reset_index()
    )

    summary["infection_rate"] = (
        summary["infected"] * 100
    )

    return px.bar(
        summary,
        x=variable,
        y="infection_rate",
        title=f"Infection Rate by {variable}",
        labels={
            "infection_rate":
                "Infection Rate (%)"
        }
    )


def epidemic_curve(df):

    return px.line(
        df,
        x="date",
        y="cases",
        markers=True,
        title="Weekly Epidemic Curve"
    )


def sir_curve(df):

    fig = go.Figure()

    for column in [
        "susceptible",
        "infected",
        "recovered"
    ]:

        fig.add_trace(
            go.Scatter(
                x=df["day"],
                y=df[column],
                mode="lines",
                name=column.title()
            )
        )

    fig.update_layout(
        title="SIR Disease Model",
        xaxis_title="Day",
        yaxis_title="Population"
    )

    return fig
