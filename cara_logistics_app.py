import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import altair as alt

st.set_page_config(page_title="Cara Logistics Optimizer", layout="wide")
st.title("🚚 Cara Orange Growers - Transportation Optimizer")

st.markdown("Edit supply, demand, and costs, then click **Run Optimization** to solve.")

# ------------------------------
# Editable Tables
# ------------------------------
regions = ["Indian River, FL", "Rio Grande Valley, TX", "Central Valley, CA"]
rdcs = ["Atlanta, GA", "Chicago, IL", "Dallas, TX", "Los Angeles, CA"]

def default_supply():
    return pd.DataFrame({"Region": regions, "Supply (tons)": [150, 170, 200]})

def default_demand():
    return pd.DataFrame({"RDC": rdcs, "Demand (tons)": [140, 130, 120, 130]})

def default_costs():
    return pd.DataFrame(
        [
            [500, 700, 800, 1200],
            [400, 600, 300, 1000],
            [900, 850, 650, 400]
        ],
        index=regions,
        columns=rdcs
    )

st.subheader("Supply Capacities")
supply_df = st.data_editor(default_supply(), num_rows="fixed", use_container_width=True)

st.subheader("RDC Demand Requirements")
demand_df = st.data_editor(default_demand(), num_rows="fixed", use_container_width=True)

st.subheader("Transportation Costs (USD per ton)")
costs = st.data_editor(default_costs(), use_container_width=True)

# ------------------------------
# Optimization Trigger
# ------------------------------
if st.button("Run Optimization"):
    supply = dict(zip(supply_df["Region"], supply_df["Supply (tons)"]))
    demand = dict(zip(demand_df["RDC"], demand_df["Demand (tons)"]))

    model = LpProblem("Minimize Transportation Cost", LpMinimize)
    routes = [(s, d) for s in supply for d in demand]
    x = LpVariable.dicts("route", routes, lowBound=0, cat='Continuous')
    model += lpSum([x[(s, d)] * costs.loc[s, d] for (s, d) in routes])

    for s in supply:
        model += lpSum([x[(s, d)] for d in demand]) <= supply[s], f"Supply_{s}"
    for d in demand:
        model += lpSum([x[(s, d)] for s in supply]) >= demand[d], f"Demand_{d}"

    model.solve()

    results = pd.DataFrame(0.0, index=supply.keys(), columns=demand.keys())
    for (s, d) in routes:
        results.loc[s, d] = x[(s, d)].varValue

    st.subheader("Optimal Shipment Plan (Tons)")
    st.dataframe(results.style.format("{:.1f}"))

    st.subheader("Total Weekly Transportation Cost")
    st.metric(label="USD", value=f"${value(model.objective):,.0f}")

    sankey_data = dict(
        type='sankey',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(supply.keys()) + list(demand.keys())
        ),
        link=dict(
            source=[list(supply.keys()).index(s) for (s, d) in routes],
            target=[len(supply) + list(demand.keys()).index(d) for (s, d) in routes],
            value=[x[(s, d)].varValue for (s, d) in routes],
            label=[f"{s} → {d}" for (s, d) in routes],
        )
    )
    fig = go.Figure(data=[sankey_data])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Flow Breakdown by Route")
    flow_df = pd.DataFrame([
        {"From": s, "To": d, "Tons": x[(s, d)].varValue, "Cost per Ton": costs.loc[s, d], "Total Cost": x[(s, d)].varValue * costs.loc[s, d]}
        for (s, d) in routes if x[(s, d)].varValue > 0
    ])
    bar_chart = alt.Chart(flow_df).mark_bar().encode(
        x=alt.X('Tons:Q', title='Shipment Volume (Tons)'),
        y=alt.Y('From:N', title='From Region'),
        color='To:N',
        tooltip=['From', 'To', 'Tons', 'Cost per Ton', 'Total Cost']
    ).properties(width=800, height=300)
    st.altair_chart(bar_chart, use_container_width=True)

    st.subheader("Model Status and Shadow Prices")
    st.write(f"Model Status: **{LpStatus[model.status]}**")

    shadow_prices = {}
    for name, c in model.constraints.items():
        if abs(c.pi) > 0.0001:
            shadow_prices[name] = round(c.pi, 2)

    if shadow_prices:
        st.write("**Binding Constraints & Shadow Prices:**")
        st.json(shadow_prices)
    else:
        st.write("No strongly binding constraints detected.")
