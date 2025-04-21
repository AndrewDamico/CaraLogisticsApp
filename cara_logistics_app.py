
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

st.set_page_config(page_title="Cara Logistics Optimizer", layout="wide")
st.title("🚚 Cara Orange Growers - Transportation Optimizer")

st.markdown("Edit supply, demand, and costs, then click **Run Optimization** to solve.")

# ------------------------------
# Editable Inputs
# ------------------------------
regions = ["Indian River, FL", "Rio Grande Valley, TX", "Central Valley, CA"]
rdcs = ["Atlanta, GA", "Chicago, IL", "Dallas, TX", "Los Angeles, CA"]

supply = {r: st.number_input(f"Supply: {r}", min_value=0, value=150 if r == regions[0] else 170 if r == regions[1] else 200) for r in regions}
demand = {d: st.number_input(f"Demand: {d}", min_value=0, value=140 if d == rdcs[0] else 130) for d in rdcs}

st.subheader("Transportation Costs (USD per ton)")
costs_data = []
for r in regions:
    row = []
    for d in rdcs:
        default = 500 if (r, d) == (regions[0], rdcs[0]) else 700 if r == regions[0] else 400 if r == regions[1] else 900
        row.append(st.number_input(f"{r} → {d}", min_value=0, value=default))
    costs_data.append(row)

costs = pd.DataFrame(costs_data, index=regions, columns=rdcs)

# ------------------------------
# Optimization Trigger
# ------------------------------
if st.button("Run Optimization"):
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
