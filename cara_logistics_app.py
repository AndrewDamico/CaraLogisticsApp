import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
import altair as alt
import pydeck as pdk

st.set_page_config(page_title="Cara Logistics Optimizer", layout="wide")
st.title("🚚 Cara Orange Growers - Transportation Optimizer")

st.markdown("Edit supply, demand, and costs, then click **Run Optimization** to solve.")

# ------------------------------
# Editable Tables
# ------------------------------
regions = ["Indian River, FL", "Rio Grande Valley, TX", "Central Valley, CA"]
rdcs = ["Atlanta, GA", "Chicago, IL", "Dallas, TX", "Los Angeles, CA"]

region_coords = {
    "Indian River, FL": [27.6, -80.4],
    "Rio Grande Valley, TX": [26.3, -98.1],
    "Central Valley, CA": [36.6, -119.7],
    "Atlanta, GA": [33.7, -84.4],
    "Chicago, IL": [41.9, -87.6],
    "Dallas, TX": [32.8, -96.8],
    "Los Angeles, CA": [34.0, -118.2]
}

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
        {
            "From": s,
            "To": d,
            "Tons": x[(s, d)].varValue,
            "Cost per Ton": costs.loc[s, d],
            "Total Cost": x[(s, d)].varValue * costs.loc[s, d]
        }
        for (s, d) in routes if x[(s, d)].varValue > 0
    ])
    bar_chart = alt.Chart(flow_df).mark_bar().encode(
        x=alt.X('Tons:Q', title='Shipment Volume (Tons)'),
        y=alt.Y('From:N', title='From Region'),
        color='To:N',
        tooltip=['From', 'To', 'Tons', 'Cost per Ton', 'Total Cost']
    ).properties(width=800, height=300)
    st.altair_chart(bar_chart, use_container_width=True)

    st.subheader("Map View of Transportation Routes")
    map_lines = []
    for (s, d) in routes:
        tons = x[(s, d)].varValue
        if tons > 0:
            s_lat, s_lon = region_coords[s]
            d_lat, d_lon = region_coords[d]
            map_lines.append({
                'start_lat': s_lat,
                'start_lon': s_lon,
                'end_lat': d_lat,
                'end_lon': d_lon,
                'tons': tons,
                'tooltip': f"{s} → {d}: {tons:.1f} tons"
            })

    map_df = pd.DataFrame(map_lines)
    # Scale tons to a nicer line width for visualization
    min_width, max_width = 2, 10
    t_min, t_max = map_df['tons'].min(), map_df['tons'].max()
    if t_max > t_min:
        map_df['line_width'] = map_df['tons'].apply(lambda t: min_width + (max_width - min_width) * (t - t_min) / (t_max - t_min))
    else:
        map_df['line_width'] = min_width
    if not map_df.empty:
        line_layer = pdk.Layer(
            "LineLayer",
            data=map_df,
            get_source_position='[start_lon, start_lat]',
            get_target_position='[end_lon, end_lat]',
            get_width='line_width',
            get_color='[0, 100, 255]',
            pickable=True,
            auto_highlight=True
        )
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position='[start_lon, start_lat]',
            get_radius=20000,
            get_color='[0, 100, 255]',
            pickable=True,
            get_fill_color='[0, 100, 255]'
        )
        tooltip_text = {"html": "<b>{tooltip}</b>", "style": {"backgroundColor": "steelblue", "color": "white"}}
        view_state = pdk.ViewState(latitude=37, longitude=-95, zoom=3.5, pitch=0)
        st.pydeck_chart(pdk.Deck(layers=[line_layer, scatter_layer], initial_view_state=view_state, tooltip=tooltip_text))

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
