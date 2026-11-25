import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

DB = "driver_safety.db"

st.set_page_config(page_title="Driver Safety Dashboard", layout="wide")
st.title("🚗 AI Driver Safety Dashboard")

try:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        "SELECT timestamp,event,drowsiness,distraction,status "
        "FROM events ORDER BY timestamp ASC", conn
    )
    conn.close()
except Exception:
    df = pd.DataFrame()

if df.empty:
    st.info("No events logged yet. Run main.py and use the camera.")
else:
    df["time"] = pd.to_datetime(df["timestamp"], unit="s")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(df))
    c2.metric("Drowsiness events", int((df.event == "drowsiness").sum()))
    c3.metric("Critical events", int((df.event == "critical_drowsiness").sum()))
    c4.metric("Distraction events", int((df.event == "distraction").sum()))

    st.subheader("Risk over time")
    risk_df = df[["time", "drowsiness", "distraction"]].set_index("time")
    st.line_chart(risk_df)

    st.subheader("Event distribution")
    counts = df["event"].value_counts().reset_index()
    counts.columns = ["event", "count"]
    fig = px.bar(counts, x="event", y="count", title="Safety Events")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Event log")
    st.dataframe(
        df[["time", "event", "drowsiness", "distraction", "status"]].sort_values(
            "time", ascending=False
        ),
        use_container_width=True,
    )
