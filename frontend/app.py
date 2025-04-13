import streamlit as st
import pandas as pd
import random

# --- Sample Data ---

# Driver ride history
ride_data = pd.DataFrame({
    "Date": ["2025-04-01", "2025-03-30", "2025-03-28"],
    "Route": ["Downtown → Uptown", "City Center → Airport", "Station → Mall"],
    "Duration (mins)": [25, 40, 15],
    "Incidents": ["None", "Phone usage", "Drowsiness alert"],
    "Rating": [4.8, 3.5, 4.0]
})

# Manager behavior monitoring
drivers = ["John Doe", "Alice Smith", "Ravi Kumar"]
behavior_data = pd.DataFrame({
    "Driver": random.choices(drivers, k=10),
    "Drowsiness Alerts": [random.randint(0, 2) for _ in range(10)],
    "Phone Usage": [random.randint(0, 3) for _ in range(10)],
    "Harsh Brakes": [random.randint(0, 4) for _ in range(10)],
    "Rating": [round(random.uniform(3.0, 5.0), 2) for _ in range(10)],
})

# --- App Config ---
st.set_page_config(
    page_title="Driver Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("Driver Monitoring System")
view = st.sidebar.selectbox("Select View", ["Driver", "Manager"])

# --- Main Layout ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Driver Monitoring Dashboard")

# --- Driver View ---
if view == "Driver":
    st.subheader("Driver Profile: John Doe")

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Current Rating", value="4.4 / 5")
    with col2:
        st.metric(label="Total Rides", value=str(len(ride_data)))

    st.markdown("---")
    st.markdown("#### Previous Ride Summary")
    st.dataframe(ride_data, use_container_width=True)

# --- Manager View ---
elif view == "Manager":
    st.subheader("Driver Behavior Overview")

    st.markdown("#### All Drivers - Behavior Data")
    st.dataframe(behavior_data, use_container_width=True)

    st.markdown("#### Filter by Driver")
    selected_driver = st.selectbox("Select Driver", ["All"] + drivers)

    if selected_driver != "All":
        filtered = behavior_data[behavior_data["Driver"] == selected_driver]
        st.dataframe(filtered, use_container_width=True)


