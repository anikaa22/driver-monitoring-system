import streamlit as st

# --- App Config ---
st.set_page_config(
    page_title="Driver Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sqlite3
from datetime import datetime
import hashlib
import pandas as pd
import random

# --- Utility Functions ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

def get_user(email):
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    return cursor.fetchone()

# --- Database Setup ---
conn = sqlite3.connect("driver_monitoring.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password_hash TEXT,
    role TEXT CHECK(role IN ('driver', 'manager'))
)''')
conn.commit()

# --- Login Page UI ---
def login_page():
    st.title("Login - Driver Monitoring System")
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Select Action", menu)

    if choice == "Register":
        st.subheader("Create New Account")
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Role", ["driver", "manager"])

        if st.button("Register"):
            if name and email and password:
                try:
                    hashed = hash_password(password)
                    cursor.execute("INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                                   (name, email, hashed, role))
                    conn.commit()
                    st.success("User registered successfully. Please log in.")
                except sqlite3.IntegrityError:
                    st.error("Email already exists. Please use a different email.")
            else:
                st.warning("Please fill all fields")

    elif choice == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = get_user(email)
            if user and verify_password(password, user[3]):
                st.session_state["user"] = {
                    "id": user[0],
                    "name": user[1],
                    "email": user[2],
                    "role": user[4]
                }
                st.success(f"Welcome {user[1]}! Redirecting to dashboard...")
                st.rerun()
            else:
                st.error("Invalid email or password")

# --- Launch Login Page First ---
if "user" not in st.session_state:
    login_page()
    st.stop()

# --- Load Main App if Logged In ---
user = st.session_state["user"]
st.sidebar.markdown(f"**Logged in as:** {user['name']} ({user['role']})")


# The rest of your frontend code from GitHub can remain below this point.
# Add your detection results, session start/stop, logging, and ratings view here.
# You can use user['role'] to show different dashboards for drivers and managers.
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


