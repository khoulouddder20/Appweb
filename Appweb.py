import streamlit as st
import asyncio
import nest_asyncio
import numpy as np
from bleak import BleakScanner
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque
import pandas as pd
import time

# Allow nested async calls in Streamlit
nest_asyncio.apply()

# BLE beacons' real coordinates (default)
def get_default_beacons():
    return {
        "BE:AC:F5:26:EC:AA": (0.25 , 0.5),
        "09:8C:D3:F6:55:E8": (3.5, 3.5),
        "EB:2B:9C:7D:C8:0E": (0.25, 6.5),
    }

if "BEACONS" not in st.session_state:
    st.session_state["BEACONS"] = get_default_beacons()

TX_POWER = -59
N = 2.5

class KalmanFilter:
    def __init__(self, process_noise=1e-4, measurement_noise=0.5, estimated_error=1.0, initial_value=0.0):
        self.q = process_noise
        self.r = measurement_noise
        self.p = estimated_error
        self.x = initial_value

    def update(self, measurement):
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x += k * (measurement - self.x)
        self.p *= (1 - k)
        return self.x

# Initialize Kalman filters and history trackers
if "kalman_filters" not in st.session_state:
    st.session_state.kalman_filters = {mac: KalmanFilter(initial_value=-70) for mac in st.session_state["BEACONS"]}
if "distance_history" not in st.session_state:
    st.session_state.distance_history = {mac: deque(maxlen=5) for mac in st.session_state["BEACONS"]}
if "rssi_log" not in st.session_state:
    st.session_state.rssi_log = {mac: deque(maxlen=50) for mac in st.session_state["BEACONS"]}

def rssi_to_distance(rssi, tx_power=TX_POWER, n=N):
    return 10 ** ((tx_power - rssi) / (10 * n))

def trilaterate(p1, r1, p2, r2, p3, r3):
    A = 2 * np.array([
        [p2[0] - p1[0], p2[1] - p1[1]],
        [p3[0] - p1[0], p3[1] - p1[1]]
    ])
    B = np.array([
        r1**2 - r2**2 - p1[0]**2 + p2[0]**2 - p1[1]**2 + p2[1]**2,
        r1**2 - r3**2 - p1[0]**2 + p3[0]**2 - p1[1]**2 + p3[1]**2
    ])
    try:
        pos = np.linalg.lstsq(A, B, rcond=None)[0]
        return pos
    except:
        return None

async def scan_ble():
    devices = await BleakScanner.discover(timeout=2.0)
    distances = {}
    timestamp = datetime.now().strftime("%H:%M:%S")
    for d in devices:
        if d.address in st.session_state["BEACONS"]:
            filtered_rssi = st.session_state.kalman_filters[d.address].update(d.rssi)
            st.session_state.rssi_log[d.address].append((timestamp, filtered_rssi))
            distance = rssi_to_distance(filtered_rssi)
            st.session_state.distance_history[d.address].append(distance)
            avg_distance = sum(st.session_state.distance_history[d.address]) / len(st.session_state.distance_history[d.address])
            if 0.3 < avg_distance < 10:
                distances[d.address] = avg_distance
    return distances

# Async wrapper for compatibility
def run_async_task(task):
    return asyncio.get_event_loop().run_until_complete(task)

# Streamlit UI
st.set_page_config(page_title="BLE Indoor Positioning", layout="centered")
st.title("📍 BLE Indoor Positioning System")

# Guide
with st.expander("📘 Guide - About the Project"):
    st.markdown("""
    **🔧 Description:**  
    This project uses **Bluetooth Low Energy (BLE)** beacons to estimate the position of a device indoors using **trilateration**.

    **🧠 How it Works:**  
    - BLE beacons send signals with a strength called **RSSI**.  
    - A **Kalman Filter** smooths the signal to reduce noise.  
    - The filtered RSSI is converted to distance.  
    - With 3 known distances, **trilateration** estimates the device's 2D position.  
    - The result is plotted on a virtual map.

    **📏 Room Size:** 7m x 4m  
    **📍 Beacons:** 3 fixed in known positions  
    **📡 Scanner:** This app scans BLE and estimates position in real-time.
    """)

# Beacon configuration
# Beacon configuration with customizable room dimensions
with st.expander("🛠️ Configure Beacons"):
    # Input fields for room dimensions
    room_width = st.number_input("Room Width (m)", min_value=1.0, value=7.0, step=0.1)
    room_height = st.number_input("Room Height (m)", min_value=1.0, value=4.0, step=0.1)

    # Store custom room dimensions in session state
    st.session_state.room_width = room_width
    st.session_state.room_height = room_height
    
    # Display beacon configuration sliders based on room dimensions
    for i, mac in enumerate(st.session_state["BEACONS"]):
        # Adjust beacon position sliders based on the custom room dimensions
        x = st.slider(f"Beacon {i+1} X (MAC: {mac})", 0.0, room_width, st.session_state["BEACONS"][mac][0], 0.1)
        y = st.slider(f"Beacon {i+1} Y (MAC: {mac})", 0.0, room_height, st.session_state["BEACONS"][mac][1], 0.1)
        
        # Update beacon coordinates in session state
        st.session_state["BEACONS"][mac] = (x, y)




# Initialize session states
if "position_log" not in st.session_state:
    st.session_state.position_log = []
if "scanning" not in st.session_state:
    st.session_state.scanning = False

# Real-time scanning toggle
realtime = st.checkbox("🔁 Enable Real-Time Scanning")
if realtime:
    interval = st.slider("Scan Interval (seconds)", 2, 10, 5)
    if st.button("Start Scanning"):
        st.session_state.scanning = True

    if st.session_state.scanning:
        distances = run_async_task(scan_ble())
        if len(distances) >= 3:
            keys = list(distances.keys())[:3]
            b = st.session_state["BEACONS"]
            p1, r1 = b[keys[0]], distances[keys[0]]
            p2, r2 = b[keys[1]], distances[keys[1]]
            p3, r3 = b[keys[2]], distances[keys[2]]
            pos = trilaterate(p1, r1, p2, r2, p3, r3)
            if pos is not None:
                st.session_state.position_log.append(pos)
                st.success(f"📍 Estimated Position: x={pos[0]:.2f}m, y={pos[1]:.2f}m")
                
                # Plot the position
                fig, ax = plt.subplots()
                for mac, (x, y) in b.items():
                    ax.plot(x, y, 'bo')
                    ax.text(x + 0.1, y, mac[-5:], fontsize=8)
                ax.plot(pos[0], pos[1], 'ro')
                ax.text(pos[0] + 0.1, pos[1], "📱 Device", fontsize=9)
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 7)
                ax.set_title("📌 Position Map")
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.grid(True)
                st.pyplot(fig)
        st.toast(f"Waiting {interval} seconds...")
        time.sleep(interval)
        st.experimental_rerun()

# BLE Scan Button
if st.button("📡 Scan for BLE Beacons"):
    with st.spinner("Scanning nearby BLE devices..."):
        distances = run_async_task(scan_ble())
        if len(distances) >= 3:
            keys = list(distances.keys())[:3]
            b = st.session_state["BEACONS"]
            p1, r1 = b[keys[0]], distances[keys[0]]
            p2, r2 = b[keys[1]], distances[keys[1]]
            p3, r3 = b[keys[2]], distances[keys[2]]
            pos = trilaterate(p1, r1, p2, r2, p3, r3)
            if pos is not None:
                st.session_state.position_log.append(pos)
                st.success(f"📍 Estimated Position: x={pos[0]:.2f}m, y={pos[1]:.2f}m")
                fig, ax = plt.subplots()
                for mac, (x, y) in b.items():
                    ax.plot(x, y, 'bo')
                    ax.text(x + 0.1, y, mac[-5:], fontsize=8)
                ax.plot(pos[0], pos[1], 'ro')
                ax.text(pos[0] + 0.1, pos[1], "📱 Device", fontsize=9)
                ax.set_xlim(0, 4)
                ax.set_ylim(0, 7)
                ax.set_title("📌 Position Map")
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.error("❌ Trilateration failed. Try scanning again.")
        else:
            st.warning("⚠️ Less than 3 beacons detected.")

# RSSI Graph
if st.button("📉 Show RSSI Graph History"):
    st.subheader("📶 RSSI History per Beacon")
    fig_rssi, ax_rssi = plt.subplots()
    for mac, values in st.session_state.rssi_log.items():
        if values:
            times, rssis = zip(*values)
            ax_rssi.plot(times, rssis, label=mac[-5:])
    ax_rssi.set_title("RSSI History")
    ax_rssi.set_xlabel("Time")
    ax_rssi.set_ylabel("RSSI (dBm)")
    ax_rssi.legend()
    ax_rssi.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig_rssi)

# Trail view
with st.expander("📈 Position Trail"):
    fig2, ax2 = plt.subplots()
    if st.session_state.position_log:
        trail = np.array(st.session_state.position_log)
        ax2.plot(trail[:, 0], trail[:, 1], 'r.-', label="Path")

    for idx, (mac, (x, y)) in enumerate(st.session_state["BEACONS"].items(), start=1):
        ax2.plot(x, y, 'bo')
        ax2.text(x + 0.1, y, f"Beacon {idx}", fontsize=8)  # 👈 affichage du nom sans adresse MAC

    ax2.set_xlim(0, 4)
    ax2.set_ylim(0, 7)
    ax2.set_title("Movement Trail")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.grid(True)
    st.pyplot(fig2)


# Export button
if st.button("💾 Export Position Log"):
    if st.session_state.position_log:
        df = pd.DataFrame(st.session_state.position_log, columns=["X", "Y"])
        st.download_button(
            "📥 Download CSV", 
            df.to_csv(index=False), 
            file_name="position_log.csv", 
            mime="text/csv"
        )
    else:
        st.warning("No position data to export")