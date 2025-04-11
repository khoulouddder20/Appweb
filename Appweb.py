import streamlit as st
import numpy as np
import asyncio
import time
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

# Set page config
st.set_page_config(page_title="BLE Indoor Positioning", layout="centered")
st.title("📍 BLE Indoor Positioning System")

# BLE beacons' real coordinates (default)
def get_default_beacons():
    return {
        "BE:AC:F5:26:EC:AA": (0.1, 0.1),
        "09:8C:D3:F6:55:E8": (3.5, 3.5),
        "EB:2B:9C:7D:C8:0E": (0.1, 6.9),
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

kalman_filters = {mac: KalmanFilter(initial_value=-70) for mac in st.session_state["BEACONS"]}
rssi_log = {mac: deque(maxlen=50) for mac in st.session_state["BEACONS"]}

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

# Initialize position_log if it doesn't exist yet
if "position_log" not in st.session_state:
    st.session_state["position_log"] = []

# BLE Scanning with JavaScript
st.markdown("## 🛰️ BLE Scanner (Simulated)")
st.write("Click the button below to scan for BLE devices (browser support required).")

# JavaScript to Scan BLE Devices (using Web Bluetooth API)
st.components.v1.html("""
    <script>
    async function scanBLE() {
        try {
            const device = await navigator.bluetooth.requestDevice({
                acceptAllDevices: true,
                optionalServices: ['battery_service']
            });

            const server = await device.gatt.connect();
            const rssi = Math.floor(Math.random() * 20) - 80; // Simulate RSSI
            const mac = device.id;

            const pyMsg = {
                type: "streamlit:setComponentValue",
                value: {
                    "mac": mac,
                    "rssi": rssi
                }
            };
            const streamlitDoc = window.parent.document;
            streamlitDoc.dispatchEvent(new CustomEvent("streamlit:setComponentValue", {detail: pyMsg}));
        } catch (error) {
            alert("BLE Scan failed: " + error);
        }
    }

    const button = document.createElement("button");
    button.innerText = "🔍 Start BLE Scan";
    button.style.fontSize = "18px";
    button.onclick = scanBLE;
    document.body.appendChild(button);
    </script>
""", height=150)

# Initialize simulated data for testing
st.session_state["web_ble_devices"] = {
    "BE:AC:F5:26:EC:AA": -67,
    "09:8C:D3:F6:55:E8": -72,
    "EB:2B:9C:7D:C8:0E": -65,
}

# Real-time BLE Scanning
st.session_state["scanning"] = False

# Start BLE Scan button
if st.button("🔍 Start BLE Scan"):
    st.session_state["scanning"] = True
    st.write("Scanning for BLE devices...")
    
# Simulated scan: using the web_ble_devices state
if st.session_state["scanning"]:
    scanned_devices = st.session_state.get("web_ble_devices", {})
    
    distances = {}
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    for mac, rssi in scanned_devices.items():
        if mac in st.session_state["BEACONS"]:
            filtered_rssi = kalman_filters[mac].update(rssi)
            rssi_log[mac].append((timestamp, filtered_rssi))
            distance = rssi_to_distance(filtered_rssi)
            if 0.3 < distance < 10:
                distances[mac] = distance
    
    if len(distances) >= 3:
        keys = list(distances.keys())[:3]
        b = st.session_state["BEACONS"]
        p1, r1 = b[keys[0]], distances[keys[0]]
        p2, r2 = b[keys[1]], distances[keys[1]]
        p3, r3 = b[keys[2]], distances[keys[2]]
        pos = trilaterate(p1, r1, p2, r2, p3, r3)
        if pos is not None:
            st.session_state["position_log"].append(pos)
            st.success(f"📍 Estimated Position: x={pos[0]:.2f}m, y={pos[1]:.2f}m")
    
    st.session_state["scanning"] = False

# RSSI Graph
if st.button("📉 Show RSSI Graph History"):
    st.subheader("📶 RSSI History per Beacon")
    fig_rssi, ax_rssi = plt.subplots()
    for mac, values in rssi_log.items():
        if values:
            times, rssis = zip(*values)
            ax_rssi.plot(times, rssis, label=mac[-5:])
    ax_rssi.set_title("📉 RSSI History")
    ax_rssi.set_xlabel("Time")
    ax_rssi.set_ylabel("RSSI (dBm)")
    ax_rssi.legend()
    ax_rssi.grid(True)
    st.pyplot(fig_rssi)

# Position Trail (Movement Trail)
with st.expander("📈 Position Trail"):
    trail = np.array(st.session_state["position_log"])
    if len(trail) > 0:
        fig, ax = plt.subplots()
        ax.plot(trail[:, 0], trail[:, 1], 'r.-', label="Path")
        ax.set_xlim(0, 7)
        ax.set_ylim(0, 4)
        ax.set_title("📍 Movement Trail")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True)
        st.pyplot(fig)
