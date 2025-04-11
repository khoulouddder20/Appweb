import streamlit as st
import numpy as np
from datetime import datetime
from collections import deque
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
import time

# ✅ SET PAGE CONFIG FIRST
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

if "position_log" not in st.session_state:
    st.session_state["position_log"] = []

if "room_size" not in st.session_state:
    st.session_state["room_size"] = {"width": 7.0, "height": 4.0}

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
distance_history = {mac: deque(maxlen=5) for mac in st.session_state["BEACONS"]}
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

def scan_ble():
    distances = {}
    timestamp = datetime.now().strftime("%H:%M:%S")

    scanned_devices = st.session_state.get("web_ble_devices", {})

    for mac, rssi in scanned_devices.items():
        if mac in st.session_state["BEACONS"]:
            filtered_rssi = kalman_filters[mac].update(rssi)
            rssi_log[mac].append((timestamp, filtered_rssi))
            distance = rssi_to_distance(filtered_rssi)
            distance_history[mac].append(distance)
            avg_distance = sum(distance_history[mac]) / len(distance_history[mac])
            if 0.3 < avg_distance < 10:
                distances[mac] = avg_distance

    return distances
# BLE Scanner section
st.markdown("## 🛰️ BLE Scanner (Simulated)")
st.write("Click the button below to simulate a BLE scan (real BLE scanning requires browser integration).")

if st.button("🔍 Start BLE Scan"):
    # Simulate BLE scan with random RSSI values
    st.session_state["web_ble_devices"] = {
        "BE:AC:F5:26:EC:AA": np.random.randint(-75, -65),
        "09:8C:D3:F6:55:E8": np.random.randint(-78, -70),
        "EB:2B:9C:7D:C8:0E": np.random.randint(-70, -60),
    }
    st.success("✅ Simulated BLE scan completed!")
    st.json(st.session_state["web_ble_devices"])


st.components.v1.html("""
    <script>
    async function scanBLE() {
        try {
            const device = await navigator.bluetooth.requestDevice({
                acceptAllDevices: true,
                optionalServices: ['battery_service']
            });

            const server = await device.gatt.connect();
            const rssi = Math.floor(Math.random() * 20) - 80;
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

# Example placeholder input for testing
st.session_state["web_ble_devices"] = {
    "BE:AC:F5:26:EC:AA": -67,
    "09:8C:D3:F6:55:E8": -72,
    "EB:2B:9C:7D:C8:0E": -65,
}

# Sidebar for room and beacon configuration
st.sidebar.header("Room Configuration")
room_width = st.sidebar.slider("Room Width (meters)", 1.0, 10.0, st.session_state["room_size"]["width"], 0.1)
room_height = st.sidebar.slider("Room Height (meters)", 1.0, 10.0, st.session_state["room_size"]["height"], 0.1)
st.session_state["room_size"] = {"width": room_width, "height": room_height}

st.sidebar.header("Beacon Configuration")
for mac in st.session_state["BEACONS"]:
    x = st.sidebar.slider(f"Beacon {mac[-5:]} X", 0.0, room_width, st.session_state["BEACONS"][mac][0], 0.1)
    y = st.sidebar.slider(f"Beacon {mac[-5:]} Y", 0.0, room_height, st.session_state["BEACONS"][mac][1], 0.1)
    st.session_state["BEACONS"][mac] = (x, y)

# Real-time scanning section
with st.expander("🔁 Real-Time Scanning"):
    realtime = st.checkbox("Enable Real-Time Scanning")
    if realtime:
        interval = st.slider("Scan Interval (seconds)", 2, 10, 5)
        if st.button("Start Scanning"):
            st.session_state["scanning"] = True

        if st.session_state.get("scanning"):
            distances = asyncio.run(scan_ble())
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
            time.sleep(interval)
            st.experimental_rerun()

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

# Position trail display
with st.expander("📈 Position Trail"):
    trail = np.array(st.session_state["position_log"])
    if len(trail) > 0:
        fig, ax = plt.subplots()
        ax.plot(trail[:, 0], trail[:, 1], 'r.-', label="Path")
        ax.set_xlim(0, st.session_state["room_size"]["width"])
        ax.set_ylim(0, st.session_state["room_size"]["height"])
        ax.set_title("📍 Movement Trail")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.grid(True)
        st.pyplot(fig)

# Export button
if st.button("📂 Export Position Log"):
    df = pd.DataFrame(st.session_state["position_log"], columns=["X", "Y"])
    st.download_button("📅 Download CSV", df.to_csv(index=False), file_name="position_log.csv", mime="text/csv")
