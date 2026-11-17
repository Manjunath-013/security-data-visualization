# Real-Time Cybersecurity Data Visualization and Intelligent Threat Analytics Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scapy](https://img.shields.io/badge/Scapy-2.8%2B-008080.svg)](https://scapy.net/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: Academic Evaluation](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, integrated SOC threat analytics and network surveillance platform. This system fuses historical cyber incident intelligence with an attribute-grounded threat risk engine, statistical anomaly detection, attacker attribution graphs, an offline machine-learning URL phishing verification pipeline, and **real-time network traffic visualization** (PCAP replay, simulated attack scenarios, and live packet capture).

---

## 1. Project Introduction
The **Real-Time Cybersecurity Data Visualization and Intelligent Threat Analytics Platform** is an interactive, browser-based cybersecurity dashboard engineered to transform raw, complex security data into clear, visual, and actionable insights. Designed for academic evaluation, college-level demonstrations, and SOC operational workflows, it addresses both macro-level historical trends and micro-level live network anomalies.

---

## 2. Problem Statement
Security analysts face a persistent disconnect between historical intelligence databases and real-time network telemetry:
- **Historical incident databases** lack actionable risk severity scoring, statistical anomaly detection, and cross-border attacker flow visualization.
- **Phishing detection systems** operate in isolation without feature-level explainability or safe offline handling guarantees.
- **Network monitoring tools** often produce overwhelming, unreadable log dumps rather than intuitive visual topologies and traffic timelines.
- Organizations need a **unified security data visualization platform** that bridges macroscopic cyber incident patterns with microscopic real-time network traffic surveillance.

---

## 3. Objectives
1. **Interactive Security Data Visualization:** Transform raw cyber incidents, URL parameters, and packet headers into intuitive, interactive visual charts.
2. **Deterministic Threat Risk Scoring:** Compute a 0–100 Analytical Risk Score grounded in verified incident attributes (disruption, data theft, hijacking, zero-day exploitation, critical infrastructure targeting).
3. **Statistical Anomaly Detection:** Implement historical time-series baseline analysis (Rolling mean, IQR, Z-score) to detect anomalous surges in volume, DDoS waves, ransomware spikes, and sector concentrations.
4. **Attacker Attribution Intelligence & Flow Graphs:** Map state vs. non-state threat actors and visualize multi-stage flows (*Attacker Origin &rarr; Attack Vector &rarr; Receiver Country &rarr; Targeted Critical Sector*).
5. **Offline Machine-Learning Phishing Detection:** Extract 21 lexical, structural, and character-entropy features from user-submitted URLs to predict phishing risk without visiting or executing untrusted links.
6. **Real-Time Network Traffic Surveillance:** Ingest network events from PCAP files, deterministic traffic simulations, or live packet capture, updating visual timelines, protocol donut charts, host talker rankings, communication graphs, and geographic endpoint maps.
7. **Real-Time Suspicious Activity Analytics:** Automatically flag Port Scans, SYN Floods, Traffic Bursts, Horizontal Sweeps, DNS Tunneling, and Outbound Exfiltration with deduplication and cooldown logic.

---

## 4. Existing Features (Preserved & Enhanced)
The platform preserves all existing capabilities across 11 core analytical views:
1. **Command Center:** Executive SOC KPIs, threat mix donuts, analytical risk distribution, and deterministic security alerts.
2. **Threat Intelligence:** Multi-year attack trajectories, system disruption intensity, and 3-year statistical trend forecast (OLS with 80% CI).
3. **Attack Analytics:** Monthly distributions, Year &times; Month density heatmaps, and target sector exposure across DDoS, ransomware, and malware.
4. **Geographic Intelligence:** Interactive global choropleth map, receiver country rankings, and cross-border attacker &rarr; target flows.
5. **Attacker Intelligence:** Attributed origin states, State vs. Non-State classifications, and leading attribution intelligence authorities.
6. **Risk & Anomalies:** Risk tier distribution, dossier cards for critical incidents with itemized factors, and statistical anomaly detection results ($Z \ge +1.75\sigma$).
7. **Phishing Detector:** Live offline URL scanner, risk score gauge (0–100), risk tier badge, evidence breakdown, 21-D feature inspection, and actionable recommendations.
8. **Network / Relationship Graph:** Multi-stage Sankey diagram mapping *Initiator State &rarr; Attack Vector &rarr; Target Country &rarr; Target Sector*.
9. **Attack Lifecycle (MITRE):** MITRE TA0001 (Initial Access, highlighting phishing) and TA0040 (Impact) rankings and lifecycle co-occurrence matrix.
10. **Incident Explorer:** Multi-attribute text search, severity tier filters, risk score sorting, detailed dossier inspector, and CSV export.
11. **Model Performance & Methodology:** Empirical ML evaluation tables, confusion matrix, feature importance charts, and academic methodology documentation.

---

## 5. New Real-Time Visualization Features (Module 12)
The platform introduces a dedicated **Real-Time Security Data** module featuring:
- **Multi-Source Ingestion:** Stream events from a Deterministic Traffic Simulator, a PCAP File Replayer, or Live Network Sniffing.
- **Clear Data Source Labelling:** Active stream badge explicitly distinguishing simulated, replayed, and live events.
- **Traffic Overview KPIs:** Packets Ingested, Bytes Ingested, Packets / Sec (PPS), Bandwidth (Kbps), Active Source/Destination IPs, Active Connections, and Triggered Alerts.
- **Live Throughput Timeline:** Rolling 60-second dual-axis line chart for PPS and bandwidth.
- **Protocol Breakdown:** Real-time donut chart visualizing HTTPS, HTTP, DNS, TCP, UDP, ICMP, SMB, and SSH shares.
- **Top IP Talkers:** Horizontal ranking of active Source and Destination IPs, distinguishing Internal LAN (RFC 1918) from Public endpoints.
- **Interactive Network Communication Graph:** 2D circular node-link graph sized by packet volume and color-coded by role (Internal Host, External Server, Suspicious Scanner).
- **Geographic Endpoint Map:** Natural Earth projection showing approximate locations of public endpoints.
- **Real-Time Security Alert Center:** Live SOC feed for detected anomalies with deduplication, cooldown, review toggles, and CSV export.
- **Live Packet Stream Inspector:** Filterable table displaying the latest 100 raw packet headers.

---

## 6. System Architecture

```
                 +-------------------------------------------------------+
                 |    REAL-TIME SECURITY DATA VISUALIZATION PLATFORM     |
                 |                       (app.py)                        |
                 +---------------------------+---------------------------+
                                             |
                     +-----------------------+-----------------------+
                     |                                               |
                     v                                               v
     +-------------------------------+               +-------------------------------+
     |   HISTORICAL CYBER ATTACK     |               |     REAL-TIME NETWORK DATA    |
     |         INTELLIGENCE          |               |          SURVEILLANCE         |
     +---------------+---------------+               +---------------+---------------+
                     |                                               |
     +---------------+---------------+               +---------------+---------------+
     | - EuRepoC v1.3 Preprocessing  |               | - Packet Collector (Thread)   |
     | - Threat Risk Scoring Engine  |               | - Traffic Simulator (6 Scenarios)|
     | - Statistical Anomaly Engine  |               | - PCAP Replayer (Scapy)       |
     | - Attacker Attribution Matrix |               | - Live Network Sniffer (Scapy)|
     | - MITRE Lifecycle Mapping     |               | - Event Normalizer & Geo IP   |
     | - Phishing URL Detector (ML)  |               | - Anomaly Detector (6 Rules)  |
     | - Trend Forecasting (OLS)     |               | - Alert Manager & Dedup       |
     +---------------+---------------+               +---------------+---------------+
                     |                                               |
                     +-----------------------+-----------------------+
                                             |
                                             v
                           +-----------------------------------+
                           |       UNIFIED SOC DASHBOARD       |
                           |   INTERACTIVE VISUAL TOPOLOGIES   |
                           |    ENTERPRISE EXPORT PIPELINES    |
                           +-----------------------------------+
```

---

## 7. Technology Stack
- **Frontend & Interface:** [Streamlit](https://streamlit.io/) with customized dark-mode SOC CSS styling.
- **Packet Capture & PCAP Parsing:** [Scapy](https://scapy.net/).
- **Data Manipulation & Analytics:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/).
- **Interactive Visualizations:** [Plotly Express & Plotly Graph Objects](https://plotly.com/python/) (Dual-axis timelines, Sankey graphs, Node-link graphs, Choropleths, Gauges, Heatmaps).
- **Machine Learning & Modeling:** [Scikit-Learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/), [SciPy](https://scipy.org/).
- **Testing:** Python `unittest` framework (27 automated tests).

---

## 8. Dataset Descriptions & Data Streams
The platform strictly separates its four data streams:

| Stream Type | Source File / Module | Description | Usage Rule |
| :--- | :--- | :--- | :--- |
| **Historical Dataset** | `data/incidents.csv` | EuRepoC Global Dataset v1.3 (3,414 incidents, 2000–2024). | Never labeled as live monitoring. |
| **Phishing Benchmark** | `data/phishing/phishing_dataset.csv` | 6,000 balanced URL records (3,000 Benign, 3,000 Phishing). | Machine learning model training. |
| **PCAP Replay** | `data/realtime/sample_traffic.pcap` | Bundled PCAP containing normal browsing, port scans, SYN floods, and DNS queries. | Replayed at controlled speed. |
| **Simulated Traffic** | `src/realtime/traffic_simulator.py` | In-memory packet generator simulating 6 realistic network scenarios. | Isolated in memory; labeled as simulation. |
| **Live Capture** | `scapy.sniff` | Direct capture from network interface metadata. | Requires Npcap & admin privileges. |

---

## 9. Real-Time Monitoring Instructions
1. Open the navigation menu and select **"Real-Time Security Data"**.
2. Under **Ingestion Source**, choose **"Live Network Interface Capture"**.
3. (Optional) Enter the specific interface name (e.g., `Ethernet` or `Wi-Fi`), or leave empty for the system default.
4. Click **"START INGESTION"**.
5. *Note:* On Windows systems, live packet capture requires the installation of Npcap (or WinPcap) with "WinPcap API-compatible mode" enabled and Administrator permissions. If Npcap is missing, the dashboard will display a clear notification directing you to use the PCAP Replay or Traffic Simulator options.

---

## 10. PCAP Replay Instructions
1. Navigate to **"Real-Time Security Data"**.
2. Select **"PCAP File Replay"** under Ingestion Mode.
3. Choose either:
   - **Bundled Sample PCAP (`sample_traffic.pcap`):** An out-of-the-box capture file pre-loaded with HTTP/HTTPS browsing, a 20-port vertical port scan from `45.154.255.88`, a SYN flood against `192.168.1.10`, and DNS tunneling queries to `9.9.9.9`.
   - **Upload Custom `.pcap` / `.pcapng`:** Upload any standard Wireshark or tcpdump capture file.
4. Adjust the **Replay Pace** slider (e.g., 0.5 seconds per batch).
5. Click **"START INGESTION"** to watch the timeline, communication graph, protocol donut, and alert feed update as packets stream in.
6. Use **"PAUSE / RESUME"** or **"STOP STREAM"** to control execution.

---

## 11. Simulation Instructions
1. Navigate to **"Real-Time Security Data"**.
2. Select **"Simulated Traffic Generator"**.
3. Choose from 6 controlled traffic scenarios:
   - **Normal Enterprise Traffic Mix:** Routine HTTP/HTTPS browsing, DNS lookups, internal SMB, and SSH management.
   - **Port Scan Attack (Vertical Recon):** Rapid TCP SYN probes across 22 distinct ports from scanner `45.154.255.88`.
   - **SYN Flood DDoS Attack:** High-rate TCP SYN flood barrage targeted at `192.168.1.10:443`.
   - **DNS Tunneling & C2 Exfiltration:** High-entropy, encoded subdomains queried to `9.9.9.9`.
   - **Horizontal Subnet Sweep (SMB/SSH):** Single internal host probing port 445 across multiple subnet IPs.
   - **Outbound High-Volume Transfer Spike:** Large payload transfer (25KB+) from internal database to external host `194.26.29.112`.
4. Click **"START INGESTION"** or **"STEP +25 PACKETS"** to observe immediate visual and alert responses.

---

## 12. Suspicious Activity Analytics & Alert Rules
The real-time anomaly engine (`src/realtime/anomaly_detector.py`) evaluates sliding traffic windows:
1. **Vertical Port Scan:** Triggered when a single source IP contacts $\ge 8$ distinct destination ports within 60s (Critical if $\ge 18$ ports).
2. **Horizontal Subnet Scan:** Triggered when a single host contacts $\ge 6$ separate destination IPs on the same port.
3. **SYN Flood Burst:** Triggered when TCP traffic contains $\ge 12$ SYN packets with a SYN ratio $\ge 65\%$ without ACK completion.
4. **DNS Tunneling / Anomaly:** Triggered when a DNS query displays character entropy $\ge 4.0$ or length $\ge 32$ characters.
5. **Outbound Data Spike:** Triggered when outbound transfer from an internal host to a single external IP exceeds 25 KB within a short window.
6. **Traffic Rate Burst:** Triggered when aggregate packet arrival rate exceeds $35$ packets/sec.

*Deduplication & Cooldown:* Alerts are throttled by a 15-second cooldown per `(category, src_ip)` pair to prevent notification flooding.

---

## 13. Testing Instructions
Run the comprehensive 27-test automated test suite:
```bash
# Run all unit tests across platform
python -m unittest discover tests

# Or run specific modules:
python -m unittest tests/test_platform.py   # Phishing, Risk Engine, Anomaly Detection
python -m unittest tests/test_realtime.py   # PCAP, Simulator, Alert Manager, Geo IP
```
*Expected Result:*
```
Ran 27 tests in 1.654s
OK
```

---

## 14. Installation
```bash
# 1. Clone repository
cd security_data_visualization

# 2. Set up virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
python -m pip install -r requirements.txt
```

---

## 15. Execution Instructions
Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 16. Academic & Demonstration Workflow
For evaluators and college project demonstrations, follow this structured demo sequence:

- **STEP 1 — Command Center Overview:** Show executive KPIs, threat category breakdown, risk severity distribution, and deterministic SOC alerts.
- **STEP 2 — Incident Intelligence:** Explore multi-year trends, system disruption analysis, and the 3-year statistical trend forecast.
- **STEP 3 — Attacker & MITRE Intelligence:** Demonstrate the multi-stage Sankey diagram (*Attacker Origin &rarr; Attack Vector &rarr; Receiver Country &rarr; Target Sector*) and MITRE initial access alignment (highlighting phishing).
- **STEP 4 — Phishing Website Detector:** Enter a suspicious URL (e.g., `http://192.168.1.100/paypal-security-update/login.php?token=9281928`). Show the risk gauge (0–100), classification badge, itemized threat factors, benign signals, and safety recommendations.
- **STEP 5 — Real-Time Surveillance (PCAP Replay):** Open "Real-Time Security Data". Select "Bundled Sample PCAP". Click "START INGESTION". Observe the rolling 60s throughput timeline, protocol donut, communication graph, and real-time security alert feed.
- **STEP 6 — Real-Time Attack Simulation:** Switch to "Simulated Traffic Generator" &rarr; select "Port Scan Attack" or "SYN Flood DDoS". Click "STEP +25 PACKETS" or "START INGESTION". Point out how the anomaly detector flags the attack, triggers alerts, and updates host node colors on the communication topology.

---

## 17. Limitations & Future Scope
- **Live Capture Dependencies:** On Windows, live network interface sniffing requires Npcap. The platform gracefully falls back to native PCAP replay and simulation if Npcap is unavailable.
- **Passive Analysis Scope:** In accordance with ethical guidelines, the platform is strictly a passive analytics and visualization tool; it does not drop packets, block IPs, or modify firewall rules.
- **Future Scope:** Adding optional Suricata EVE JSON file tailing, Zeek connection log parsing, and WebSocket streaming for distributed collectors.

---
*Developed for Academic Cybersecurity Research, Project Evaluation & Security Operations Demonstration.*
