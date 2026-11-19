# CyberShield: Cybersecurity Data Visualization & Threat Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![Scapy](https://img.shields.io/badge/Scapy-2.6%2B-green.svg)](https://scapy.net/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()

---

## 📌 Executive Overview

**CyberShield** is an interactive **Cybersecurity Data Visualization & Threat Analytics Platform** designed to visualize multi-dimensional cyber threat intelligence, real-time network traffic flows, and machine learning models.

The platform unifies two core visualization tiers:
1. **Geopolitical Macro Threat Intelligence**: Explores 3,414 publicly reported, verified cyber operations from the European Repository of Cyber Incidents (EuRepoC v1.3) spanning 120+ nations, charting threat actors, target industries, and MITRE ATT&CK techniques.
2. **Line-Rate Micro Network Telemetry & Intrusion Detection (IDS)**: Real-time packet sniffer capturing Layer 3 & 4 metadata (TCP, UDP, ICMP, DNS), detecting anomalous patterns (port scans, SYN floods, traffic spikes) via 6 transparent heuristic rules paired with automated SOAR incident response playbooks.

---

## 🏗️ System Architecture

```
+-----------------------------------------------------------------------------------+
|               CYBERSHIELD: CYBERSECURITY DATA VISUALIZATION PLATFORM               |
+-----------------------------------------------------------------------------------+
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
┌─────────────────────────────────┐                     ┌──────────────────────────────────┐
│   LINE-RATE TELEMETRY LAYER     │                     │   HISTORICAL INTELLIGENCE LAYER  │
├─────────────────────────────────┤                     ├──────────────────────────────────┤
│ • Scapy Line-Rate Packet Sniffer│                     │ • EuRepoC Global Incidents DB    │
│ • Deterministic Traffic Engine  │                     │ • MITRE ATT&CK Framework v14.1   │
│ • Layer 3 & 4 Header Decoder    │                     │ • 20+ Years Geopolitical History │
│ • Packet Deep-Dive Inspector    │                     │ • Critical Infrastructure Matrix │
└────────────────┬────────────────┘                     └─────────────────┬────────────────┘
                 │                                                        │
                 ▼                                                        ▼
┌─────────────────────────────────┐                     ┌──────────────────────────────────┐
│    STATEFUL DETECTION ENGINE    │                     │   PREDICTIVE AI & NLP ENGINES    │
├─────────────────────────────────┤                     ├──────────────────────────────────┤
│ • 6 Heuristic Signatures (IDS)  │                     │ • NLP Impact Assessment Engine:  │
│ • Port Scan & SYN Flood Rules   │                     │   TF-IDF + Linear SVM (91.23%)   │
│ • System Visualization Posture  │                     │ • 1-Click Scenario Evaluator     │
│ • Automated SOAR Response Play  │                     │ • Cross-Validation Benchmarks    │
└────────────────┬────────────────┘                     └─────────────────┬────────────────┘
                 │                                                        │
                 └────────────────────────┬───────────────────────────────┘
                                          │
                                          ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                 INTERACTIVE CYBERSECURITY VISUALIZATION WORKSPACES               │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1. 🛡️ Security Operations & Threat Dashboard   4. 🗺️ Global Threat Map & Flows   │
│ 2. 📡 Live Packet Capture & Traffic Visualizer  5. 📈 Longitudinal Threat Trends  │
│ 3. 🚨 Real-Time Intrusion Detection (IDS)       6. 🧩 MITRE ATT&CK Matrix         │
│ 7. 🤖 AI Impact Prediction (NLP Classifier)    8. 📁 Threat Dossier & Search     │
+──────────────────────────────────────────────────────────────────────────────────+
```

---

## 🚀 Visualization Workspaces & Features

### 1. 🛡️ Security Operations & Threat Dashboard
- **System Visualization Status & Threat Posture**: Real-time status indicator (`NORMAL MONITORING`, `ELEVATED ANOMALY`, `CRITICAL INTRUSION`) reflecting live network health.
- **Executive Threat KPIs**: Verified Threat Incidents, Real-Time IDS Alerts, Ingested Packets, and Ransomware Exposure Rate.
- **Architectural Triad Card**: Explains Macro Threat Intel, Micro Network Telemetry, and Autonomous AI in 30 seconds.
- **Comparative Threat Charts**: Historical attack category distribution side-by-side with real-time protocol shares.

### 2. 📡 Live Packet Capture & Traffic Visualizer
- **4-Profile Traffic Scenario Deck**:
  - 🟢 **Standard Traffic** (Baseline HTTP, HTTPS, DNS, API)
  - 🟡 **Reconnaissance Scan** (Nmap port sweep probing sensitive ports)
  - 🔴 **Volumetric DoS Flood** (High-rate TCP SYN flood)
  - 🟣 **Unusual Traffic Spike** (High volume data download / bandwidth burst)
- **Interactive Packet Deep-Dive Decoder**:
  - Click any captured packet to decode Layer 3 & 4 parameters.
  - Automatically translates port numbers into standard services (`Port 443 = HTTPS`, `Port 53 = DNS`, `Port 22 = SSH`).
  - Differentiates between internal LAN (`RFC 1918 Private`) and public external endpoints.
  - Generates a plain-English analyst explanation of the packet's activity.
- **Networking Fundamentals Cheat Sheet**: Collapsible reference covering TCP 3-way handshakes, TCP vs. UDP, and standard ports.

### 3. 🚨 Real-Time Intrusion Detection (IDS)
- **6 Heuristic Detection Signatures**:
  1. **Vertical Port Scan**: Single IP probing $\ge 8$ destination ports in 60s (`T1046`).
  2. **TCP SYN Flood**: Rapid incomplete TCP SYN handshakes (`T1498`).
  3. **Horizontal Subnet Sweep**: Single IP probing the same port across $\ge 6$ internal hosts (`T1046`).
  4. **Volumetric Traffic Burst**: Throughput spike $> 300\%$ above rolling baseline (`T1499`).
  5. **DNS Anomaly**: Abnormally high frequency of encoded queries or domain query bursts (`T1071.004`).
  6. **Large Outbound Transfer**: Heavy payload transmission to external endpoints (`T1041`).
- **Automated SOAR Incident Response Playbook**:
  - Generates actionable Standard Operating Procedures (SOP) for any triggered alert.
  - Maps alerts to their specific **Cyber Kill Chain Stage** (*Reconnaissance*, *C2 Communication*, *Actions on Objectives*).
  - Automatically outputs executable firewall containment commands:
    ```bash
    iptables -A INPUT -s <Source_IP> -j DROP
    ```
  - Recommends network quarantine and volatile memory acquisition steps.

### 4. 🗺️ Global Threat Map & Attack Flows
- Interactive choropleth map across 120+ nations with country-level incident counts.
- Sankey relational flow chart visualizing *Threat Origin State ➔ Attack Vector ➔ Target Country ➔ Target Sector*.

### 5. 📈 Longitudinal Threat Trends & Dynamics
- Multi-year temporal trends covering 20+ years of cyber warfare.
- Comparative analysis across Critical Infrastructure, Healthcare, Finance, and Public Administration.
- Peak incident surge detection correlating with major geopolitical conflicts.

### 6. 🧩 MITRE ATT&CK Matrix & Framework
- Mapping incidents across standard ATT&CK Tactics (Initial Access, Execution, Persistence, Lateral Movement, Exfiltration, Impact).
- Interactive technique frequency bars and impact category breakdowns.

### 7. 🤖 AI Impact Prediction (NLP Classifier)
- Supervised Linear Support Vector Machine (Linear SVM) with TF-IDF vectorization.
- **Empirical Validation Accuracy**: **91.23%** (Macro F1: **0.90**) on untouched holdout incident writeups.
- 1-Click interactive scenario testing for live text classification.

### 8. 📁 Threat Dossier & Incident Search
- Searchable forensic database across historical incident writeups with multi-factor risk scoring.
- 1-click filtered CSV export for security presentations.

---

## 🔬 Machine Learning Evaluation & Integrity

> **Academic Rigor Guarantee**: CyberShield strictly avoids synthetic score inflation or data leakage. Train and test sets are partitioned using stratified holdouts with reproducible seeds (`random_state=42`).

| Model | Target Task | Algorithm | Train / Test Samples | Empirical Accuracy | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MITRE Impact NLP** | Incident Impact Prediction | Linear SVM + TF-IDF | 1,480 / 370 | **91.23%** | 0.90 |

---

## 💻 Quickstart & Launch

```bash
# 1. Navigate to the project directory
cd security_data_visualization(meg)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the platform
streamlit run app.py
```
Access the application at `http://localhost:8501`.

---
*Developed for College Cybersecurity Data Visualization Project Evaluation.*
