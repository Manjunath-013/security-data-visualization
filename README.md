---
title: CyberShield - Cybersecurity Threat Intelligence & Network Analytics
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# 🛡️ CyberShield: Cybersecurity Threat Intelligence & Network Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Scapy](https://img.shields.io/badge/Scapy-2.6%2B-009688?style=for-the-badge)](https://scapy.net/)
[![Tests](https://img.shields.io/badge/Tests-27%20Passed-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)]()

**CyberShield** is an interactive, full-stack **Cybersecurity Data Visualization & Threat Analytics Platform**. It bridges the gap between macro-level geopolitical cyber intelligence and micro-level line-rate network telemetry, integrating rule-based Intrusion Detection Systems (IDS), automated SOAR incident response playbooks, and predictive machine learning models into a unified security operations command center.

---

## 📑 Table of Contents

- [Executive Overview](#-executive-overview)
- [System Architecture](#-system-architecture)
- [Core Workspaces & Features](#-core-workspaces--features)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Quickstart Guide (Clone & Run)](#-quickstart-guide-clone--run)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Set Up a Virtual Environment](#2-set-up-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Launch the Platform](#4-launch-the-platform)
- [Running Automated Tests](#-running-automated-tests)
- [Machine Learning Integrity & Performance](#-machine-learning-integrity--performance)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Contributing & License](#-contributing--license)

---

## 📌 Executive Overview

Modern security teams struggle to correlate high-level geopolitical threat campaign trends with low-level network packet anomalies. **CyberShield** unifies these two perspectives:

1. **Macro Geopolitical Threat Intelligence**: Explores **3,414 verified cyber operations** from the European Repository of Cyber Incidents (EuRepoC v1.3) spanning **120+ nations**, charting nation-state threat actors, targeted critical infrastructure sectors, and MITRE ATT&CK techniques.
2. **Micro Line-Rate Network Telemetry & IDS**: Features a real-time Scapy packet engine capturing Layer 3 & 4 metadata (TCP, UDP, ICMP, DNS), detecting intrusions via **6 heuristic anomaly detection signatures** paired with automated SOAR containment playbooks.
3. **Predictive AI & NLP**: Employs a supervised Linear SVM + TF-IDF pipeline delivering **91.23% empirical accuracy** on holdout incident writeups to forecast cyber attack impact severity.

---

## 🏗️ System Architecture

```text
+-----------------------------------------------------------------------------------+
|               CYBERSHIELD: CYBERSECURITY DATA VISUALIZATION PLATFORM               |
+-----------------------------------------------------------------------------------+
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
┌─────────────────────────────────┐                     ┌──────────────────────────────────┐
│   LINE-RATE TELEMETRY LAYER     │                     │   HISTORICAL INTELLIGENCE LAYER  │
├─────────────────────────────────┤                     ├──────────────────────────────────┤
│ • Scapy Packet Analysis Engine  │                     │ • EuRepoC Global Incidents DB    │
│ • Synthetic Profile Traffic Gen │                     │ • MITRE ATT&CK Framework v14.1   │
│ • Layer 3 & 4 Header Decoder    │                     │ • 20+ Years Geopolitical History │
│ • Deep-Dive Packet Inspector    │                     │ • Critical Infrastructure Matrix │
└────────────────┬────────────────┘                     └─────────────────┬────────────────┘
                 │                                                        │
                 ▼                                                        ▼
┌─────────────────────────────────┐                     ┌──────────────────────────────────┐
│    STATEFUL DETECTION ENGINE    │                     │   PREDICTIVE AI & NLP ENGINES    │
├─────────────────────────────────┤                     ├──────────────────────────────────┤
│ • 6 Heuristic IDS Signatures    │                     │ • NLP Impact Assessment Engine:  │
│ • Port Scan & SYN Flood Rules   │                     │   TF-IDF + Linear SVM (91.23%)   │
│ • System Visualization Posture  │                     │ • 1-Click Scenario Evaluator     │
│ • Automated SOAR Response Play  │                     │ • Phishing URL Detector Model    │
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

## 🚀 Core Workspaces & Features

| # | Workspace | Key Functionality |
| :---: | :--- | :--- |
| **1** | **🛡️ Security Operations & Dashboard** | Central command view showing real-time threat posture status (`NORMAL`, `ELEVATED`, `CRITICAL`), executive KPIs (3,414 incidents, alert counts, ransomware rates), and comparative attack category distributions. |
| **2** | **📡 Live Packet Capture & Visualizer** | Interactive network telemetry generator with 4 scenario profiles (*Standard Traffic*, *Recon Scan*, *SYN Flood*, *Traffic Spike*), Layer 3/4 header decoders, private/public IP classification, and protocol decoders. |
| **3** | **🚨 Real-Time Intrusion Detection (IDS)** | Rule-based detection engine evaluating 6 attack signatures (Vertical Port Scan, TCP SYN Flood, Subnet Sweep, Bandwidth Burst, DNS Anomaly, Large Exfiltration) accompanied by automated **SOAR containment playbooks** (`iptables` rules, quarantine steps). |
| **4** | **🗺️ Global Threat Map & Attack Flows** | Choropleth world map across 120+ countries and multi-hop Sankey flow diagram (*Threat Origin ➔ Attack Vector ➔ Target Country ➔ Target Sector*). |
| **5** | **📈 Longitudinal Threat Trends** | Multi-year historical timeline tracking evolution of attacks against Critical Infrastructure, Healthcare, Government, and Financial sectors. |
| **6** | **🧩 MITRE ATT&CK Matrix** | Tactical mapping across Initial Access, Execution, Persistence, Lateral Movement, and Impact with interactive technique frequencies. |
| **7** | **🤖 AI Impact Prediction (NLP Classifier)** | Supervised Linear Support Vector Machine (Linear SVM) trained with TF-IDF vectorization to predict operational impact directly from raw incident narratives, plus interactive sample scenarios. |
| **8** | **📁 Threat Dossier & Incident Search** | Forensic searchable database with multi-field filtering, risk score calculation, and 1-click filtered CSV data export. |

---

## 📁 Repository Structure

```text
security-data-visualization/
├── app.py                      # Main Streamlit multi-page dashboard application
├── requirements.txt            # Python dependencies
├── inspect_ml_targets.py       # Utility for inspecting dataset target label distributions
├── select_features.py          # Data pipeline for feature selection & correlation analysis
│
├── data/                       # Pre-bundled datasets
│   ├── incidents.csv           # Full EuRepoC v1.3 global incident dataset (3,414 records)
│   ├── selected_incidents.csv  # Curated ML-ready features dataset
│   ├── phishing/               # Phishing URL benchmark dataset
│   └── realtime/               # Sample network PCAP captures
│
├── models/                     # Trained ML model weights and pipelines
│   ├── mitre_impact_final_model.joblib
│   ├── mitre_initial_access_final_model.joblib
│   └── phishing_model/
│
├── outputs/                    # Exported evaluation metrics, plots, and confusion matrices
│   └── ml/
│
├── src/                        # Modular platform source code
│   ├── analysis.py             # Statistical and geopolitical aggregation logic
│   ├── anomaly_detector.py     # Network anomaly detection algorithms
│   ├── attacker_intelligence.py# Threat actor and Sankey flow builders
│   ├── data.py                 # Data access layer
│   ├── ml_pipeline.py          # Scikit-learn NLP training and inference engine
│   ├── preprocessing.py        # Data cleaning, normalization, and feature extraction
│   ├── risk_engine.py          # Multi-factor risk scoring engine
│   └── realtime/               # Network sniffing and telemetry modules
│       ├── event_processor.py  # Layer 3/4 header decoder and IP geolocation
│       ├── ids_engine.py       # Stateful 6-signature heuristic IDS
│       └── packet_collector.py # Thread-safe singleton packet sniffer & replay engine
│
└── tests/                      # Automated unit test suite
    ├── test_platform.py        # Tests for preprocessing, analysis, and risk scoring
    └── test_realtime.py        # Tests for network parsing, IDS rules, and collectors
```

---

## ⚙️ Prerequisites

Before you start, ensure you have the following installed on your system:

- **Python**: Version **3.10**, **3.11**, **3.12**, or **3.13+** (`python --version`)
- **Git**: For cloning the repository (`git --version`)
- *(Optional)* **Npcap / libpcap**: Only required if you intend to capture live packets directly from physical network interfaces using Scapy. For pre-bundled PCAP replays and synthetic profile simulations, **no external packet driver is needed**.

---

## 💻 Quickstart Guide (Clone & Run)

Follow these steps to clone and run the application on your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/Manjunath-013/security-data-visualization.git
cd security-data-visualization
```

### 2. Set Up a Virtual Environment

It is strongly recommended to use an isolated Python virtual environment:

#### On Windows (PowerShell / Command Prompt):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# (Or Command Prompt)
.\venv\Scripts\activate.bat
```

> **Note for Windows PowerShell users**: If you encounter an execution policy error when activating the venv, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### On macOS & Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Install Dependencies

Upgrade `pip` and install all required libraries:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Platform

Run the Streamlit application:

```bash
streamlit run app.py
```

Once started, Streamlit will display local and network URLs in your terminal:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

Open **[http://localhost:8501](http://localhost:8501)** in your web browser to explore CyberShield!

---

## 🧪 Running Automated Tests

CyberShield comes with a comprehensive unit test suite covering data preprocessing, analytics engines, network packet parsers, and IDS heuristics.

To run all 27 tests:

```bash
python -m unittest discover -s tests
```

Expected output:
```text
...........................
----------------------------------------------------------------------
Ran 27 tests in 2.0s

OK
```

---

## 🔬 Machine Learning Integrity & Performance

The platform uses a supervised **Linear Support Vector Machine (Linear SVM)** coupled with a **TF-IDF vectorizer** to categorize incident impact severity based on threat incident narratives.

> **Academic Rigor Guarantee**: Evaluated with a stratified holdout train/test split (`random_state=42`) to guarantee no data leakage between training and testing subsets.

| Metric | Empirical Score | Description |
| :--- | :---: | :--- |
| **Accuracy** | **91.23%** | Overall correct classification rate on untouched holdout test set |
| **Macro F1-Score** | **0.90** | Balanced harmonic mean accounting for class distribution |
| **Training Samples** | **1,480** | Preprocessed EuRepoC narrative descriptions |
| **Test Holdout** | **370** | Independent evaluation records |

Confusion matrices and metric reports are exported to `outputs/ml/`.

---

## ❓ Troubleshooting & FAQs

### Port `8501` is already in use
If another application or instance of Streamlit is already running on port 8501, specify an alternative port:
```bash
streamlit run app.py --server.port 8502
```

### `WARNING: No libpcap provider available ! pcap won't be used`
This warning from Scapy is purely informational on Windows systems without WinPcap/Npcap. CyberShield's synthetic traffic simulator and pre-bundled PCAP analyzer will function without issues. If you want to sniff packets live from your physical Wi-Fi/Ethernet adapter, install [Npcap](https://npcap.com/) with "WinPcap API-compatible Mode" checked.

### Streamlit asks for an email address on first run
You can press `Enter` to skip, or run:
```bash
streamlit run app.py --server.headless true
```

### `ModuleNotFoundError: No module named 'src'`
Ensure you are running the `streamlit run app.py` or `python -m unittest discover -s tests` command from the root directory where `app.py` resides.

---

## 👥 Contributing & License

- **License**: Released for academic, educational, and research use under the MIT License.
- **Contributions**: Issues, pull requests, and feedback are welcome! Feel free to fork the repository, open an issue, or submit a feature enhancement.
