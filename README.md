# 🔍 Bias Detective — Algorithmic Fairness & Model Bias Auditing Engine

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=311042,701A75,D946EF&height=220&section=header&text=Bias%20Detective&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Algorithmic%20Fairness%20%26%20Model%20Bias%20Auditing%20Engine&descSize=18&descAlignY=60&descAlign=50" width="100%" alt="Bias Detective Header Banner" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python%203.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Engine-Hugging%20Face%20Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Backend-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Focus-AI%20Fairness%20%26%20Ethics-701A75?style=for-the-badge" alt="AI Ethics" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 📌 Table of Contents

- [📖 Executive Summary](#-executive-summary)
- [🎯 Problem Statement & Solution](#-problem-statement--solution)
- [✨ Key Features & Capabilities](#-key-features--capabilities)
- [🏗️ System Architecture & Workflow](#️-system-architecture--workflow)
- [🛠️ Tech Stack Matrix](#️-tech-stack-matrix)
- [⚙️ Reproducible Quickstart & Setup](#️-reproducible-quickstart--setup)
- [📁 Project Structure](#-project-structure)
- [📊 Evaluation Metrics & Output](#-evaluation-metrics--output)
- [👩‍💻 Author & Contributions](#-author--contributions)

---

## 📖 Executive Summary

**Bias Detective** is an interactive open-source algorithmic auditing toolkit designed to evaluate, quantify, and visualize demographic bias and sentiment disparities in pre-trained generative language models. By running causal language models (such as GPT-2 or lightweight variants) locally through Hugging Face Transformers, the system executes controlled counterfactual prompt testing across sensitive demographic attributes—measuring token probabilities, lexical sentiment shifts, and paired word-count distributions with rigorous audit reporting.

---

## 🎯 Problem Statement & Solution

- **The Problem:** Modern generative models trained on vast web corpora absorb and amplify systemic societal prejudices, gender stereotypes, and demographic biases. Developers often deploy these models without auditing how changing a simple demographic identifier (e.g., gender, nationality, or ethnicity) alters the model's tone and lexical generation.
- **The Solution:** Bias Detective provides an automated counterfactual evaluation harness. Users supply matched prompt pairs differing strictly in demographic target phrases, and the engine conducts simultaneous local inference, computes statistical sentiment metrics, plots lexical rate distributions, and outputs a reproducible audit dossier.

---

## ✨ Key Features & Capabilities

- **⚖️ Counterfactual Prompt Auditing:** Paired generation engine comparing model outputs on identical semantic frames with perturbed demographic variables.
- **🧠 Local Transformer Inference:** Zero third-party API exposure—runs inference locally using Hugging Face Transformers (`gpt2`, `sshleifer/tiny-gpt2`, or custom models) via PyTorch.
- **📊 Quantitative Lexical Analysis:** Evaluates positive and negative sentiment token frequencies, length distributions, and token divergence rates across groups.
- **📈 Interactive Streamlit Visualizations:** Real-time side-by-side completion displays, comparative bar charts, and sentiment differential metrics.
- **📑 Automated Compliance Dossier:** Generates structured fairness reports summarizing statistical evidence of bias for AI governance and documentation.

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Bias Detective Architecture                           │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────┐        ┌──────────────────────┐        ┌───────────────────┐
│ Counterfactual    │───────►│ Hugging Face Causal  │───────►│ Lexical & Stat    │
│ Prompt Generator  │        │ Transformer Engine   │        │ Analysis Engine   │
│ (Demographic Pair)│        │ (Local PyTorch / LM) │        │ (Sentiment & Rate)│
└───────────────────┘        └──────────────────────┘        └─────────┬─────────┘
                                                                       │
                                                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Streamlit Interactive UI                             │
│   ┌───────────────────────────┐         ┌───────────────────────────────────┐  │
│   │ Comparative Token Chart   │         │ Matched Output Differential Card  │  │
│   └───────────────────────────┘         └───────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack Matrix

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core language for analysis and UI orchestration |
| **Deep Learning** | PyTorch (`torch>=2.2.0`) | Tensor computation and local model execution |
| **Transformers** | Hugging Face (`transformers>=4.41.0`) | Pre-trained causal LM loading, tokenization & pipeline |
| **Data & Metrics** | Pandas (`pandas>=2.0.0`) | Dataframe manipulation, frequency aggregation |
| **Interface** | Streamlit (`streamlit>=1.36.0`) | Reactive web interface, interactive sliders & plotting |

---

## ⚙️ Reproducible Quickstart & Setup

### Prerequisites
- Python 3.10 or newer installed
- ~1–3 GB free disk space for local model cache (or minimal for `tiny-gpt2`)

### 1. Clone the Repository
```bash
git clone https://github.com/keerthi19hub/Bias_detective.git
cd Bias_detective
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the Interactive Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to start running demographic bias audits.

---

## 📁 Project Structure

```text
Bias_detective/
├── app.py                  # Streamlit application dashboard and UI layout
├── model_backend.py        # Hugging Face pipeline integration and text generation logic
├── analysis.py             # Lexical sentiment metrics, counterfactual divergence calculations
├── preprocessing.py        # Prompt normalization, demographic token pair substitution
├── demo_screenshot.png     # Interface preview and visual audit demonstration
├── requirements.txt        # Production dependency specifications
└── README.md               # Technical documentation
```

---

## 📊 Evaluation Metrics & Output

- **Token Divergence Index (TDI):** Quantifies variance in generation trajectory between demographic pairs.
- **Lexical Sentiment Ratio:** Compares positive vs. negative semantic word occurrences using structured wordlists.
- **Length Disparity:** Measures whether specific demographic prompts trigger disproportionately curt or verbose completions.

---

## 👩‍💻 Author & Contributions

**Keerthi N**  
- **GitHub:** [@keerthi19hub](https://github.com/keerthi19hub)  
- **Portfolio:** [keerthi19hub.github.io](https://keerthi19hub.github.io)  
- **Role:** End-to-end architecture, prompt evaluation harness, statistical analysis algorithms, and Streamlit interface design.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=311042,701A75,D946EF&height=100&section=footer" width="100%" alt="Footer Banner" />
</p>
