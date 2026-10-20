# 🔐 KEYMORPH
### Dynamic Morse-Style Encoding with Neural Sequence Recovery

KEYMORPH is a **Language Modelling and Text Analytics (LMTA) mini-project** that explores dynamic Morse-style encoding, seed-based key mapping, and neural sequence recovery under simulated communication-channel interference.

The system combines a polymorphic encoding pipeline with a Bidirectional Gated Recurrent Unit (BiGRU) and Bahdanau Attention to explore the recovery of corrupted message sequences.

## 🎯 Project Objectives

- Explore dynamic, seed-dependent Morse-style encoding.
- Simulate communication-channel interference and symbol corruption.
- Investigate neural sequence recovery using recurrent neural networks and attention.
- Implement key-based routing with decoy outputs for invalid keys.

## ⚙️ System Architecture

The proposed pipeline consists of the following stages:

1. **Command + Seed:** Provide the input command or message and select a shared seed key (`K0–K9`).
2. **Polymorphic Encoder:** Transform the input using a seed-dependent encoding mapping.
3. **EW Channel:** Simulate communication interference and possible symbol corruption.
4. **Neural Repair:** Use a Bidirectional GRU with Bahdanau Attention to help reconstruct the corrupted sequence.
5. **Key-Based Routing:** Route the output according to key validation, returning a decoy output for an invalid key.

## 🧠 Technologies Used

- **Python** — Core implementation
- **PyTorch** — Neural network implementation and model handling
- **Gradio** — Interactive application interface
- **NumPy** — Numerical operations
- **Bidirectional GRU** — Sequence modelling
- **Bahdanau Attention** — Attention-based sequence processing

## 📁 Project Structure

```text
KEYMORPH/
├── simulation.py       # Simulation and application logic
├── train.py            # Model training
├── model_weights.pth   # Saved model weights
├── vocab.pt            # Saved vocabulary
├── requirements.txt    # Python dependencies
├── .gitignore          # Git exclusions
└── README.md           # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Git
- pip

### 1. Clone the repository

```bash
git clone https://github.com/Godson-1307/KEYMORPH.git
cd KEYMORPH
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the project

Inspect `simulation.py` for the application's entry point and run command. The exact command depends on how the simulation and interface are initialized in the implementation.

## 📦 Model Artifacts

- `model_weights.pth` — Saved PyTorch model parameters.
- `vocab.pt` — Serialized vocabulary used by the model.

These artifacts may be required for inference and sequence processing.

## 🔬 Project Scope

KEYMORPH brings together concepts from sequence modelling, attention mechanisms, dynamic encoding, and simulated communication-channel noise.

This is an academic prototype intended for experimentation and demonstration. Its security properties have not been independently verified, and it should not be treated as a production-grade encryption system.

## 🎓 Academic Information

- **Institution:** St. Francis Institute of Technology (SFIT)
- **Department:** Artificial Intelligence and Machine Learning
- **Course:** Language Modelling and Text Analytics (LMTA)
- **Project:** KEYMORPH — LMTA Mini-Project

## 👥 Team

- Viraj Salunke
- Godson Nadar
- Jhoshua Coutinho

## 📄 License

No license has currently been specified. Please contact the project contributors before reusing or redistributing the code.
