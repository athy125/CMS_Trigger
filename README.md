
```
# CMS Level-1 Trigger Simulation

![Trigger Rate Plot](data/trigger_rate_sample.png)

A simulation framework for the CMS Level-1 (L1) Trigger system, built with C++ and Python to model particle collision events at the Large Hadron Collider (LHC). This project mimics the L1 Trigger's rapid filtering of 40 MHz events, offering a software-based exploration of detector responses, trigger decisions, and physics processes through advanced visualizations.

## Contents
- [Overview](#overview)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
The CMS experiment at CERN uses its Level-1 Trigger to select potentially significant collision events in real-time, reducing the LHC's 40 MHz bunch-crossing rate to a manageable 100 kHz. This project simulates that process, capturing energy deposits in key detector subsystems—electromagnetic (ECAL) and hadronic (HCAL) calorimeters, muon chambers, and silicon tracker—while applying trigger logic to identify signatures like muons, jets, and electrons. Implemented in C++ (`main.cpp`) for efficiency and Python (`tvis.py`) for interactive analysis, it generates events ranging from QCD backgrounds to leptonic signals, providing a platform to study trigger performance and event characteristics.

## Features
- **C++ Core (`main.cpp`)**:
  - Multi-threaded event processing with realistic detector layouts.
  - Trigger algorithms: single muons (pT > 25 GeV), di-jets (pT > 50 GeV), electrons/photons (pT > 30 GeV).
  - Event generation for QCD, single-lepton, dilepton, and multijet processes.
  - JSON output for event data.

- **Python Visualizations (`tvis.py`)**:
  - *Live Particle Animation*: Real-time trajectories (η, φ) during simulation.
  - *3D Detector Display*: Cylindrical view of ECAL energy deposits.
  - *Trigger Tuning Dashboard*: Interactive pT threshold adjustments with rate feedback.
  - *Event Timeline*: Process-typed (QCD, leptonic) event sequence with tooltips.
  - *AR Export*: JSON files for augmented reality integration.

- **Metrics**: Efficiency, acceptance rates, and detailed trigger statistics.

## Setup
### Requirements
- **C++**: `g++` with C++17 support (e.g., GCC 11+).
- **Python**: 3.8+.
- **Libraries**: `numpy`, `matplotlib`, `pandas`.

### Steps in GitHub Codespace
1. **Open Repository**:
   - Navigate to `https://github.com/your-username/cms-l1-trigger-sim`.
   - Click "Code" > "Codespaces" > "Create codespace on main".
   - Or clone locally:
     ```bash
     git clone https://github.com/your-username/cms-l1-trigger-sim.git
     cd cms-l1-trigger-sim
     ```

2. **Install Python Dependencies**:
   ```bash
   pip3 install numpy matplotlib pandas
   ```

3. **Compile Simulation**:
   ```bash
   g++ -std=c++17 -pthread main.cpp -o cms_l1_trigger_sim
   ```

4. **Prepare Output Directory**:
   ```bash
   mkdir data
   ```

## Usage
### Simulation Commands
- **Live Run with Visuals**:
  ```bash
  python3 tvis.py --simulate --executable ./cms_l1_trigger_sim --events 500
  ```
  - Shows live plots; saves data to `data/`.

- **Test Data Generation**:
  ```bash
  python3 tvis.py --test --events 50
  ```

- **Parse Output File**:
  ```bash
  ./cms_l1_trigger_sim > output.txt
  python3 tvis.py --parse output.txt
  ```

### Visualization Commands
- **3D Event View**:
  ```bash
  python3 tvis.py --view-event 0
  ```
  - Needs `data/event_0.json` from a prior run.

- **Trigger Tuning**:
  ```bash
  python3 tvis.py --tune
  ```
  - Requires existing event data.

- **Event Timeline**:
  ```bash
  python3 tvis.py --timeline
  ```

- **AR Export**:
  ```bash
  python3 tvis.py --ar-export 0
  ```
  - Generates `data/event_0_ar.json`.

### Outputs
- **Logs**: Event details and summary in console.
- **Files**: JSON events, CSV stats, PNG plots in `data/`.
- **Plots**: Named with timestamps (e.g., `trigger_rate_20250324_120000.png`).

## Project Structure
```
cms-l1-trigger-sim/
├── main.cpp          # Core simulation logic
├── tvis.py           # Visualization and analysis tool
├── data/             # Output directory (auto-created)
│   ├── event_*.json  # Event data
│   ├── trigger_*.png # Visualization outputs
│   └── trigger_log.txt # Execution log
└── README.md         # Project documentation
```

## Contributing
We welcome enhancements! To contribute:
1. Fork this repository.
2. Branch off (`git checkout -b feature-name`).
3. Commit your work (`git commit -m "Add feature X"`).
4. Push to your fork (`git push origin feature-name`).
5. Submit a Pull Request with a clear explanation.

Include:
- Tests for new functionality.
- Updates to this README.
- Rationale for physics or algorithmic changes.

## License
MIT License. See [LICENSE](LICENSE) for details.

```
