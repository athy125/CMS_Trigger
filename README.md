
# CMS Level-1 Trigger Simulation

![Simulation Screenshot](data/trigger_rate_example.png)  
*Example output: Trigger acceptance rate over time from a 500-event simulation.*

## Overview

The CMS Level-1 Trigger Simulation is a sophisticated software framework designed to model and visualize the Level-1 (L1) trigger system of the Compact Muon Solenoid (CMS) experiment at CERN's Large Hadron Collider (LHC). The L1 trigger is a critical hardware-based system that reduces the event rate from approximately 40 MHz to 100 kHz by identifying signatures of interesting physics processes (e.g., muons, jets, electrons) in real-time. This project simulates the L1 trigger's decision-making process, detector interactions, and event reconstruction, providing a platform for studying trigger performance, efficiency, and physics signatures.

The simulation is implemented in C++ (`main.cpp`) for high-performance event generation and trigger logic, paired with a Python visualization suite (`tvis.py`) for interactive, graphical analysis. Key features include:

- **Realistic Detector Modeling**: Emulates CMS subsystems (ECAL, HCAL, Muon, Tracker) with energy deposits in eta-phi space.
- **Trigger Algorithms**: Implements single muon, di-jet, and electron/photon triggers with configurable thresholds.
- **Advanced Visualizations**: Offers 2D/3D event displays, real-time particle animations, trigger tuning dashboards, and physics process timelines.
- **Augmented Reality (AR) Export**: Generates data for AR visualization of detector events.

This project is ideal for experimental purposes, trigger optimization studies, and prototyping new L1 algorithms in a controlled environment.

---

## Operational Principles

The CMS L1 trigger operates under stringent latency constraints (< 4 μs per event) and must process proton-proton collisions at the LHC's 40 MHz bunch crossing rate. It relies on coarse-grained data from calorimeters (ECAL, HCAL) and muon detectors, supplemented by tracker information in modern upgrades (e.g., Phase-2). This simulation abstracts these components into a software model, focusing on:

- **Event Generation**: Simulates QCD, single-lepton, dilepton, and multijet processes with realistic energy distributions.
- **Trigger Reconstruction**: Identifies trigger candidates (muons, jets, electrons/photons) based on energy deposits and isolation criteria.
- **Performance Metrics**: Computes trigger efficiency (passed events / total events) and rates per physics process.

The framework is extensible, allowing users to modify trigger thresholds, add new algorithms, or integrate with real CMS data for validation.

---

## Repository Structure

```
cms-l1-trigger-sim/
├── main.cpp          # C++ simulation core
├── tvis.py           # Python visualization script
├── data/             # Output directory for JSON files, plots, and logs
│   ├── event_*.json  # Per-event detector and candidate data
│   ├── trigger_rate_*.png  # Acceptance rate plots
│   └── trigger_log.txt  # Simulation log
└── README.md         # This documentation
```

---

## Prerequisites

### Software Requirements
- **C++ Compiler**: `g++` with C++17 support (e.g., GCC 11+).
- **Python**: Version 3.8+.
- **Python Libraries**:
  - `numpy` (>=1.23.0): Numerical computations.
  - `matplotlib` (>=3.8.0): Plotting and visualization.
  - `pandas` (>=2.2.0): Data manipulation.

### Hardware Recommendations
- **OS**: Linux (tested on Ubuntu 22.04 via GitHub Codespaces), macOS, or Windows (with WSL).
- **RAM**: ≥4 GB for 500-event simulations.
- **Storage**: ~100 MB for output data from 500 events.

---

## Installation

### Local Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/cms-l1-trigger-sim.git
   cd cms-l1-trigger-sim
   ```

2. **Install Python Dependencies**:
   ```bash
   pip3 install numpy matplotlib pandas
   ```

3. **Verify Tools**:
   - Check `g++`: `g++ --version`
   - Check Python: `python3 --version`

### GitHub Codespaces
1. **Open Codespace**:
   - Navigate to `https://github.com/<your-username>/cms-l1-trigger-sim`.
   - Click “Code” > “Codespaces” > “Create codespace on main”.

2. **Install Dependencies**:
   - In the Codespace terminal:
     ```bash
     pip3 install numpy matplotlib pandas
     ```

3. **Create Data Directory**:
   ```bash
   mkdir data
   ```

---

## Compilation

Compile the C++ simulation executable:
```bash
g++ -std=c++17 -pthread main.cpp -o cms_l1_trigger_sim
```
- **Flags**: `-std=c++17` enables modern C++ features; `-pthread` supports multi-threading.
- **Output**: Creates `cms_l1_trigger_sim` executable.

---

## Usage

The project supports multiple execution modes via `tvis.py`, integrating simulation and visualization.

### 1. Live Simulation with Animation
Run the simulation and visualize live rates and particle trajectories:
```bash
python3 tvis.py --simulate --executable ./cms_l1_trigger_sim --events 500
```
- **Output**: Console logs, JSON files in `data/`, live Matplotlib windows (rate plot + particle animation).

### 2. Parse Existing Output
Analyze a pre-run simulation log:
```bash
./cms_l1_trigger_sim > output.txt
python3 tvis.py --parse output.txt
```

### 3. Generate Test Data
Create synthetic events for testing:
```bash
python3 tvis.py --test --events 50
```

### 4. 3D Event Display
Visualize a specific event in 3D:
```bash
python3 tvis.py --view-event 0
```
- **Requires**: `data/event_0.json` from a prior simulation.

### 5. Trigger Tuning Dashboard
Interactively adjust trigger thresholds:
```bash
python3 tvis.py --tune
```
- **Requires**: Prior simulation data.

### 6. Event Timeline
View a timeline of events by physics process:
```bash
python3 tvis.py --timeline
```
- **Features**: Clickable points with tooltips.

### 7. AR Export
Export an event for augmented reality:
```bash
python3 tvis.py --ar-export 0
```
- **Output**: `data/event_0_ar.json` (convert to GLTF externally).

---

## Command-Line Arguments

| Argument          | Type    | Description                          | Default |
|-------------------|---------|--------------------------------------|---------|
| `--simulate`      | Flag    | Run live simulation                  | -       |
| `--executable`    | String  | Path to `cms_l1_trigger_sim`         | -       |
| `--events`        | Integer | Number of events to simulate         | 500     |
| `--parse`         | String  | Parse simulation output file         | -       |
| `--test`          | Flag    | Generate test data                   | -       |
| `--view-event`    | Integer | Display event in 3D                  | -       |
| `--tune`          | Flag    | Open trigger tuning dashboard        | -       |
| `--timeline`      | Flag    | Show event timeline                  | -       |
| `--ar-export`     | Integer | Export event for AR                  | -       |

---

## Output Description

- **Console**: Event processing logs (e.g., “Event X PASSED | Candidates: Y”), summary stats.
- **JSON Files**: `data/event_X.json` contains detector energy maps and candidate details.
- **Plots**: Saved in `data/` (e.g., `trigger_rate_*.png`, `3d_event_*.png`).
- **Log**: `data/trigger_log.txt` records runtime events.

---

## Code Architecture

### `main.cpp`
- **Namespaces**: `CMSL1Sim` encapsulates all classes and enums.
- **Key Classes**:
  - `DetectorLayer`: Base class for ECAL, HCAL, Muon, and Tracker subsystems.
  - `CollisionEvent`: Stores event data (ID, detectors, candidates, trigger decision).
  - `TriggerLogic`: Abstract base for trigger algorithms (e.g., `SingleMuonLogic`).
  - `L1TriggerSystem`: Manages event queue and multi-threaded processing.
  - `EventGenerator`: Simulates physics processes with random energy deposits.
- **Features**: Thread-safe event processing, JSON export, trajectory tracking.

### `tvis.py`
- **Class**: `TriggerVisualizer` handles simulation parsing and visualization.
- **Visualizations**:
  - 2D: Trigger rates, distributions.
  - 3D: Detector event displays.
  - Interactive: Real-time animations, tuning dashboard, timeline with tooltips.
- **Dependencies**: Leverages Matplotlib for plotting, Pandas for data handling.

---

## Physics Processes

The simulation models four LHC-like processes:
1. **QCD**: High-rate background with random energy deposits (20–50 hits).
2. **Single Lepton**: Muon or electron with tracker activity (30–80 GeV).
3. **Dilepton**: Two leptons, mimicking rare signals (e.g., Z boson decay).
4. **Multijet**: 3–6 jets with clustered energy (20–100 GeV).

Process weights (60% QCD, 20% Single Lepton, 10% Dilepton, 10% Multijet) reflect typical LHC conditions.

---

## Extending the Project

1. **New Triggers**: Inherit from `TriggerLogic` and add to `L1TriggerSystem::triggerMenu`.
2. **Detector Upgrades**: Extend `DetectorLayer` for Phase-2 tracker or HGCAL.
3. **Real Data**: Replace `EventGenerator` with CMS Open Data parsing.
4. **AR Integration**: Convert JSON exports to GLTF using tools like Blender.

---

## Troubleshooting

### Compilation Errors
- **Error**: `g++: command not found`
  - **Fix**: Install `g++` in Codespace: `sudo apt update && sudo apt install g++`.

### Python Dependency Issues
- **Error**: `ModuleNotFoundError: No module named 'numpy'`
  - **Fix**: Re-run `pip3 install numpy matplotlib pandas`.

### Empty Plots
- **Issue**: `plot_trigger_rates` shows nothing.
- **Fix**: Ensure simulation ran first to populate `rate_data`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions, open an issue or contact `<atharvajoshi477@gmail.com>`.
