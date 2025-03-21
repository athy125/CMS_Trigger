// CMS L1 Trigger Simulation Framework

#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <memory>
#include <functional>

// Namespace to encompass all CMS L1 Trigger related classes
namespace CMSL1Trigger {

// Forward declarations
class Event;
class DetectorData;
class TriggerSystem;

// Simulated detector data components
enum DetectorComponent {
    ECAL,  // Electromagnetic Calorimeter
    HCAL,  // Hadronic Calorimeter
    MUON,  // Muon Chambers
    TRACKER // Silicon Tracker
};

// Structure to represent particle candidates found in the detector
struct ParticleCandidate {
    enum Type { ELECTRON, PHOTON, MUON, TAU, JET, MET };
    
    Type type;
    double pt;        // Transverse momentum
    double eta;       // Pseudorapidity
    double phi;       // Azimuthal angle
    double energy;
    bool isolated;    // Isolation flag for leptons
    
    ParticleCandidate(Type t, double p, double e, double ph, double en, bool iso = false)
        : type(t), pt(p), eta(e), phi(ph), energy(en), isolated(iso) {}
};

// Base class for detector subsystem data
class DetectorData {
protected:
    DetectorComponent component;
    std::vector<std::vector<double>> energyDeposits; // Simplified 2D array of energy deposits
    
public:
    DetectorData(DetectorComponent comp, int etaBins, int phiBins)
        : component(comp), 
          energyDeposits(etaBins, std::vector<double>(phiBins, 0.0)) {}
    
    virtual ~DetectorData() = default;
    
    DetectorComponent getComponent() const { return component; }
    
    void setEnergyDeposit(int etaBin, int phiBin, double energy) {
        if (etaBin >= 0 && etaBin < static_cast<int>(energyDeposits.size()) &&
            phiBin >= 0 && phiBin < static_cast<int>(energyDeposits[0].size())) {
            energyDeposits[etaBin][phiBin] = energy;
        }
    }
    
    double getEnergyDeposit(int etaBin, int phiBin) const {
        if (etaBin >= 0 && etaBin < static_cast<int>(energyDeposits.size()) &&
            phiBin >= 0 && phiBin < static_cast<int>(energyDeposits[0].size())) {
            return energyDeposits[etaBin][phiBin];
        }
        return 0.0;
    }
    
    int getEtaBins() const { return energyDeposits.size(); }
    int getPhiBins() const { return energyDeposits.empty() ? 0 : energyDeposits[0].size(); }
};

// Concrete detector subsystem data classes
class ECALData : public DetectorData {
public:
    ECALData() : DetectorData(ECAL, 72, 72) {} // Simplified ECAL granularity
};

class HCALData : public DetectorData {
public:
    HCALData() : DetectorData(HCAL, 72, 72) {} // Simplified HCAL granularity
};

class MuonData : public DetectorData {
public:
    MuonData() : DetectorData(MUON, 36, 36) {} // Simplified Muon granularity
};

class TrackerData : public DetectorData {
public:
    TrackerData() : DetectorData(TRACKER, 100, 180) {} // Simplified Tracker granularity
};

// Class representing a complete detector event
class Event {
private:
    uint64_t eventId;
    std::unique_ptr<ECALData> ecalData;
    std::unique_ptr<HCALData> hcalData;
    std::unique_ptr<MuonData> muonData;
    std::unique_ptr<TrackerData> trackerData;
    std::vector<ParticleCandidate> triggerCandidates;
    bool passed;
    
public:
    Event(uint64_t id) : eventId(id), 
                         ecalData(std::make_unique<ECALData>()),
                         hcalData(std::make_unique<HCALData>()),
                         muonData(std::make_unique<MuonData>()),
                         trackerData(std::make_unique<TrackerData>()),
                         passed(false) {}
    
    uint64_t getEventId() const { return eventId; }
    
    ECALData* getECALData() { return ecalData.get(); }
    HCALData* getHCALData() { return hcalData.get(); }
    MuonData* getMuonData() { return muonData.get(); }
    TrackerData* getTrackerData() { return trackerData.get(); }
    
    void addTriggerCandidate(const ParticleCandidate& candidate) {
        triggerCandidates.push_back(candidate);
    }
    
    const std::vector<ParticleCandidate>& getTriggerCandidates() const {
        return triggerCandidates;
    }
    
    void setPassedTrigger(bool pass) { passed = pass; }
    bool hasPassedTrigger() const { return passed; }
};

// Base class for trigger algorithm components
class TriggerAlgorithm {
protected:
    std::string name;
    double threshold;
    
public:
    TriggerAlgorithm(const std::string& n, double t) : name(n), threshold(t) {}
    virtual ~TriggerAlgorithm() = default;
    
    virtual bool evaluate(const Event& event) = 0;
    
    std::string getName() const { return name; }
    double getThreshold() const { return threshold; }
};

// Concrete trigger algorithms
class SingleMuonTrigger : public TriggerAlgorithm {
public:
    SingleMuonTrigger(double ptThreshold) 
        : TriggerAlgorithm("SingleMuon", ptThreshold) {}
    
    bool evaluate(const Event& event) override {
        for (const auto& candidate : event.getTriggerCandidates()) {
            if (candidate.type == ParticleCandidate::MUON && 
                candidate.pt > threshold) {
                return true;
            }
        }
        return false;
    }
};

class SingleElectronTrigger : public TriggerAlgorithm {
public:
    SingleElectronTrigger(double ptThreshold) 
        : TriggerAlgorithm("SingleElectron", ptThreshold) {}
    
    bool evaluate(const Event& event) override {
        for (const auto& candidate : event.getTriggerCandidates()) {
            if (candidate.type == ParticleCandidate::ELECTRON && 
                candidate.pt > threshold && candidate.isolated) {
                return true;
            }
        }
        return false;
    }
};

class DoubleMuonTrigger : public TriggerAlgorithm {
private:
    double secondThreshold;
    
public:
    DoubleMuonTrigger(double ptThreshold1, double ptThreshold2) 
        : TriggerAlgorithm("DoubleMuon", ptThreshold1), 
          secondThreshold(ptThreshold2) {}
    
    bool evaluate(const Event& event) override {
        int muonsAboveThreshold = 0;
        int muonsAboveSecondThreshold = 0;
        
        for (const auto& candidate : event.getTriggerCandidates()) {
            if (candidate.type == ParticleCandidate::MUON) {
                if (candidate.pt > threshold) {
                    muonsAboveThreshold++;
                }
                if (candidate.pt > secondThreshold) {
                    muonsAboveSecondThreshold++;
                }
            }
        }
        
        return (muonsAboveThreshold >= 1 && muonsAboveSecondThreshold >= 2);
    }
};

class JetTrigger : public TriggerAlgorithm {
private:
    int minJets;
    
public:
    JetTrigger(double ptThreshold, int numJets) 
        : TriggerAlgorithm("Jet", ptThreshold), minJets(numJets) {}
    
    bool evaluate(const Event& event) override {
        int jetsAboveThreshold = 0;
        
        for (const auto& candidate : event.getTriggerCandidates()) {
            if (candidate.type == ParticleCandidate::JET && 
                candidate.pt > threshold) {
                jetsAboveThreshold++;
            }
        }
        
        return (jetsAboveThreshold >= minJets);
    }
};

// Main Trigger system class
class TriggerSystem {
private:
    std::vector<std::unique_ptr<TriggerAlgorithm>> triggers;
    std::thread processingThread;
    std::mutex queueMutex;
    std::condition_variable queueCondition;
    std::queue<std::shared_ptr<Event>> eventQueue;
    std::atomic<bool> running;
    uint64_t eventsProcessed;
    uint64_t eventsPassed;
    
    // Method to be run in the processing thread
    void processingLoop() {
        while (running) {
            std::shared_ptr<Event> currentEvent = nullptr;
            
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCondition.wait(lock, [this] { 
                    return !eventQueue.empty() || !running; 
                });
                
                if (!running && eventQueue.empty()) {
                    break;
                }
                
                if (!eventQueue.empty()) {
                    currentEvent = eventQueue.front();
                    eventQueue.pop();
                }
            }
            
            if (currentEvent) {
                // Run trigger candidate reconstruction
                reconstructCandidates(*currentEvent);
                
                // Evaluate trigger algorithms
                bool passed = evaluateTriggers(*currentEvent);
                currentEvent->setPassedTrigger(passed);
                
                // Update statistics
                eventsProcessed++;
                if (passed) {
                    eventsPassed++;
                }
                
                // Print trigger decision (for demonstration)
                std::cout << "Event " << currentEvent->getEventId() 
                          << (passed ? " PASSED" : " REJECTED") << std::endl;
            }
        }
    }
    
    // Reconstruct trigger candidates from detector data
    void reconstructCandidates(Event& event) {
        // Simplified reconstruction algorithms
        
        // Muon reconstruction
        for (int etaBin = 0; etaBin < event.getMuonData()->getEtaBins(); etaBin++) {
            for (int phiBin = 0; phiBin < event.getMuonData()->getPhiBins(); phiBin++) {
                double energy = event.getMuonData()->getEnergyDeposit(etaBin, phiBin);
                if (energy > 5.0) {  // Energy threshold
                    // Convert eta/phi bin to actual values (simplified)
                    double eta = -3.0 + (6.0 * etaBin / event.getMuonData()->getEtaBins());
                    double phi = -3.14 + (6.28 * phiBin / event.getMuonData()->getPhiBins());
                    
                    // Simple transverse momentum calculation
                    double pt = energy * std::sin(2 * std::atan(std::exp(-eta)));
                    
                    // Add muon candidate
                    event.addTriggerCandidate(ParticleCandidate(
                        ParticleCandidate::MUON, pt, eta, phi, energy, true));
                }
            }
        }
        
        // Electron/Photon reconstruction (combines ECAL and Tracker)
        for (int etaBin = 0; etaBin < event.getECALData()->getEtaBins(); etaBin++) {
            for (int phiBin = 0; phiBin < event.getECALData()->getPhiBins(); phiBin++) {
                double energy = event.getECALData()->getEnergyDeposit(etaBin, phiBin);
                if (energy > 2.0) {  // Energy threshold
                    // Convert eta/phi bin to actual values
                    double eta = -3.0 + (6.0 * etaBin / event.getECALData()->getEtaBins());
                    double phi = -3.14 + (6.28 * phiBin / event.getECALData()->getPhiBins());
                    
                    // Simple transverse energy calculation
                    double pt = energy * std::sin(2 * std::atan(std::exp(-eta)));
                    
                    // Check if there's corresponding tracker activity (for electrons)
                    int trackerEtaBin = static_cast<int>((eta + 3.0) * event.getTrackerData()->getEtaBins() / 6.0);
                    int trackerPhiBin = static_cast<int>((phi + 3.14) * event.getTrackerData()->getPhiBins() / 6.28);
                    
                    bool hasTrack = false;
                    if (trackerEtaBin >= 0 && trackerEtaBin < event.getTrackerData()->getEtaBins() &&
                        trackerPhiBin >= 0 && trackerPhiBin < event.getTrackerData()->getPhiBins()) {
                        hasTrack = event.getTrackerData()->getEnergyDeposit(trackerEtaBin, trackerPhiBin) > 0.5;
                    }
                    
                    // Add electron or photon candidate
                    if (hasTrack) {
                        // Calculate isolation (simplified)
                        bool isolated = true;
                        for (int dEta = -1; dEta <= 1; dEta++) {
                            for (int dPhi = -1; dPhi <= 1; dPhi++) {
                                if (dEta == 0 && dPhi == 0) continue;
                                
                                int neighborEta = etaBin + dEta;
                                int neighborPhi = phiBin + dPhi;
                                
                                if (neighborEta >= 0 && neighborEta < event.getECALData()->getEtaBins() &&
                                    neighborPhi >= 0 && neighborPhi < event.getECALData()->getPhiBins()) {
                                    if (event.getECALData()->getEnergyDeposit(neighborEta, neighborPhi) > 1.0) {
                                        isolated = false;
                                        break;
                                    }
                                }
                            }
                            if (!isolated) break;
                        }
                        
                        event.addTriggerCandidate(ParticleCandidate(
                            ParticleCandidate::ELECTRON, pt, eta, phi, energy, isolated));
                    } else {
                        event.addTriggerCandidate(ParticleCandidate(
                            ParticleCandidate::PHOTON, pt, eta, phi, energy));
                    }
                }
            }
        }
        
        // Jet reconstruction (using ECAL + HCAL)
        std::vector<std::vector<double>> totalEnergy(
            event.getHCALData()->getEtaBins(),
            std::vector<double>(event.getHCALData()->getPhiBins(), 0.0));
        
        // Sum ECAL and HCAL energies
        for (int etaBin = 0; etaBin < event.getHCALData()->getEtaBins(); etaBin++) {
            for (int phiBin = 0; phiBin < event.getHCALData()->getPhiBins(); phiBin++) {
                // Get HCAL energy
                double hcalEnergy = event.getHCALData()->getEnergyDeposit(etaBin, phiBin);
                
                // Map to corresponding ECAL bins (simplified)
                int ecalEtaBin = etaBin * event.getECALData()->getEtaBins() / event.getHCALData()->getEtaBins();
                int ecalPhiBin = phiBin * event.getECALData()->getPhiBins() / event.getHCALData()->getPhiBins();
                
                double ecalEnergy = 0.0;
                if (ecalEtaBin >= 0 && ecalEtaBin < event.getECALData()->getEtaBins() &&
                    ecalPhiBin >= 0 && ecalPhiBin < event.getECALData()->getPhiBins()) {
                    ecalEnergy = event.getECALData()->getEnergyDeposit(ecalEtaBin, ecalPhiBin);
                }
                
                totalEnergy[etaBin][phiBin] = hcalEnergy + ecalEnergy;
            }
        }
        
        // Simple jet finding algorithm
        for (int etaBin = 0; etaBin < static_cast<int>(totalEnergy.size()); etaBin++) {
            for (int phiBin = 0; phiBin < static_cast<int>(totalEnergy[0].size()); phiBin++) {
                if (totalEnergy[etaBin][phiBin] > 10.0) {  // Energy threshold for jet seed
                    bool isLocalMaximum = true;
                    
                    // Check if it's a local maximum
                    for (int dEta = -1; dEta <= 1 && isLocalMaximum; dEta++) {
                        for (int dPhi = -1; dPhi <= 1; dPhi++) {
                            if (dEta == 0 && dPhi == 0) continue;
                            
                            int neighborEta = etaBin + dEta;
                            int neighborPhi = phiBin + dPhi;
                            
                            // Check boundary
                            if (neighborEta >= 0 && neighborEta < static_cast<int>(totalEnergy.size()) &&
                                neighborPhi >= 0 && neighborPhi < static_cast<int>(totalEnergy[0].size())) {
                                if (totalEnergy[neighborEta][neighborPhi] > totalEnergy[etaBin][phiBin]) {
                                    isLocalMaximum = false;
                                    break;
                                }
                            }
                        }
                    }
                    
                    if (isLocalMaximum) {
                        // Calculate jet energy (including surrounding cells)
                        double jetEnergy = 0.0;
                        for (int dEta = -2; dEta <= 2; dEta++) {
                            for (int dPhi = -2; dPhi <= 2; dPhi++) {
                                int jetEtaBin = etaBin + dEta;
                                int jetPhiBin = phiBin + dPhi;
                                
                                if (jetEtaBin >= 0 && jetEtaBin < static_cast<int>(totalEnergy.size()) &&
                                    jetPhiBin >= 0 && jetPhiBin < static_cast<int>(totalEnergy[0].size())) {
                                    jetEnergy += totalEnergy[jetEtaBin][jetPhiBin];
                                }
                            }
                        }
                        
                        // Convert eta/phi bin to actual values
                        double eta = -3.0 + (6.0 * etaBin / totalEnergy.size());
                        double phi = -3.14 + (6.28 * phiBin / totalEnergy[0].size());
                        
                        // Simple transverse energy calculation
                        double pt = jetEnergy * std::sin(2 * std::atan(std::exp(-eta)));
                        
                        // Add jet candidate
                        event.addTriggerCandidate(ParticleCandidate(
                            ParticleCandidate::JET, pt, eta, phi, jetEnergy));
                    }
                }
            }
        }
    }
    
    // Evaluate all trigger algorithms
    bool evaluateTriggers(const Event& event) {
        for (const auto& trigger : triggers) {
            if (trigger->evaluate(event)) {
                std::cout << "  Passed trigger: " << trigger->getName() << std::endl;
                return true;
            }
        }
        return false;
    }
    
public:
    TriggerSystem() : running(false), eventsProcessed(0), eventsPassed(0) {
        // Add default triggers
        triggers.push_back(std::make_unique<SingleMuonTrigger>(20.0));
        triggers.push_back(std::make_unique<SingleElectronTrigger>(27.0));
        triggers.push_back(std::make_unique<DoubleMuonTrigger>(17.0, 8.0));
        triggers.push_back(std::make_unique<JetTrigger>(40.0, 4));
    }
    
    ~TriggerSystem() {
        stop();
    }
    
    void start() {
        if (!running) {
            running = true;
            processingThread = std::thread(&TriggerSystem::processingLoop, this);
            std::cout << "Trigger system started" << std::endl;
        }
    }
    
    void stop() {
        if (running) {
            running = false;
            queueCondition.notify_all();
            if (processingThread.joinable()) {
                processingThread.join();
            }
            std::cout << "Trigger system stopped" << std::endl;
        }
    }
    
    void addTrigger(std::unique_ptr<TriggerAlgorithm> trigger) {
        triggers.push_back(std::move(trigger));
    }
    
    void queueEvent(std::shared_ptr<Event> event) {
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            eventQueue.push(event);
        }
        queueCondition.notify_one();
    }
    
    int getQueueSize() const {
        std::lock_guard<std::mutex> lock(queueMutex);
        return eventQueue.size();
    }
    
    uint64_t getEventsProcessed() const { return eventsProcessed; }
    uint64_t getEventsPassed() const { return eventsPassed; }
    double getAcceptanceRate() const { 
        return eventsProcessed > 0 ? static_cast<double>(eventsPassed) / eventsProcessed : 0.0;
    }
};

// Event Generator class to simulate detector data
class EventGenerator {
private:
    std::mt19937 rng;
    uint64_t nextEventId;
    
    // Different physics processes to simulate
    enum PhysicsProcess {
        MINIMUM_BIAS,
        W_PRODUCTION,
        Z_PRODUCTION,
        TTBAR_PRODUCTION,
        HIGGS_PRODUCTION
    };
    
    // Simulate specific physics processes
    void simulatePhysicsProcess(Event& event, PhysicsProcess process) {
        switch (process) {
            case MINIMUM_BIAS: {
                // Simulate low-energy QCD interactions (noise)
                std::uniform_int_distribution<int> numDeposits(10, 30);
                int deposits = numDeposits(rng);
                
                for (int i = 0; i < deposits; i++) {
                    fillRandomEnergyDeposit(event, 0.1, 5.0);
                }
                break;
            }
            
            case W_PRODUCTION: {
                // Simulate W->eν or W->μν
                std::bernoulli_distribution electronMode(0.5);
                bool isElectron = electronMode(rng);
                
                if (isElectron) {
                    // W->eν: High-energy electron and missing ET
                    simulateElectron(event, 30.0, 60.0);
                } else {
                    // W->μν: High-energy muon and missing ET
                    simulateMuon(event, 30.0, 60.0);
                }
                break;
            }
            
            case Z_PRODUCTION: {
                // Simulate Z->ee or Z->μμ
                std::bernoulli_distribution electronMode(0.5);
                bool isElectron = electronMode(rng);
                
                if (isElectron) {
                    // Z->ee: Two high-energy electrons
                    simulateElectron(event, 25.0, 50.0);
                    simulateElectron(event, 25.0, 50.0);
                } else {
                    // Z->μμ: Two high-energy muons
                    simulateMuon(event, 25.0, 50.0);
                    simulateMuon(event, 25.0, 50.0);
                }
                break;
            }
            
            case TTBAR_PRODUCTION: {
                // Simulate ttbar: multiple jets, possibly leptons
                int numJets = 4 + (std::uniform_int_distribution<int>(0, 2))(rng);
                
                for (int i = 0; i < numJets; i++) {
                    simulateJet(event, 20.0, 100.0);
                }
                
                // Possibly add a lepton
                std::bernoulli_distribution hasLepton(0.4);
                if (hasLepton(rng)) {
                    std::bernoulli_distribution isElectron(0.5);
                    if (isElectron(rng)) {
                        simulateElectron(event, 25.0, 50.0);
                    } else {
                        simulateMuon(event, 25.0, 50.0);
                    }
                }
                break;
            }
            
            case HIGGS_PRODUCTION: {
                // Simulate H->γγ or H->ZZ->4l
                std::bernoulli_distribution diphotonMode(0.3);
                
                if (diphotonMode(rng)) {
                    // H->γγ: Two high-energy photons
                    simulatePhoton(event, 40.0, 70.0);
                    simulatePhoton(event, 40.0, 70.0);
                } else {
                    // H->ZZ->4l: Four leptons (e or μ)
                    for (int i = 0; i < 4; i++) {
                        std::bernoulli_distribution isElectron(0.5);
                        if (isElectron(rng)) {
                            simulateElectron(event, 15.0, 40.0);
                        } else {
                            simulateMuon(event, 15.0, 40.0);
                        }
                    }
                }
                break;
            }
        }
        
        // Add some background noise to all events
        std::uniform_int_distribution<int> numNoiseDeposits(5, 15);
        int deposits = numNoiseDeposits(rng);
        
        for (int i = 0; i < deposits; i++) {
            fillRandomEnergyDeposit(event, 0.1, 3.0);
        }
    }
    
    void fillRandomEnergyDeposit(Event& event, double minEnergy, double maxEnergy) {
        std::uniform_real_distribution<double> energyDist(minEnergy, maxEnergy);
        std::uniform_int_distribution<int> detectorDist(0, 3);
        
        double energy = energyDist(rng);
        int detector = detectorDist(rng);
        
        DetectorData* data = nullptr;
        
        switch (detector) {
            case 0: data = event.getECALData(); break;
            case 1: data = event.getHCALData(); break;
            case 2: data = event.getMuonData(); break;
            case 3: data = event.getTrackerData(); break;
        }
        
        if (data) {
            std::uniform_int_distribution<int> etaDist(0, data->getEtaBins() - 1);
            std::uniform_int_distribution<int> phiDist(0, data->getPhiBins() - 1);
            
            int etaBin = etaDist(rng);
            int phiBin = phiDist(rng);
            
            data->setEnergyDeposit(etaBin, phiBin, energy);
        }
    }
    
    void simulateElectron(Event& event, double minEnergy, double maxEnergy) {
        std::uniform_real_distribution<double> energyDist(minEnergy, maxEnergy);
        std::uniform_real_distribution<double> etaDist(-2.5, 2.5);
        std::uniform_real_distribution<double> phiDist(-3.14, 3.14);
        
        double energy = energyDist(rng);
        double eta = etaDist(rng);
        double phi = phiDist(rng);
        
        // Determine bin indices for ECAL and Tracker
        int ecalEtaBin = static_cast<int>((eta + 3.0) * event.getECALData()->getEtaBins() / 6.0);
        int ecalPhiBin = static_cast<int>((phi + 3.14) * event.getECALData()->getPhiBins() / 6.28);
        
        int trackerEtaBin = static_cast<int>((eta + 3.0) * event.getTrackerData()->getEtaBins() / 6.0);
        int trackerPhiBin = static_cast<int>((phi + 3.14) * event.getTrackerData()->getPhiBins() / 6.28);
        
        // Set energy deposits
        if (ecalEtaBin >= 0 && ecalEtaBin < event.getECALData()->getEtaBins() &&
            ecalPhiBin >= 0 && ecalPhiBin < event.getECALData()->getPhiBins()) {
            event.getECALData()->setEnergyDeposit(ecalEtaBin, ecalPhiBin, energy);
        }
        
        if (trackerEtaBin >= 0 && trackerEtaBin < event.getTrackerData()->getEtaBins() &&
            trackerPhiBin >= 0 && trackerPhiBin < event.getTrackerData()->getPhiBins()) {
            event.getTrackerData()->setEnergyDeposit(trackerEtaBin, trackerPhiBin, energy * 0.1);
        }
    }
    
void simulatePhoton(Event& event, double minEnergy, double maxEnergy) {
        std::uniform_real_distribution<double> energyDist(minEnergy, maxEnergy);
        std::uniform_real_distribution<double> etaDist(-2.5, 2.5);
        std::uniform_real_distribution<double> phiDist(-3.14, 3.14);
        
        double energy = energyDist(rng);
        double eta = etaDist(rng);
        double phi = phiDist(rng);
        
        // Determine bin indices for ECAL
        int ecalEtaBin = static_cast<int>((eta + 3.0) * event.getECALData()->getEtaBins() / 6.0);
        int ecalPhiBin = static_cast<int>((phi + 3.14) * event.getECALData()->getPhiBins() / 6.28);
        
        // Set energy deposits (photons leave energy only in ECAL, not in tracker)
        if (ecalEtaBin >= 0 && ecalEtaBin < event.getECALData()->getEtaBins() &&
            ecalPhiBin >= 0 && ecalPhiBin < event.getECALData()->getPhiBins()) {
            event.getECALData()->setEnergyDeposit(ecalEtaBin, ecalPhiBin, energy);
        }
    }
    
    void simulateMuon(Event& event, double minEnergy, double maxEnergy) {
        std::uniform_real_distribution<double> energyDist(minEnergy, maxEnergy);
        std::uniform_real_distribution<double> etaDist(-2.4, 2.4);
        std::uniform_real_distribution<double> phiDist(-3.14, 3.14);
        
        double energy = energyDist(rng);
        double eta = etaDist(rng);
        double phi = phiDist(rng);
        
        // Determine bin indices for Muon system and Tracker
        int muonEtaBin = static_cast<int>((eta + 3.0) * event.getMuonData()->getEtaBins() / 6.0);
        int muonPhiBin = static_cast<int>((phi + 3.14) * event.getMuonData()->getPhiBins() / 6.28);
        
        int trackerEtaBin = static_cast<int>((eta + 3.0) * event.getTrackerData()->getEtaBins() / 6.0);
        int trackerPhiBin = static_cast<int>((phi + 3.14) * event.getTrackerData()->getPhiBins() / 6.28);
        
        // Set energy deposits
        if (muonEtaBin >= 0 && muonEtaBin < event.getMuonData()->getEtaBins() &&
            muonPhiBin >= 0 && muonPhiBin < event.getMuonData()->getPhiBins()) {
            event.getMuonData()->setEnergyDeposit(muonEtaBin, muonPhiBin, energy);
        }
        
        if (trackerEtaBin >= 0 && trackerEtaBin < event.getTrackerData()->getEtaBins() &&
            trackerPhiBin >= 0 && trackerPhiBin < event.getTrackerData()->getPhiBins()) {
            event.getTrackerData()->setEnergyDeposit(trackerEtaBin, trackerPhiBin, energy * 0.1);
        }
    }
    
    void simulateJet(Event& event, double minEnergy, double maxEnergy) {
        std::uniform_real_distribution<double> energyDist(minEnergy, maxEnergy);
        std::uniform_real_distribution<double> etaDist(-4.7, 4.7);
        std::uniform_real_distribution<double> phiDist(-3.14, 3.14);
        
        double energy = energyDist(rng);
        double eta = etaDist(rng);
        double phi = phiDist(rng);
        
        // Jets deposit energy in both ECAL and HCAL in a cone
        simulateJetCone(event, eta, phi, energy);
    }
    
    void simulateJetCone(Event& event, double eta, double phi, double energy) {
        // Determine core bin indices for HCAL
        int hcalEtaBin = static_cast<int>((eta + 5.0) * event.getHCALData()->getEtaBins() / 10.0);
        int hcalPhiBin = static_cast<int>((phi + 3.14) * event.getHCALData()->getPhiBins() / 6.28);
        
        // Radius of the jet cone
        int coneRadius = 2;
        
        // Distribute energy in a cone (70% in HCAL, 30% in ECAL)
        double hcalFraction = 0.7;
        double ecalFraction = 0.3;
        
        // Set energy deposits in a cone
        for (int dEta = -coneRadius; dEta <= coneRadius; dEta++) {
            for (int dPhi = -coneRadius; dPhi <= coneRadius; dPhi++) {
                // Calculate distance from center
                double distance = std::sqrt(dEta * dEta + dPhi * dPhi);
                if (distance > coneRadius) continue;
                
                // Energy decreases with distance from center
                double fraction = (coneRadius - distance) / coneRadius;
                
                // Set HCAL energy
                int currentHcalEtaBin = hcalEtaBin + dEta;
                int currentHcalPhiBin = hcalPhiBin + dPhi;
                
                // Account for phi wrapping
                if (currentHcalPhiBin < 0) {
                    currentHcalPhiBin += event.getHCALData()->getPhiBins();
                } else if (currentHcalPhiBin >= event.getHCALData()->getPhiBins()) {
                    currentHcalPhiBin -= event.getHCALData()->getPhiBins();
                }
                
                if (currentHcalEtaBin >= 0 && currentHcalEtaBin < event.getHCALData()->getEtaBins() &&
                    currentHcalPhiBin >= 0 && currentHcalPhiBin < event.getHCALData()->getPhiBins()) {
                    double hcalEnergy = energy * hcalFraction * fraction;
                    event.getHCALData()->setEnergyDeposit(currentHcalEtaBin, currentHcalPhiBin, hcalEnergy);
                }
                
                // Set ECAL energy (scaled by ECAL/HCAL bin ratio)
                int ecalEtaBin = static_cast<int>(currentHcalEtaBin * 
                                                event.getECALData()->getEtaBins() / 
                                                event.getHCALData()->getEtaBins());
                int ecalPhiBin = static_cast<int>(currentHcalPhiBin * 
                                                event.getECALData()->getPhiBins() / 
                                                event.getHCALData()->getPhiBins());
                
                if (ecalEtaBin >= 0 && ecalEtaBin < event.getECALData()->getEtaBins() &&
                    ecalPhiBin >= 0 && ecalPhiBin < event.getECALData()->getPhiBins()) {
                    double ecalEnergy = energy * ecalFraction * fraction;
                    event.getECALData()->setEnergyDeposit(ecalEtaBin, ecalPhiBin, ecalEnergy);
                }
            }
        }
    }
    
public:
    EventGenerator() : rng(std::random_device{}()), nextEventId(0) {}
    
    std::shared_ptr<Event> generateEvent() {
        auto event = std::make_shared<Event>(nextEventId++);
        
        // Choose physics process to simulate
        std::discrete_distribution<int> processDist({70, 10, 10, 9, 1});
        PhysicsProcess process = static_cast<PhysicsProcess>(processDist(rng));
        
        // Simulate the chosen physics process
        simulatePhysicsProcess(*event, process);
        
        return event;
    }
};

} // namespace CMSL1Trigger

// Main function to run the simulation
int main() {
    using namespace CMSL1Trigger;
    
    // Create event generator and trigger system
    EventGenerator generator;
    TriggerSystem triggerSystem;
    
    // Start trigger system
    triggerSystem.start();
    
    // Number of events to simulate
    const int numEvents = 1000;
    
    // Generate and process events
    for (int i = 0; i < numEvents; i++) {
        auto event = generator.generateEvent();
        triggerSystem.queueEvent(event);
        
        // Show progress
        if (i % 100 == 0) {
            std::cout << "Generated " << i << " events, queue size: " 
                      << triggerSystem.getQueueSize() << std::endl;
        }
        
        // Small delay to simulate realistic data rate
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Wait for all events to be processed
    while (triggerSystem.getQueueSize() > 0) {
        std::cout << "Waiting for " << triggerSystem.getQueueSize() 
                  << " events to be processed..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // Allow final events to be processed
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Stop trigger system
    triggerSystem.stop();
    
    // Print statistics
    std::cout << "Simulation complete!" << std::endl;
    std::cout << "Events processed: " << triggerSystem.getEventsProcessed() << std::endl;
    std::cout << "Events passed: " << triggerSystem.getEventsPassed() << std::endl;
    std::cout << "Acceptance rate: " << (triggerSystem.getAcceptanceRate() * 100.0) 
              << "%" << std::endl;
    
    return 0;
}

