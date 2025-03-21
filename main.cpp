#include <iostream>
#include <vector>
#include <queue>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <memory>
#include <cmath>
#include <algorithm>

// Namespace for CMS L1 Trigger simulation
namespace CMSL1Sim {

enum DetectorType {
    ECAL,    // Electromagnetic Calorimeter
    HCAL,    // Hadronic Calorimeter
    MUON,    // Muon System
    TRACKER  // Silicon Tracker
};

// Particle types for trigger candidates
enum ParticleType {
    ELECTRON,
    PHOTON,
    MUON,
    JET,
    TAU,
    MET  // Missing Transverse Energy
};

// Structure for trigger candidates
struct TriggerCandidate {
    ParticleType type;
    double pt;        // Transverse momentum (GeV)
    double eta;       // Pseudorapidity
    double phi;       // Azimuthal angle (radians)
    double energy;    // Total energy (GeV)
    bool isolated;    // Isolation flag
    
    TriggerCandidate(ParticleType t, double p, double e, double ph, double en, bool iso = false)
        : type(t), pt(p), eta(e), phi(ph), energy(en), isolated(iso) {}
};

// Base class for detector data
class DetectorLayer {
protected:
    DetectorType type;
    std::vector<std::vector<double>> energyMap;  // 2D grid of energy deposits
    int etaBins, phiBins;
    
public:
    DetectorLayer(DetectorType t, int eBins, int pBins)
        : type(t), etaBins(eBins), phiBins(pBins),
          energyMap(eBins, std::vector<double>(pBins, 0.0)) {}
    
    virtual ~DetectorLayer() = default;
    
    void depositEnergy(int etaIdx, int phiIdx, double energy) {
        if (etaIdx >= 0 && etaIdx < etaBins && phiIdx >= 0 && phiIdx < phiBins) {
            energyMap[etaIdx][phiIdx] += energy;
        }
    }
    
    double getEnergy(int etaIdx, int phiIdx) const {
        if (etaIdx >= 0 && etaIdx < etaBins && phiIdx >= 0 && phiIdx < phiBins) {
            return energyMap[etaIdx][phiIdx];
        }
        return 0.0;
    }
    
    void clear() {
        for (auto& row : energyMap) {
            std::fill(row.begin(), row.end(), 0.0);
        }
    }
    
    int getEtaBins() const { return etaBins; }
    int getPhiBins() const { return phiBins; }
    DetectorType getType() const { return type; }
};

// Specific detector implementations
class ECALLayer : public DetectorLayer {
public:
    ECALLayer() : DetectorLayer(ECAL, 140, 360) {}  // Fine granularity
};

class HCALLayer : public DetectorLayer {
public:
    HCALLayer() : DetectorLayer(HCAL, 82, 72) {}    // Coarser granularity
};

class MuonLayer : public DetectorLayer {
public:
    MuonLayer() : DetectorLayer(MUON, 48, 48) {}    // Simplified muon system
};

class TrackerLayer : public DetectorLayer {
public:
    TrackerLayer() : DetectorLayer(TRACKER, 200, 360) {}  // High resolution
};

// Event class representing a collision
class CollisionEvent {
private:
    uint64_t eventId;
    std::unique_ptr<ECALLayer> ecal;
    std::unique_ptr<HCALLayer> hcal;
    std::unique_ptr<MuonLayer> muon;
    std::unique_ptr<TrackerLayer> tracker;
    std::vector<TriggerCandidate> candidates;
    bool triggerDecision;
    std::chrono::nanoseconds timestamp;
    
public:
    CollisionEvent(uint64_t id)
        : eventId(id), ecal(std::make_unique<ECALLayer>()),
          hcal(std::make_unique<HCALLayer>()), muon(std::make_unique<MuonLayer>()),
          tracker(std::make_unique<TrackerLayer>()), triggerDecision(false),
          timestamp(std::chrono::high_resolution_clock::now().time_since_epoch()) {}
    
    uint64_t getId() const { return eventId; }
    ECALLayer* getECAL() { return ecal.get(); }
    HCALLayer* getHCAL() { return hcal.get(); }
    MuonLayer* getMuon() { return muon.get(); }
    TrackerLayer* getTracker() { return tracker.get(); }
    
    void addCandidate(const TriggerCandidate& cand) { candidates.push_back(cand); }
    const std::vector<TriggerCandidate>& getCandidates() const { return candidates; }
    
    void setTriggerDecision(bool decision) { triggerDecision = decision; }
    bool passedTrigger() const { return triggerDecision; }
    
    std::chrono::nanoseconds getTimestamp() const { return timestamp; }
};

// Base class for trigger algorithms
class TriggerLogic {
protected:
    std::string name;
    double ptThreshold;
    
public:
    TriggerLogic(const std::string& n, double pt) : name(n), ptThreshold(pt) {}
    virtual ~TriggerLogic() = default;
    virtual bool evaluate(const CollisionEvent& event) = 0;
    std::string getName() const { return name; }
};

// Specific trigger implementations
class SingleMuonLogic : public TriggerLogic {
public:
    SingleMuonLogic(double ptThresh) : TriggerLogic("SingleMuon", ptThresh) {}
    
    bool evaluate(const CollisionEvent& event) override {
        for (const auto& cand : event.getCandidates()) {
            if (cand.type == MUON && cand.pt > ptThreshold && cand.isolated) {
                return true;
            }
        }
        return false;
    }
};

class DiJetLogic : public TriggerLogic {
private:
    int minJets;
    
public:
    DiJetLogic(double ptThresh, int minJ) : TriggerLogic("DiJet", ptThresh), minJets(minJ) {}
    
    bool evaluate(const CollisionEvent& event) override {
        int jetCount = 0;
        for (const auto& cand : event.getCandidates()) {
            if (cand.type == JET && cand.pt > ptThreshold) {
                jetCount++;
            }
        }
        return jetCount >= minJets;
    }
};

class ElectronPhotonLogic : public TriggerLogic {
public:
    ElectronPhotonLogic(double ptThresh) : TriggerLogic("ElectronPhoton", ptThresh) {}
    
    bool evaluate(const CollisionEvent& event) override {
        for (const auto& cand : event.getCandidates()) {
            if ((cand.type == ELECTRON || cand.type == PHOTON) && 
                cand.pt > ptThreshold && cand.isolated) {
                return true;
            }
        }
        return false;
    }
};

// Main L1 Trigger System
class L1TriggerSystem {
private:
    std::vector<std::unique_ptr<TriggerLogic>> triggerMenu;
    std::queue<std::shared_ptr<CollisionEvent>> eventQueue;
    std::mutex queueMutex;
    std::condition_variable queueCond;
    std::thread processorThread;
    std::atomic<bool> isRunning;
    uint64_t processedEvents;
    uint64_t passedEvents;
    
    void processEvents() {
        while (isRunning) {
            std::shared_ptr<CollisionEvent> event;
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                queueCond.wait(lock, [this] { return !eventQueue.empty() || !isRunning; });
                
                if (!isRunning && eventQueue.empty()) break;
                if (eventQueue.empty()) continue;
                
                event = eventQueue.front();
                eventQueue.pop();
            }
            
            reconstructCandidates(*event);
            bool passed = evaluateTriggers(*event);
            event->setTriggerDecision(passed);
            
            processedEvents++;
            if (passed) passedEvents++;
            
            std::cout << "Event " << event->getId() << (passed ? " PASSED" : " REJECTED") 
                      << " | Candidates: " << event->getCandidates().size() << std::endl;
        }
    }
    
    void reconstructCandidates(CollisionEvent& event) {
        // Muon reconstruction
        auto* muonLayer = event.getMuon();
        for (int eta = 0; eta < muonLayer->getEtaBins(); eta++) {
            for (int phi = 0; phi < muonLayer->getPhiBins(); phi++) {
                double energy = muonLayer->getEnergy(eta, phi);
                if (energy > 5.0) {  // Threshold for muon hit
                    double etaVal = -2.4 + (4.8 * eta) / muonLayer->getEtaBins();
                    double phiVal = -M_PI + (2 * M_PI * phi) / muonLayer->getPhiBins();
                    double pt = energy * std::sin(2 * std::atan(std::exp(-etaVal)));
                    event.addCandidate(TriggerCandidate(MUON, pt, etaVal, phiVal, energy, true));
                }
            }
        }
        
        // Jet reconstruction (simple clustering)
        auto* hcal = event.getHCAL();
        std::vector<std::vector<double>> jetEnergy(hcal->getEtaBins(), 
                                                  std::vector<double>(hcal->getPhiBins(), 0.0));
        
        for (int eta = 0; eta < hcal->getEtaBins(); eta++) {
            for (int phi = 0; phi < hcal->getPhiBins(); phi++) {
                jetEnergy[eta][phi] = hcal->getEnergy(eta, phi) + 
                                     event.getECAL()->getEnergy(eta * 2, phi * 5);  // Simplified mapping
            }
        }
        
        for (int eta = 0; eta < hcal->getEtaBins(); eta++) {
            for (int phi = 0; phi < hcal->getPhiBins(); phi++) {
                if (jetEnergy[eta][phi] > 15.0) {
                    bool isMax = true;
                    for (int dEta = -1; dEta <= 1 && isMax; dEta++) {
                        for (int dPhi = -1; dPhi <= 1; dPhi++) {
                            if (dEta == 0 && dPhi == 0) continue;
                            int nEta = eta + dEta;
                            int nPhi = phi + dPhi;
                            if (nEta >= 0 && nEta < hcal->getEtaBins() && 
                                nPhi >= 0 && nPhi < hcal->getPhiBins()) {
                                if (jetEnergy[nEta][nPhi] > jetEnergy[eta][phi]) {
                                    isMax = false;
                                }
                            }
                        }
                    }
                    if (isMax) {
                        double etaVal = -4.7 + (9.4 * eta) / hcal->getEtaBins();
                        double phiVal = -M_PI + (2 * M_PI * phi) / hcal->getPhiBins();
                        double pt = jetEnergy[eta][phi] * std::sin(2 * std::atan(std::exp(-etaVal)));
                        event.addCandidate(TriggerCandidate(JET, pt, etaVal, phiVal, jetEnergy[eta][phi]));
                    }
                }
            }
        }
        
        // Electron/Photon reconstruction
        auto* ecal = event.getECAL();
        for (int eta = 0; eta < ecal->getEtaBins(); eta++) {
            for (int phi = 0; phi < ecal->getPhiBins(); phi++) {
                double energy = ecal->getEnergy(eta, phi);
                if (energy > 3.0) {
                    double etaVal = -2.5 + (5.0 * eta) / ecal->getEtaBins();
                    double phiVal = -M_PI + (2 * M_PI * phi) / ecal->getPhiBins();
                    double pt = energy * std::sin(2 * std::atan(std::exp(-etaVal)));
                    
                    int trackEta = eta * 2;  // Simplified mapping
                    int trackPhi = phi;
                    bool hasTrack = event.getTracker()->getEnergy(trackEta, trackPhi) > 0.5;
                    bool isolated = true;
                    for (int dEta = -1; dEta <= 1; dEta++) {
                        for (int dPhi = -1; dPhi <= 1; dPhi++) {
                            if (dEta == 0 && dPhi == 0) continue;
                            int nEta = eta + dEta;
                            int nPhi = phi + dPhi;
                            if (nEta >= 0 && nEta < ecal->getEtaBins() && 
                                nPhi >= 0 && nPhi < ecal->getPhiBins()) {
                                if (ecal->getEnergy(nEta, nPhi) > 1.0) isolated = false;
                            }
                        }
                    }
                    
                    if (hasTrack) {
                        event.addCandidate(TriggerCandidate(ELECTRON, pt, etaVal, phiVal, energy, isolated));
                    } else {
                        event.addCandidate(TriggerCandidate(PHOTON, pt, etaVal, phiVal, energy, isolated));
                    }
                }
            }
        }
    }
    
    bool evaluateTriggers(const CollisionEvent& event) {
        for (const auto& trigger : triggerMenu) {
            if (trigger->evaluate(event)) {
                std::cout << "  Trigger fired: " << trigger->getName() << std::endl;
                return true;
            }
        }
        return false;
    }
    
public:
    L1TriggerSystem() : isRunning(false), processedEvents(0), passedEvents(0) {
        triggerMenu.push_back(std::make_unique<SingleMuonLogic>(25.0));
        triggerMenu.push_back(std::make_unique<DiJetLogic>(50.0, 2));
        triggerMenu.push_back(std::make_unique<ElectronPhotonLogic>(30.0));
    }
    
    ~L1TriggerSystem() { stop(); }
    
    void start() {
        if (!isRunning) {
            isRunning = true;
            processorThread = std::thread(&L1TriggerSystem::processEvents, this);
            std::cout << "L1 Trigger System started\n";
        }
    }
    
    void stop() {
        if (isRunning) {
            isRunning = false;
            queueCond.notify_all();
            if (processorThread.joinable()) processorThread.join();
            std::cout << "L1 Trigger System stopped\n";
        }
    }
    
    void queueEvent(std::shared_ptr<CollisionEvent> event) {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventQueue.push(event);
        queueCond.notify_one();
    }
    
    uint64_t getProcessedEvents() const { return processedEvents; }
    uint64_t getPassedEvents() const { return passedEvents; }
    double getEfficiency() const { 
        return processedEvents > 0 ? static_cast<double>(passedEvents) / processedEvents : 0.0;
    }
};

// Event generator simulating physics processes
class EventGenerator {
private:
    std::mt19937 rng;
    uint64_t eventCounter;
    
    enum ProcessType {
        QCD,        // Minimum bias
        SINGLE_LEP, // W production
        DILEP,      // Z production
        MULTIJET    // ttbar-like
    };
    
    void generateQCD(CollisionEvent& event) {
        std::uniform_int_distribution<int> nHits(20, 50);
        int hits = nHits(rng);
        for (int i = 0; i < hits; i++) {
            depositRandomEnergy(event, 0.5, 10.0);
        }
    }
    
    void generateSingleLepton(CollisionEvent& event) {
        std::bernoulli_distribution lepType(0.5);
        double energy = std::uniform_real_distribution<double>(30.0, 80.0)(rng);
        double eta = std::uniform_real_distribution<double>(-2.4, 2.4)(rng);
        double phi = std::uniform_real_distribution<double>(-M_PI, M_PI)(rng);
        
        int etaBin, phiBin;
        if (lepType(rng)) {  // Muon
            etaBin = static_cast<int>((eta + 2.4) * event.getMuon()->getEtaBins() / 4.8);
            phiBin = static_cast<int>((phi + M_PI) * event.getMuon()->getPhiBins() / (2 * M_PI));
            event.getMuon()->depositEnergy(etaBin, phiBin, energy);
            etaBin = static_cast<int>((eta + 2.5) * event.getTracker()->getEtaBins() / 5.0);
            phiBin = static_cast<int>((phi + M_PI) * event.getTracker()->getPhiBins() / (2 * M_PI));
            event.getTracker()->depositEnergy(etaBin, phiBin, energy * 0.1);
        } else {  // Electron
            etaBin = static_cast<int>((eta + 2.5) * event.getECAL()->getEtaBins() / 5.0);
            phiBin = static_cast<int>((phi + M_PI) * event.getECAL()->getPhiBins() / (2 * M_PI));
            event.getECAL()->depositEnergy(etaBin, phiBin, energy);
            etaBin = static_cast<int>((eta + 2.5) * event.getTracker()->getEtaBins() / 5.0);
            event.getTracker()->depositEnergy(etaBin, phiBin, energy * 0.1);
        }
        depositRandomEnergy(event, 0.5, 5.0, 10);  // Background
    }
    
    void generateDilepton(CollisionEvent& event) {
        for (int i = 0; i < 2; i++) {
            generateSingleLepton(event);
        }
    }
    
    void generateMultijet(CollisionEvent& event) {
        std::uniform_int_distribution<int> nJets(3, 6);
        int jets = nJets(rng);
        for (int i = 0; i < jets; i++) {
            double energy = std::uniform_real_distribution<double>(20.0, 100.0)(rng);
            double eta = std::uniform_real_distribution<double>(-4.7, 4.7)(rng);
            double phi = std::uniform_real_distribution<double>(-M_PI, M_PI)(rng);
            int etaBin = static_cast<int>((eta + 4.7) * event.getHCAL()->getEtaBins() / 9.4);
            int phiBin = static_cast<int>((phi + M_PI) * event.getHCAL()->getPhiBins() / (2 * M_PI));
            for (int dEta = -1; dEta <= 1; dEta++) {
                for (int dPhi = -1; dPhi <= 1; dPhi++) {
                    int eBin = etaBin + dEta;
                    int pBin = phiBin + dPhi;
                    if (eBin >= 0 && eBin < event.getHCAL()->getEtaBins() &&
                        pBin >= 0 && pBin < event.getHCAL()->getPhiBins()) {
                        event.getHCAL()->depositEnergy(eBin, pBin, energy * 0.7);
                        event.getECAL()->depositEnergy(eBin * 2, pBin * 5, energy * 0.3);
                    }
                }
            }
        }
        depositRandomEnergy(event, 0.5, 5.0, 15);
    }
    
    void depositRandomEnergy(CollisionEvent& event, double minE, double maxE, int n = 1) {
        std::uniform_real_distribution<double> eDist(minE, maxE);
        for (int i = 0; i < n; i++) {
            int det = std::uniform_int_distribution<int>(0, 3)(rng);
            DetectorLayer* layer = nullptr;
            switch (det) {
                case 0: layer = event.getECAL(); break;
                case 1: layer = event.getHCAL(); break;
                case 2: layer = event.getMuon(); break;
                case 3: layer = event.getTracker(); break;
            }
            int eta = std::uniform_int_distribution<int>(0, layer->getEtaBins() - 1)(rng);
            int phi = std::uniform_int_distribution<int>(0, layer->getPhiBins() - 1)(rng);
            layer->depositEnergy(eta, phi, eDist(rng));
        }
    }
    
public:
    EventGenerator() : rng(std::random_device{}()), eventCounter(0) {}
    
    std::shared_ptr<CollisionEvent> generate() {
        auto event = std::make_shared<CollisionEvent>(eventCounter++);
        std::discrete_distribution<int> processDist({60, 20, 10, 10});  // QCD, SingleLep, DiLep, MultiJet
        ProcessType process = static_cast<ProcessType>(processDist(rng));
        
        switch (process) {
            case QCD: generateQCD(*event); break;
            case SINGLE_LEP: generateSingleLepton(*event); break;
            case DILEP: generateDilepton(*event); break;
            case MULTIJET: generateMultijet(*event); break;
        }
        return event;
    }
};

} // namespace CMSL1Sim

int main() {
    using namespace CMSL1Sim;
    
    EventGenerator generator;
    L1TriggerSystem trigger;
    
    trigger.start();
    
    const int nEvents = 500;
    for (int i = 0; i < nEvents; i++) {
        auto event = generator.generate();
        trigger.queueEvent(event);
        
        if (i % 50 == 0) {
            std::cout << "Generated " << i << " events\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));  // ~40 kHz LHC rate
    }
    
    while (trigger.getProcessedEvents() < nEvents) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::cout << "Processed: " << trigger.getProcessedEvents() << "/" << nEvents << std::endl;
    }
    
    trigger.stop();
    
    std::cout << "\nSimulation Summary:\n";
    std::cout << "Events Processed: " << trigger.getProcessedEvents() << std::endl;
    std::cout << "Events Passed: " << trigger.getPassedEvents() << std::endl;
    std::cout << "Trigger Efficiency: " << (trigger.getEfficiency() * 100) << "%\n";
    
    return 0;
}
