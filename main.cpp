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
    MUON_DET,    // Muon System (renamed to avoid conflict)
    TRACKER  // Silicon Tracker
};

enum ParticleType {
    ELECTRON,
    PHOTON,
    MUON,    // Muon as a particle (distinct from MUON_DET)
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
    ECALLayer() : DetectorLayer(ECAL, 140, 360) {}
};

class HCALLayer : public DetectorLayer {
public:
    HCALLayer() : DetectorLayer(HCAL, 82, 72) {}
};

class MuonLayer : public DetectorLayer {
public:
    MuonLayer() : DetectorLayer(MUON_DET, 48, 48) {}
};

class TrackerLayer : public DetectorLayer {
public:
    TrackerLayer() : DetectorLayer(TRACKER, 200, 360) {}
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
            if (cand.type == ParticleType::MUON && cand.pt > ptThreshold && cand.isolated) {
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
            if (cand.type == ParticleType::JET && cand.pt > ptThreshold) {
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
            if ((cand.type == ParticleType::ELECTRON || cand.type == ParticleType::PHOTON) && 
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
                    double pt
