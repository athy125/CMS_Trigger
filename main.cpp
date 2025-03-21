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

namespace CMSL1Sim {

    // Enum for detector types
    enum DetectorType {
        ECAL,    // Electromagnetic Calorimeter
        HCAL,    // Hadronic Calorimeter
        DET_MUON, // Renamed to avoid conflict (Detector Muon)
        TRACKER  // Silicon Tracker
    };

    // Enum for particle types
    enum ParticleType {
        ELECTRON,
        PHOTON,
        MUON,    // Particle Muon (no conflict now)
        JET,
        TAU,
        MET      // Missing Transverse Energy
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
        std::vector<std::vector<double>> energyMap;
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
        MuonLayer() : DetectorLayer(DET_MUON, 48, 48) {}
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

    // Specific trigger implementation: Single Muon Trigger
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

    // Main L1 Trigger System (partial implementation)
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
                        event.addCandidate(TriggerCandidate(ParticleType::MUON, pt, etaVal, phiVal, energy, true));
                    }
                }
            }
            // Add other reconstructions (jets, electrons, etc.) as in your original code...
        }
        
        // Placeholder for processEvents (not fully shown here)
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
                // ... trigger evaluation logic ...
            }
        }
        
    public:
        L1TriggerSystem() : isRunning(false), processedEvents(0), passedEvents(0) {
            triggerMenu.push_back(std::make_unique<SingleMuonLogic>(25.0));
            // Add other triggers as needed...
        }
        
        ~L1TriggerSystem() { /* stop logic */ }
        
        void start() {
            if (!isRunning) {
                isRunning = true;
                processorThread = std::thread(&L1TriggerSystem::processEvents, this);
            }
        }
        
        void queueEvent(std::shared_ptr<CollisionEvent> event) {
            std::lock_guard<std::mutex> lock(queueMutex);
            eventQueue.push(event);
            queueCond.notify_one();
        }
        // ... other methods ...
    };

    // Event generator (placeholder, not fully modified here)
    class EventGenerator {
    public:
        EventGenerator() {}
        std::shared_ptr<CollisionEvent> generate() {
            // Simplified for brevity; add your original logic
            return std::make_shared<CollisionEvent>(0);
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
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2)); // Wait for processing
    return 0;
}
