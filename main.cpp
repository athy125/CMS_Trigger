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
#include <fstream>

namespace CMSL1Sim {

    enum DetectorType {
        ECAL,
        HCAL,
        DET_MUON,
        TRACKER
    };

    enum ParticleType {
        ELECTRON,
        PHOTON,
        MUON,
        JET,
        TAU,
        MET
    };

    enum ProcessType {
        QCD,
        SINGLE_LEP,
        DILEP,
        MULTIJET
    };

    struct TriggerCandidate {
        ParticleType type;
        double pt;
        double eta;
        double phi;
        double energy;
        bool isolated;
        std::vector<std::pair<double, double>> trajectory; // (eta, phi) at each layer
        
        TriggerCandidate(ParticleType t, double p, double e, double ph, double en, bool iso = false)
            : type(t), pt(p), eta(e), phi(ph), energy(en), isolated(iso) {}
    };

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
        ProcessType processType;
        
    public:
        CollisionEvent(uint64_t id, ProcessType proc = QCD)
            : eventId(id), ecal(std::make_unique<ECALLayer>()),
              hcal(std::make_unique<HCALLayer>()), muon(std::make_unique<MuonLayer>()),
              tracker(std::make_unique<TrackerLayer>()), triggerDecision(false),
              timestamp(std::chrono::high_resolution_clock::now().time_since_epoch()),
              processType(proc) {}
        
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
        ProcessType getProcessType() const { return processType; }
        
        void saveToJSON(const std::string& filename) const {
            std::ofstream out(filename);
            out << "{\n";
            out << "  \"id\": " << eventId << ",\n";
            out << "  \"passed\": " << (triggerDecision ? "true" : "false") << ",\n";
            out << "  \"process\": " << processType << ",\n";
            out << "  \"detectors\": {\n";
            out << "    \"ECAL\": [";
            for (int i = 0; i < ecal->getEtaBins(); i++) {
                out << (i > 0 ? "," : "") << "[";
                for (int j = 0; j < ecal->getPhiBins(); j++) {
                    out << (j > 0 ? "," : "") << ecal->getEnergy(i, j);
                }
                out << "]";
            }
            out << "],\n";
            out << "    \"HCAL\": [";
            for (int i = 0; i < hcal->getEtaBins(); i++) {
                out << (i > 0 ? "," : "") << "[";
                for (int j = 0; j < hcal->getPhiBins(); j++) {
                    out << (j > 0 ? "," : "") << hcal->getEnergy(i, j);
                }
                out << "]";
            }
            out << "],\n";
            out << "    \"MUON\": [";
            for (int i = 0; i < muon->getEtaBins(); i++) {
                out << (i > 0 ? "," : "") << "[";
                for (int j = 0; j < muon->getPhiBins(); j++) {
                    out << (j > 0 ? "," : "") << muon->getEnergy(i, j);
                }
                out << "]";
            }
            out << "],\n";
            out << "    \"TRACKER\": [";
            for (int i = 0; i < tracker->getEtaBins(); i++) {
                out << (i > 0 ? "," : "") << "[";
                for (int j = 0; j < tracker->getPhiBins(); j++) {
                    out << (j > 0 ? "," : "") << tracker->getEnergy(i, j);
                }
                out << "]";
            }
            out << "]\n";
            out << "  },\n";
            out << "  \"candidates\": [";
            for (size_t i = 0; i < candidates.size(); i++) {
                const auto& cand = candidates[i];
                out << (i > 0 ? "," : "") << "{";
                out << "\"type\": " << cand.type << ",";
                out << "\"pt\": " << cand.pt << ",";
                out << "\"eta\": " << cand.eta << ",";
                out << "\"phi\": " << cand.phi << ",";
                out << "\"energy\": " << cand.energy << ",";
                out << "\"isolated\": " << (cand.isolated ? "true" : "false") << ",";
                out << "\"trajectory\": [";
                for (size_t j = 0; j < cand.trajectory.size(); j++) {
                    out << (j > 0 ? "," : "") << "[" << cand.trajectory[j].first << "," << cand.trajectory[j].second << "]";
                }
                out << "]}";
            }
            out << "]\n";
            out << "}\n";
            out.close();
        }
    };

    class TriggerLogic {
    protected:
        std::string name;
        double ptThreshold;
        
    public:
        TriggerLogic(const std::string& n, double pt) : name(n), ptThreshold(pt) {}
        virtual ~TriggerLogic() = default;
        virtual bool evaluate(const CollisionEvent& event) = 0;
        std::string getName() const { return name; }
        void setPtThreshold(double pt) { ptThreshold = pt; }
        double getPtThreshold() const { return ptThreshold; }
    };

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
                event->saveToJSON("data/event_" + std::to_string(event->getId()) + ".json");
            }
        }
        
        void reconstructCandidates(CollisionEvent& event) {
            auto* muonLayer = event.getMuon();
            for (int eta = 0; eta < muonLayer->getEtaBins(); eta++) {
                for (int phi = 0; phi < muonLayer->getPhiBins(); phi++) {
                    double energy = muonLayer->getEnergy(eta, phi);
                    if (energy > 5.0) {
                        double etaVal = -2.4 + (4.8 * eta) / muonLayer->getEtaBins();
                        double phiVal = -M_PI + (2 * M_PI * phi) / muonLayer->getPhiBins();
                        double pt = energy * std::sin(2 * std::atan(std::exp(-etaVal)));
                        TriggerCandidate cand(ParticleType::MUON, pt, etaVal, phiVal, energy, true);
                        cand.trajectory.push_back({etaVal, phiVal}); // Simplified trajectory
                        event.addCandidate(cand);
                    }
                }
            }
            
            auto* hcal = event.getHCAL();
            std::vector<std::vector<double>> jetEnergy(hcal->getEtaBins(), 
                                                      std::vector<double>(hcal->getPhiBins(), 0.0));
            
            for (int eta = 0; eta < hcal->getEtaBins(); eta++) {
                for (int phi = 0; phi < hcal->getPhiBins(); phi++) {
                    jetEnergy[eta][phi] = hcal->getEnergy(eta, phi) + 
                                         event.getECAL()->getEnergy(eta * 2, phi * 5);
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
                            TriggerCandidate cand(ParticleType::JET, pt, etaVal, phiVal, jetEnergy[eta][phi]);
                            cand.trajectory.push_back({etaVal, phiVal});
                            event.addCandidate(cand);
                        }
                    }
                }
            }
            
            auto* ecal = event.getECAL();
            for (int eta = 0; eta < ecal->getEtaBins(); eta++) {
                for (int phi = 0; phi < ecal->getPhiBins(); phi++) {
                    double energy = ecal->getEnergy(eta, phi);
                    if (energy > 3.0) {
                        double etaVal = -2.5 + (5.0 * eta) / ecal->getEtaBins();
                        double phiVal = -M_PI + (2 * M_PI * phi) / ecal->getPhiBins();
                        double pt = energy * std::sin(2 * std::atan(std::exp(-etaVal)));
                        
                        int trackEta = eta * 2;
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
                        
                        TriggerCandidate cand(hasTrack ? ParticleType::ELECTRON : ParticleType::PHOTON, 
                                             pt, etaVal, phiVal, energy, isolated);
                        cand.trajectory.push_back({etaVal, phiVal});
                        event.addCandidate(cand);
                    }
                }
            }
        }
        
        bool evaluateTriggers(const CollisionEvent& event) {
            for (const auto& trigger : triggerMenu) {
                if (trigger->evaluate(event)) {
                    std::cout << "  Passed trigger: " << trigger->getName() << std::endl;
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
        
        ~L1TriggerSystem() {
            if (isRunning) {
                isRunning = false;
                queueCond.notify_all();
                if (processorThread.joinable()) processorThread.join();
                std::cout << "L1 Trigger System stopped\n";
            }
        }
        
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

    class EventGenerator {
    private:
        std::mt19937 rng;
        uint64_t eventCounter;
        
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
            if (lepType(rng)) {
                etaBin = static_cast<int>((eta + 2.4) * event.getMuon()->getEtaBins() / 4.8);
                phiBin = static_cast<int>((phi + M_PI) * event.getMuon()->getPhiBins() / (2 * M_PI));
                event.getMuon()->depositEnergy(etaBin, phiBin, energy);
                etaBin = static_cast<int>((eta + 2.5) * event.getTracker()->getEtaBins() / 5.0);
                phiBin = static_cast<int>((phi + M_PI) * event.getTracker()->getPhiBins() / (2 * M_PI));
                event.getTracker()->depositEnergy(etaBin, phiBin, energy * 0.1);
            } else {
                etaBin = static_cast<int>((eta + 2.5) * event.getECAL()->getEtaBins() / 5.0);
                phiBin = static_cast<int>((phi + M_PI) * event.getECAL()->getPhiBins() / (2 * M_PI));
                event.getECAL()->depositEnergy(etaBin, phiBin, energy);
                etaBin = static_cast<int>((eta + 2.5) * event.getTracker()->getEtaBins() / 5.0);
                event.getTracker()->depositEnergy(etaBin, phiBin, energy * 0.1);
            }
            depositRandomEnergy(event, 0.5, 5.0, 10);
        }
        
        void generateDilepton(CollisionEvent& event) {
            generateSingleLepton(event);
            generateSingleLepton(event);
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
            auto event = std::make_shared<CollisionEvent>(eventCounter);
            std::discrete_distribution<int> processDist({60, 20, 10, 10});
            ProcessType process = static_cast<ProcessType>(processDist(rng));
            event->processType = process;
            
            switch (process) {
                case QCD: generateQCD(*event); break;
                case SINGLE_LEP: generateSingleLepton(*event); break;
                case DILEP: generateDilepton(*event); break;
                case MULTIJET: generateMultijet(*event); break;
            }
            eventCounter++;
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
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
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
