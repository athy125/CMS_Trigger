import os
import re
import json
import subprocess
import threading
import queue
import time
from collections import defaultdict
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

class TriggerVisualizer:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.events = []
        self.rate_data = []  # To store {'event_block': int, 'rate': float}
        self.trigger_stats = defaultdict(int)
        self.live_queue = queue.Queue()
        self.lock = threading.Lock()
        
        logging.basicConfig(filename=os.path.join(data_dir, 'trigger_log.txt'),
                          level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()

    def parse_simulation_output(self, output_file):
        self.logger.info(f"Parsing simulation output from {output_file}")
        event_pattern = re.compile(r"Event (\d+) (PASSED|REJECTED) \| Candidates: (\d+)")
        trigger_pattern = re.compile(r"  Passed trigger: (\w+)")
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        current_event = None
        for line in lines:
            event_match = event_pattern.search(line)
            if event_match:
                event_id, status, cand_count = event_match.groups()
                current_event = {
                    'id': int(event_id),
                    'passed': status == 'PASSED',
                    'candidates': int(cand_count),
                    'triggers': [],
                    'timestamp': time.time()
                }
                self.events.append(current_event)
                self.trigger_stats['total_events'] += 1
                if status == 'PASSED':
                    self.trigger_stats['total_passed'] += 1
                # Update rate_data every 10 events
                if len(self.events) % 10 == 0:
                    block = len(self.events) // 10
                    passed = sum(1 for e in self.events[-10:] if e['passed'])
                    rate = passed / 10.0
                    self.rate_data.append({'event_block': block, 'rate': rate})
            elif current_event and trigger_pattern.search(line):
                trigger_name = trigger_pattern.search(line).group(1)
                current_event['triggers'].append(trigger_name)
                self.trigger_stats[trigger_name] += 1

    def run_simulation(self, executable, num_events):
        self.logger.info(f"Running simulation with {executable} for {num_events} events")
        cmd = [executable]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        def stream_output():
            event_pattern = re.compile(r"Event (\d+) (PASSED|REJECTED) \| Candidates: (\d+)")
            trigger_pattern = re.compile(r"  Passed trigger: (\w+)")
            current_event = None
            
            for line in iter(process.stdout.readline, ''):
                with self.lock:
                    event_match = event_pattern.search(line)
                    if event_match:
                        event_id, status, cand_count = event_match.groups()
                        current_event = {
                            'id': int(event_id),
                            'passed': status == 'PASSED',
                            'candidates': int(cand_count),
                            'triggers': [],
                            'timestamp': time.time()
                        }
                        self.events.append(current_event)
                        self.live_queue.put(current_event.copy())
                        self.trigger_stats['total_events'] += 1
                        if status == 'PASSED':
                            self.trigger_stats['total_passed'] += 1
                        # Update rate_data every 10 events
                        if len(self.events) % 10 == 0:
                            block = len(self.events) // 10
                            passed = sum(1 for e in self.events[-10:] if e['passed'])
                            rate = passed / 10.0
                            self.rate_data.append({'event_block': block, 'rate': rate})
                    elif current_event and trigger_pattern.search(line):
                        trigger_name = trigger_pattern.search(line).group(1)
                        current_event['triggers'].append(trigger_name)
                        self.trigger_stats[trigger_name] += 1
                print(line, end='')
        
        threading.Thread(target=stream_output, daemon=True).start()
        threading.Thread(target=self.live_particle_animation, daemon=True).start()
        process.wait()

    def generate_test_data(self, num_events):
        self.logger.info(f"Generating {num_events} test events")
        particle_types = ['MUON', 'JET', 'ELECTRON', 'PHOTON']
        for i in range(num_events):
            event = {
                'id': i,
                'passed': np.random.choice([True, False]),
                'candidates': np.random.randint(0, 5),
                'triggers': [],
                'timestamp': time.time()
            }
            if event['passed']:
                event['triggers'] = [np.random.choice(['SingleMuon', 'DiJet', 'ElectronPhoton'])]
                self.trigger_stats[event['triggers'][0]] += 1
            self.events.append(event)
            self.save_event_data(event)
            # Update rate_data every 10 events
            if len(self.events) % 10 == 0:
                block = len(self.events) // 10
                passed = sum(1 for e in self.events[-10:] if e['passed'])
                rate = passed / 10.0
                self.rate_data.append({'event_block': block, 'rate': rate})

    def save_event_data(self, event):
        filename = os.path.join(self.data_dir, f"event_{event['id']}.json")
        with open(filename, 'w') as f:
            json.dump(event, f)

    def live_rate_plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Live Trigger Acceptance Rate')
        ax.set_xlabel('Event Block (x10)')
        ax.set_ylabel('Acceptance Rate')
        line, = ax.plot([], [], 'o-', label='Rate')
        ax.legend()
        
        def update(frame):
            with self.lock:
                if not self.rate_data:
                    return line,
                df = pd.DataFrame(self.rate_data)
                line.set_data(df['event_block'], df['rate'])
                ax.relim()
                ax.autoscale_view()
            return line,
        
        ani = animation.FuncAnimation(fig, update, interval=1000, blit=True)
        plt.show()

    def live_particle_animation(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.set_title('Live Acceptance Rate')
        ax1.set_xlabel('Event Block (x10)')
        ax1.set_ylabel('Acceptance Rate')
        ax2.set_title('Particle Trajectories')
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-np.pi, np.pi)
        ax2.set_xlabel('η')
        ax2.set_ylabel('φ')
        
        rate_line, = ax1.plot([], [], 'o-', label='Rate')
        ax1.legend()
        scat = ax2.scatter([], [], c='red', s=50)
        
        def update(frame):
            with self.lock:
                if not self.rate_data or self.live_queue.empty():
                    return rate_line, scat
                df = pd.DataFrame(self.rate_data)
                rate_line.set_data(df['event_block'], df['rate'])
                ax1.relim()
                ax1.autoscale_view()
                
                event_json = os.path.join(self.data_dir, f"event_{self.events[-1]['id']}.json")
                if os.path.exists(event_json):
                    with open(event_json, 'r') as f:
                        event = json.load(f)
                    eta = [cand['eta'] for cand in event['candidates']]
                    phi = [cand['phi'] for cand in event['candidates']]
                    scat.set_offsets(np.c_[eta, phi])
            return rate_line, scat
        
        ani = animation.FuncAnimation(fig, update, interval=100, blit=True)
        plt.show()

    def plot_trigger_rates(self, save=False):
        if not self.rate_data:
            self.logger.warning("No rate data available to plot.")
            print("No rate data available to plot.")
            return
        
        df = pd.DataFrame(self.rate_data)
        plt.figure(figsize=(10, 6))
        plt.plot(df['event_block'], df['rate'], 'o-')
        plt.xlabel('Event Block (x10)')
        plt.ylabel('Acceptance Rate')
        plt.title('Trigger Acceptance Rate Over Time')
        if save:
            plt.savefig(os.path.join(self.data_dir, f"trigger_rate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.show()

    def plot_trigger_distribution(self, save=False):
        triggers = [t for e in self.events for t in e['triggers']]
        if not triggers:
            self.logger.warning("No trigger data available to plot.")
            print("No trigger data available to plot.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.hist(triggers, bins=len(set(triggers)), edgecolor='black')
        plt.xlabel('Trigger Type')
        plt.ylabel('Count')
        plt.title('Trigger Distribution')
        if save:
            plt.savefig(os.path.join(self.data_dir, f"trigger_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
        plt.show()

    def plot_3d_detector_event(self, event_id, save=False):
        filename = os.path.join(self.data_dir, f"event_{event_id}.json")
        if not os.path.exists(filename):
            self.logger.error(f"Event file {filename} not found")
            return
        
        with open(filename, 'r') as f:
            event_data = json.load(f)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        eta_bins = np.linspace(-2.5, 2.5, len(event_data['detectors']['ECAL']))
        phi_bins = np.linspace(-np.pi, np.pi, len(event_data['detectors']['ECAL'][0]))
        phi, eta = np.meshgrid(phi_bins, eta_bins)
        
        ecal = np.array(event_data['detectors']['ECAL'])
        x = np.cos(phi) * 1
        y = np.sin(phi) * 1
        z = eta
        ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(ecal / (ecal.max() + 1e-6)), alpha=0.7)
        
        ax.set_xlabel('X (cos φ)')
        ax.set_ylabel('Y (sin φ)')
        ax.set_zlabel('η')
        ax.set_title(f"3D Event Display - Event {event_id}")
        if save:
            plt.savefig(os.path.join(self.data_dir, f"3
