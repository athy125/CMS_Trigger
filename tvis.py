#!/usr/bin/env python3
# Visualization tools for the CMS L1 Trigger simulation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from datetime import datetime
import os
import subprocess
import re
import time
import argparse
from collections import defaultdict
import json
import threading
import queue
import matplotlib.animation as animation
from matplotlib.widgets import Slider

class TriggerVisualizer:
    """Class to visualize trigger data from the CMS L1 Trigger simulation"""
    
    def __init__(self, data_dir="./data", log_file="trigger_log.txt"):
        """Initialize the visualizer with data directory and logging"""
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.log_file = os.path.join(data_dir, log_file)
        self.events = []
        self.trigger_stats = defaultdict(int)
        self.rate_data = []
        self.live_queue = queue.Queue()  # For real-time data
        self.live_processing = False
        self.event_details = {}  # Store detailed event data
        self.lock = threading.Lock()  # Thread safety
        
    def log_message(self, message):
        """Log messages to file with timestamp"""
        with open(self.log_file, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
    
    def parse_simulation_output(self, output_file):
        """Parse the output of the CMS L1 Trigger simulation from a file"""
        event_pattern = re.compile(r"Event (\d+) (PASSED|REJECTED)")
        trigger_pattern = re.compile(r"  Passed trigger: (\w+)")
        
        self.log_message(f"Parsing simulation output from {output_file}")
        with open(output_file, 'r') as f:
            current_event = None
            for line in f:
                event_match = event_pattern.search(line)
                if event_match:
                    event_id = int(event_match.group(1))
                    passed = event_match.group(2) == "PASSED"
                    current_event = {
                        'id': event_id,
                        'passed': passed,
                        'triggers': [],
                        'timestamp': time.time()
                    }
                    with self.lock:
                        self.events.append(current_event)
                        if passed:
                            self.trigger_stats['total_passed'] += 1
                
                trigger_match = trigger_pattern.search(line)
                if trigger_match and current_event and current_event['passed']:
                    trigger_name = trigger_match.group(1)
                    current_event['triggers'].append(trigger_name)
                    with self.lock:
                        self.trigger_stats[trigger_name] += 1
        
        self.update_rates()
        self.log_message(f"Parsed {len(self.events)} events, {self.trigger_stats['total_passed']} passed")
        print(f"Parsed {len(self.events)} events")
        print(f"Total passed: {self.trigger_stats['total_passed']}")
    
    def parse_live_output(self, simulation_process):
        """Parse output from a running simulation process in real-time"""
        event_pattern = re.compile(r"Event (\d+) (PASSED|REJECTED)")
        trigger_pattern = re.compile(r"  Passed trigger: (\w+)")
        
        self.live_processing = True
        current_event = None
        self.log_message("Starting live output parsing")
        
        while self.live_processing:
            try:
                line = simulation_process.stdout.readline().decode('utf-8')
                if not line and simulation_process.poll() is not None:
                    break
                
                if line:
                    print(line, end='')
                    event_match = event_pattern.search(line)
                    if event_match:
                        event_id = int(event_match.group(1))
                        passed = event_match.group(2) == "PASSED"
                        current_event = {
                            'id': event_id,
                            'passed': passed,
                            'triggers': [],
                            'timestamp': time.time()
                        }
                        with self.lock:
                            self.events.append(current_event)
                            if passed:
                                self.trigger_stats['total_passed'] += 1
                        self.live_queue.put(current_event)
                    
                    trigger_match = trigger_pattern.search(line)
                    if trigger_match and current_event and current_event['passed']:
                        trigger_name = trigger_match.group(1)
                        current_event['triggers'].append(trigger_name)
                        with self.lock:
                            self.trigger_stats[trigger_name] += 1
            except Exception as e:
                self.log_message(f"Error in live parsing: {str(e)}")
                break
        
        self.live_processing = False
        self.update_rates()
        self.log_message("Live output parsing completed")
    
    def update_rates(self, window_size=100):
        """Update acceptance rate data"""
        with self.lock:
            self.rate_data.clear()
            for i in range(0, len(self.events), window_size):
                window = self.events[i:i+window_size]
                if window:
                    passes = sum(1 for e in window if e['passed'])
                    rate = passes / len(window)
                    self.rate_data.append({
                        'event_block': i // window_size,
                        'start_event': i,
                        'end_event': min(i+window_size, len(self.events)),
                        'rate': rate,
                        'timestamp': window[-1]['timestamp']
                    })
    
    def run_simulation(self, executable_path, num_events=1000):
        """Run the CMS L1 Trigger simulation and capture output"""
        cmd = [executable_path, str(num_events)]
        
        self.log_message(f"Running simulation with command: {' '.join(cmd)}")
        print(f"Running simulation with {num_events} events...")
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        
        # Start live parsing in a separate thread
        parse_thread = threading.Thread(target=self.parse_live_output, args=(process,))
        parse_thread.start()
        
        # Optionally start live visualization
        vis_thread = threading.Thread(target=self.live_rate_plot)
        vis_thread.start()
        
        return_code = process.wait()
        parse_thread.join()
        vis_thread.join()
        
        if return_code != 0:
            self.log_message(f"Simulation failed with code {return_code}")
            print(f"Simulation exited with code {return_code}")
        else:
            self.log_message("Simulation completed successfully")
    
    def plot_trigger_rates(self, save=False):
        """Plot the trigger acceptance rates"""
        if not self.events:
            print("No events to plot. Run simulation first.")
            return
            
        with self.lock:
            # Plot overall acceptance rate over time
            fig, ax = plt.subplots(figsize=(10, 6))
            df = pd.DataFrame(self.rate_data)
            ax.plot(df['event_block'], df['rate'], 'o-', label='Acceptance Rate')
            overall_rate = self.trigger_stats['total_passed'] / len(self.events)
            ax.axhline(y=overall_rate, color='r', linestyle='--', label=f'Overall Rate ({overall_rate:.2%})')
            ax.set_xlabel('Event Block (100 events each)')
            ax.set_ylabel('Acceptance Rate')
            ax.set_title('CMS L1 Trigger Acceptance Rate')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            if save:
                filename = os.path.join(self.data_dir, f'trigger_rate_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(filename, dpi=300)
                self.log_message(f"Saved trigger rate plot to {filename}")
            plt.show()
            
            # Plot trigger type distribution
            plt.figure(figsize=(12, 6))
            trigger_counts = {k: v for k, v in self.trigger_stats.items() if k != 'total_passed'}
            if trigger_counts:
                names = list(trigger_counts.keys())
                values = list(trigger_counts.values())
                bars = plt.bar(names, values)
                plt.xlabel('Trigger Type')
                plt.ylabel('Number of Events')
                plt.title('CMS L1 Trigger Type Distribution')
                plt.xticks(rotation=45)
                
                # Add percentage labels
                total_passed = self.trigger_stats['total_passed']
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height/total_passed:.1%}', ha='center', va='bottom')
                
                if save:
                    filename = os.path.join(self.data_dir, f'trigger_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                    plt.savefig(filename, dpi=300)
                    self.log_message(f"Saved trigger distribution plot to {filename}")
                plt.tight_layout()
                plt.show()
    
    def live_rate_plot(self):
        """Live plotting of trigger acceptance rate"""
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.title('Live CMS L1 Trigger Acceptance Rate')
        plt.xlabel('Event Block (100 events each)')
        plt.ylabel('Acceptance Rate')
        plt.grid(True, alpha=0.3)
        
        line, = ax.plot([], [], 'o-', label='Acceptance Rate')
        overall_line = ax.axhline(y=0, color='r', linestyle='--', label='Overall Rate')
        ax.legend()
        
        def update(frame):
            with self.lock:
                if not self.rate_data:
                    return line, overall_line
                df = pd.DataFrame(self.rate_data)
                line.set_data(df['event_block'], df['rate'])
                overall_rate = self.trigger_stats['total_passed'] / len(self.events) if self.events else 0
                overall_line.set_ydata([overall_rate])
                ax.relim()
                ax.autoscale_view()
            return line, overall_line
        
        ani = animation.FuncAnimation(fig, update, interval=1000, blit=True, repeat=False)
        plt.show()
    
    def plot_detector_event(self, event_data, save=False):
        """Plot detector data for a specific event with interactive controls"""
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(f"Event {event_data['id']} - {'Passed' if event_data['passed'] else 'Rejected'}", fontsize=16)
        
        # Create subplots
        axs = fig.subplot_mosaic([['ECAL', 'HCAL'], ['MUON', 'TRACKER']], layout='constrained')
        
        # Plot each detector
        detectors = {
            'ECAL': {'data': np.array(event_data['detectors']['ECAL']), 'cmap': 'viridis', 'vmax': 100},
            'HCAL': {'data': np.array(event_data['detectors']['HCAL']), 'cmap': 'plasma', 'vmax': 100},
            'MUON': {'data': np.array(event_data['detectors']['MUON']), 'cmap': 'cool', 'vmax': 100},
            'TRACKER': {'data': np.array(event_data['detectors']['TRACKER']), 'cmap': 'cividis', 'vmax': 10}
        }
        
        images = {}
        for det_name, det_info in detectors.items():
            ax = axs[det_name]
            im = ax.imshow(det_info['data'], cmap=det_info['cmap'], norm=LogNorm(vmin=0.1, vmax=det_info['vmax']))
            ax.set_title(f'{det_name} Energy Deposits')
            ax.set_xlabel('φ bin')
            ax.set_ylabel('η bin')
            fig.colorbar(im, ax=ax, label='Energy (GeV)')
            images[det_name] = im
        
        # Add trigger info
        trigger_text = "Triggers: " + ", ".join(event_data['triggers']) if event_data['passed'] else "No triggers"
        plt.figtext(0.5, 0.01, trigger_text, ha='center', fontsize=12, 
                    bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5})
        
        # Add slider for threshold adjustment
        ax_slider = plt.axes([0.25, 0.95, 0.5, 0.03], facecolor='lightgoldenrodyellow')
        threshold_slider = Slider(ax_slider, 'Energy Threshold', 0.1, 100.0, valinit=0.1, valstep=0.1)
        
        def update(val):
            threshold = threshold_slider.val
            for det_name, im in images.items():
                data = detectors[det_name]['data']
                masked_data = np.where(data > threshold, data, 0.1)
                im.set_data(masked_data)
            fig.canvas.draw_idle()
        
        threshold_slider.on_changed(update)
        
        if save:
            filename = os.path.join(self.data_dir, f'event_{event_data["id"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(filename, dpi=300)
            self.log_message(f"Saved event plot to {filename}")
        plt.show()
    
    def load_event_data(self, event_id):
        """Load detailed data for a specific event from file"""
        event_file = os.path.join(self.data_dir, f'event_{event_id}.json')
        if os.path.exists(event_file):
            with open(event_file, 'r') as f:
                self.log_message(f"Loaded event data for event {event_id}")
                return json.load(f)
        else:
            self.log_message(f"No detailed data found for event {event_id}")
            print(f"No detailed data found for event {event_id}")
            return None
    
    def save_event_data(self, event_data):
        """Save detailed data for an event to file"""
        event_file = os.path.join(self.data_dir, f'event_{event_data["id"]}.json')
        with open(event_file, 'w') as f:
            json.dump(event_data, f, indent=2)
        self.log_message(f"Saved event data for event {event_data['id']}")
    
    def export_stats(self, filename=None):
        """Export trigger statistics to a file with detailed analysis"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.data_dir, f'trigger_stats_{timestamp}.csv')
        
        with self.lock:
            stats_df = pd.DataFrame([
                {'trigger': trigger, 'count': count} 
                for trigger, count in self.trigger_stats.items()
                if trigger != 'total_passed'
            ])
            
            if not stats_df.empty:
                total_events = len(self.events)
                passed_events = self.trigger_stats['total_passed']
                stats_df['percentage'] = stats_df['count'] / passed_events * 100
                stats_df.to_csv(filename, index=False)
                self.log_message(f"Exported statistics to {filename}")
                
                # Detailed console output
                print("\nTrigger Statistics Summary:")
                print(f"Total events: {total_events}")
                print(f"Passed events: {passed_events} ({passed_events/total_events*100:.2f}%)")
                print("\nTrigger Type Distribution:")
                for _, row in stats_df.iterrows():
                    print(f"  {row['trigger']}: {row['count']} ({row['percentage']:.2f}%)")
                
                # Rate statistics
                rate_df = pd.DataFrame(self.rate_data)
                if not rate_df.empty:
                    print("\nRate Statistics:")
                    print(f"Mean acceptance rate: {rate_df['rate'].mean():.2%}")
                    print(f"Rate std dev: {rate_df['rate'].std():.2%}")
                    print(f"Max rate: {rate_df['rate'].max():.2%} (Block {rate_df['rate'].idxmax()})")
            else:
                print("No statistics to export")
                self.log_message("No statistics available for export")

def fake_detector_data(num_events=10):
    """Generate fake detector data for testing visualization"""
    np.random.seed(42)
    events = []
    
    for i in range(num_events):
        passed = np.random.random() > 0.7
        ecal = np.random.exponential(1.0, size=(72, 72)) * (np.random.random(size=(72, 72)) > 0.95)
        hcal = np.random.exponential(2.0, size=(72, 72)) * (np.random.random(size=(72, 72)) > 0.97)
        muon = np.random.exponential(3.0, size=(36, 36)) * (np.random.random(size=(36, 36)) > 0.98)
        tracker = np.random.exponential(0.5, size=(100, 180)) * (np.random.random(size=(100, 180)) > 0.99)
        
        # Add realistic signatures
        if np.random.random() > 0.5:  # Lepton or photon
            eta_pos, phi_pos = np.random.randint(10, 60, 2)
            energy = np.random.uniform(20, 80)
            ecal[eta_pos, phi_pos] = energy
            for de in range(-2, 3):
                for dp in range(-2, 3):
                    if 0 <= eta_pos + de < 72 and 0 <= phi_pos + dp < 72:
                        dist = np.sqrt(de**2 + dp**2)
                        if dist > 0:
                            ecal[eta_pos + de, phi_pos + dp] = energy * np.exp(-dist) * 0.3
            if np.random.random() > 0.5:  # Electron
                tracker[int(eta_pos * 100/72), int(phi_pos * 180/72)] = energy * 0.1
        
        if np.random.random() > 0.6:  # Jet
            eta_pos, phi_pos = np.random.randint(15, 55, 2)
            energy = np.random.uniform(40, 120)
            for de in range(-3, 4):
                for dp in range(-3, 4):
                    if 0 <= eta_pos + de < 72 and 0 <= phi_pos + dp < 72:
                        dist = np.sqrt(de**2 + dp**2)
                        if dist <= 3:
                            hcal[eta_pos + de, phi_pos + dp] = energy * np.exp(-dist/2) * 0.7
                            ecal[eta_pos + de, phi_pos + dp] += energy * np.exp(-dist/2) * 0.3
        
        event = {
            'id': i,
            'passed': passed,
            'triggers': ['SingleMuon', 'Jet', 'SingleElectron'][np.random.randint(0, 3)] if passed else [],
            'detectors': {
                'ECAL': ecal.tolist(),
                'HCAL': hcal.tolist(),
                'MUON': muon.tolist(),
                'TRACKER': tracker.tolist()
            }
        }
        events.append(event)
    return events

def main():
    parser = argparse.ArgumentParser(description='CMS L1 Trigger Visualization Tool')
    parser.add_argument('--simulate', action='store_true', help='Run the C++ simulation')
    parser.add_argument('--executable', type=str, default='./cms_l1_trigger_sim',
                        help='Path to the compiled C++ executable')
    parser.add_argument('--events', type=int, default=1000,
                        help='Number of events to simulate')
    parser.add_argument('--parse', type=str, help='Parse output from a file')
    parser.add_argument('--test', action='store_true', help='Run with test data')
    parser.add_argument('--view-event', type=int, help='View a specific event')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory for data files')
    
    args = parser.parse_args()
    visualizer = TriggerVisualizer(data_dir=args.data_dir)
    
    if args.simulate:
        if os.path.exists(args.executable):
            visualizer.run_simulation(args.executable, args.events)
            visualizer.plot_trigger_rates(save=True)
            visualizer.export_stats()
        else:
            print(f"Error: Executable not found at {args.executable}")
            return
    
    if args.parse:
        if os.path.exists(args.parse):
            visualizer.parse_simulation_output(args.parse)
            visualizer.plot_trigger_rates(save=True)
            visualizer.export_stats()
        else:
            print(f"Error: Output file not found at {args.parse}")
            return
    
    if args.test:
        print("Generating test data...")
        test_events = fake_detector_data(args.events if args.events else 10)
        for event in test_events:
            visualizer.events.append({'id': event['id'], 'passed': event['passed'], 
                                    'triggers': event['triggers'], 'timestamp': time.time()})
            if event['passed']:
                visualizer.trigger_stats['total_passed'] += 1
                for trigger in event['triggers']:
                    visualizer.trigger_stats[trigger] += 1
            visualizer.save_event_data(event)
        visualizer.update_rates(window_size=5)
        visualizer.plot_trigger_rates(save=True)
        visualizer.export_stats()
    
    if args.view_event is not None:
        event_data = visualizer.load_event_data(args.view_event)
        if event_data:
            visualizer.plot_detector_event(event_data, save=True)

if __name__ == "__main__":
    main()
