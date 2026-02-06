#!/usr/bin/env python3
"""
NDM-TCP Interactive CLI Application

Features:
- Real-time network data collection from current system
- Interactive shell with multiple commands
- Training mode with live network monitoring
- Prediction/inference mode
- Model save/load functionality
- Verbose logging and visualization
- Network statistics and analysis

Commands:
  train       - Train model with real network data
  predict     - Predict TCP actions for current network state
  monitor     - Monitor network in real-time
  save        - Save trained model
  load        - Load saved model
  stats       - Show network statistics
  analyze     - Analyze network patterns
  reset       - Reset model memory
  info        - Show model information
  help        - Show available commands
  exit        - Exit the application
Licensed under GPL V3.
"""

import sys
import os
import time
import socket
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Add parent directory to path to import ndm_tcp
sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    from ndm_tcp import NDMTCPController, TCPMetrics
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Make sure ndm_tcp.py and its C library are in the same directory")
    sys.exit(1)

# Try to import optional network monitoring libraries
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Install with: pip install psutil")
    print("Real network monitoring will use simulated data instead.\n")


# ============================================================================
# NETWORK DATA COLLECTOR
# ============================================================================

class NetworkDataCollector:
    """Collect real network metrics from the current system"""
    
    def __init__(self, interface: Optional[str] = None, verbose: bool = True):
        self.interface = interface
        self.verbose = verbose
        self.history = []
        self.baseline_rtt = None
        self.rtt_samples = []
        
        if HAS_PSUTIL:
            self._init_psutil()
        else:
            if verbose:
                print("⚠️  Using simulated network data (psutil not available)")
    
    def _init_psutil(self):
        """Initialize psutil network monitoring"""
        if self.verbose:
            print("✓ psutil available - using real network data")
            net_if = psutil.net_if_stats()
            print(f"\nAvailable network interfaces:")
            for iface, stats in net_if.items():
                print(f"  - {iface}: {'UP' if stats.isup else 'DOWN'}")
    
    def measure_rtt(self, host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> Optional[float]:
        """Measure RTT by connecting to a remote host"""
        try:
            start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((host, port))
            sock.close()
            rtt_ms = (time.time() - start) * 1000
            self.rtt_samples.append(rtt_ms)
            
            # Keep last 100 samples
            if len(self.rtt_samples) > 100:
                self.rtt_samples.pop(0)
            
            return rtt_ms
        except Exception as e:
            if self.verbose:
                print(f"RTT measurement failed: {e}")
            return None
    
    def get_network_stats(self) -> Dict:
        """Get current network statistics"""
        if not HAS_PSUTIL:
            return self._get_simulated_stats()
        
        try:
            # Network I/O counters
            net_io = psutil.net_io_counters()
            
            # Get bytes/packets sent and received
            stats = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout,
            }
            
            # Calculate rates if we have history
            if self.history:
                prev = self.history[-1]
                time_delta = time.time() - prev['timestamp']
                
                if time_delta > 0:
                    stats['bytes_sent_rate'] = (stats['bytes_sent'] - prev['bytes_sent']) / time_delta
                    stats['bytes_recv_rate'] = (stats['bytes_recv'] - prev['bytes_recv']) / time_delta
                    stats['packets_sent_rate'] = (stats['packets_sent'] - prev['packets_sent']) / time_delta
                    stats['packets_recv_rate'] = (stats['packets_recv'] - prev['packets_recv']) / time_delta
                else:
                    stats['bytes_sent_rate'] = 0
                    stats['bytes_recv_rate'] = 0
                    stats['packets_sent_rate'] = 0
                    stats['packets_recv_rate'] = 0
            else:
                stats['bytes_sent_rate'] = 0
                stats['bytes_recv_rate'] = 0
                stats['packets_sent_rate'] = 0
                stats['packets_recv_rate'] = 0
            
            stats['timestamp'] = time.time()
            self.history.append(stats.copy())
            
            # Keep last 1000 samples
            if len(self.history) > 1000:
                self.history.pop(0)
            
            return stats
            
        except Exception as e:
            if self.verbose:
                print(f"Error getting network stats: {e}")
            return self._get_simulated_stats()
    
    def _get_simulated_stats(self) -> Dict:
        """Generate simulated network statistics"""
        base = 1000000 if self.history else 0
        noise = np.random.randn() * 10000
        
        return {
            'bytes_sent': base + noise,
            'bytes_recv': base + noise,
            'packets_sent': base // 1500,
            'packets_recv': base // 1500,
            'errin': 0,
            'errout': 0,
            'dropin': 0,
            'dropout': 0,
            'bytes_sent_rate': 1000000 + noise,
            'bytes_recv_rate': 1000000 + noise,
            'packets_sent_rate': 666 + noise / 1500,
            'packets_recv_rate': 666 + noise / 1500,
            'timestamp': time.time()
        }
    
    def collect_tcp_metrics(self) -> TCPMetrics:
        """Collect comprehensive TCP metrics from the system"""
        stats = self.get_network_stats()
        
        # Measure RTT
        rtt = self.measure_rtt()
        if rtt is None:
            rtt = 50.0 + np.random.randn() * 10
        
        # Calculate baseline RTT (minimum observed)
        if self.baseline_rtt is None:
            self.baseline_rtt = rtt
        else:
            self.baseline_rtt = min(self.baseline_rtt, rtt)
        
        # Calculate packet loss rate from error counters
        total_sent = stats.get('packets_sent', 1)
        errors = stats.get('errout', 0) + stats.get('dropout', 0)
        packet_loss = min(errors / max(total_sent, 1), 1.0)
        
        # Estimate bandwidth from current transfer rates
        bytes_rate = stats.get('bytes_sent_rate', 0) + stats.get('bytes_recv_rate', 0)
        bandwidth_mbps = (bytes_rate * 8) / 1_000_000  # Convert to Mbps
        bandwidth_mbps = max(1.0, min(bandwidth_mbps, 10000.0))  # Clamp to reasonable range
        
        # Calculate queue delay (difference from baseline RTT)
        queue_delay = max(0, rtt - self.baseline_rtt)
        
        # Calculate jitter (RTT variance)
        if len(self.rtt_samples) > 1:
            jitter = np.std(self.rtt_samples[-10:])  # Last 10 samples
        else:
            jitter = 2.0
        
        # Estimate throughput
        throughput = bandwidth_mbps * (1.0 - packet_loss)
        
        metrics = TCPMetrics(
            current_rtt=float(rtt),
            min_rtt=float(self.baseline_rtt),
            packet_loss_rate=float(packet_loss),
            bandwidth_estimate=float(bandwidth_mbps),
            queue_delay=float(queue_delay),
            jitter=float(jitter),
            throughput=float(throughput)
        )
        
        if self.verbose:
            self._print_metrics(metrics, stats)
        
        return metrics
    
    def _print_metrics(self, metrics: TCPMetrics, stats: Dict):
        """Print collected metrics in verbose mode"""
        print(f"\n{'─'*70}")
        print(f"📊 Network Metrics [Timestamp: {datetime.now().strftime('%H:%M:%S')}]")
        print(f"{'─'*70}")
        print(f"  RTT:              {metrics.current_rtt:>8.2f} ms")
        print(f"  Baseline RTT:     {metrics.min_rtt:>8.2f} ms")
        print(f"  Queue Delay:      {metrics.queue_delay:>8.2f} ms")
        print(f"  Jitter:           {metrics.jitter:>8.2f} ms")
        print(f"  Packet Loss:      {metrics.packet_loss_rate:>8.4f}")
        print(f"  Bandwidth:        {metrics.bandwidth_estimate:>8.2f} Mbps")
        print(f"  Throughput:       {metrics.throughput:>8.2f} Mbps")
        print(f"  Bytes Sent Rate:  {stats.get('bytes_sent_rate', 0):>8.0f} B/s")
        print(f"  Bytes Recv Rate:  {stats.get('bytes_recv_rate', 0):>8.0f} B/s")
        print(f"{'─'*70}")


# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    """Manage model saving and loading"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.models_dir = Path("./ndm_tcp_models")
        self.models_dir.mkdir(exist_ok=True)
    
    def save_model(self, controller: NDMTCPController, name: str, metadata: Optional[Dict] = None):
        """Save model configuration and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
        filepath = self.models_dir / filename
        
        model_data = {
            'config': {
                'input_size': controller.input_size,
                'hidden_size': controller.hidden_size,
                'output_size': controller.output_size,
                'manifold_size': controller.manifold_size,
                'learning_rate': controller.learning_rate,
            },
            'histories': {
                'cwnd': controller.get_cwnd_history().tolist(),
                'entropy': controller.get_entropy_history().tolist(),
                'reward': controller.get_reward_history().tolist(),
            },
            'stats': {
                'avg_weight_velocity': controller.avg_weight_velocity,
                'avg_plasticity': controller.avg_plasticity,
                'avg_manifold_energy': controller.avg_manifold_energy,
                'avg_entropy': controller.avg_entropy,
                'current_cwnd': controller.current_cwnd,
                'current_ssthresh': controller.current_ssthresh,
                'current_pacing_rate': controller.current_pacing_rate,
            },
            'metadata': metadata or {},
            'timestamp': timestamp,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.verbose:
            print(f"\n✓ Model saved to: {filepath}")
            print(f"  Config: {model_data['config']}")
            print(f"  Stats: CWND={model_data['stats']['current_cwnd']:.2f}, "
                  f"Entropy={model_data['stats']['avg_entropy']:.4f}")
        
        return str(filepath)
    
    def load_model(self, filepath: str) -> Tuple[NDMTCPController, Dict]:
        """Load model configuration and create new controller"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['config']
        controller = NDMTCPController(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            output_size=config['output_size'],
            manifold_size=config['manifold_size'],
            learning_rate=config['learning_rate'],
        )
        
        if self.verbose:
            print(f"\n✓ Model loaded from: {filepath}")
            print(f"  Config: {config}")
            print(f"  Timestamp: {model_data.get('timestamp', 'unknown')}")
        
        return controller, model_data
    
    def list_models(self) -> List[str]:
        """List all saved models"""
        models = list(self.models_dir.glob("*.pkl"))
        return sorted([str(m) for m in models], reverse=True)


# ============================================================================
# INTERACTIVE CLI APPLICATION
# ============================================================================

class NDMTCPCLIApp:
    """Interactive CLI application for NDM-TCP"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.controller = None
        self.collector = NetworkDataCollector(verbose=verbose)
        self.model_manager = ModelManager(verbose=verbose)
        self.running = True
        self.training_history = []
        
        # Initialize controller with default parameters
        self.reset_controller()
    
    def reset_controller(self):
        """Create a new controller instance"""
        if self.verbose:
            print("\n🔄 Creating new NDM-TCP controller...")
        
        self.controller = NDMTCPController(
            input_size=15,
            hidden_size=64,
            output_size=3,
            manifold_size=32,
            learning_rate=0.01
        )
        
        if self.verbose:
            print("✓ Controller initialized")
            self.controller.print_info()
    
    def print_banner(self):
        """Print application banner"""
        print("\n" + "="*70)
        print("NDM-TCP INTERACTIVE CLI - Neural Differential Manifolds for TCP")
        print("Entropy-Aware Congestion Control with Real Network Data")
        print("="*70)
        print(f"\nType 'help' for available commands")
        print(f"Network monitoring: {'REAL DATA' if HAS_PSUTIL else 'SIMULATED'}")
        print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
        print("="*70)
    
    def print_help(self):
        """Print help message"""
        print("\n📚 Available Commands:")
        print("─"*70)
        print("  train [steps]      - Train model with real network data (default: 10 steps)")
        print("  predict            - Predict TCP actions for current network state")
        print("  monitor [duration] - Monitor network in real-time (default: 30s)")
        print("  save [name]        - Save trained model (default name: 'model')")
        print("  load [path]        - Load saved model")
        print("  list               - List all saved models")
        print("  stats              - Show network statistics")
        print("  analyze            - Analyze network patterns and entropy")
        print("  reset              - Reset model memory")
        print("  info               - Show model information")
        print("  config             - Show/modify model configuration")
        print("  history            - Show training history")
        print("  benchmark          - Run performance benchmark")
        print("  help               - Show this help message")
        print("  exit/quit          - Exit the application")
        print("─"*70)
    
    def cmd_train(self, args: List[str]):
        """Train the model with real network data"""
        steps = int(args[0]) if args else 10
        
        print(f"\n🎓 Training NDM-TCP Controller ({steps} steps)")
        print(f"{'─'*70}")
        
        for step in range(steps):
            print(f"\n[Step {step+1}/{steps}]")
            
            # Collect real network metrics
            metrics = self.collector.collect_tcp_metrics()
            
            # Forward pass
            actions = self.controller.forward(metrics)
            
            # Calculate reward
            reward = self._calculate_reward(metrics)
            
            # Train
            loss = self.controller.train_step(metrics, reward)
            
            # Store history
            self.training_history.append({
                'step': step + 1,
                'metrics': metrics,
                'actions': actions,
                'reward': reward,
                'loss': loss,
                'timestamp': datetime.now()
            })
            
            # Print verbose output
            if self.verbose:
                self._print_training_step(step + 1, metrics, actions, reward, loss)
            
            # Sleep briefly between steps
            time.sleep(0.5)
        
        print(f"\n✓ Training complete!")
        print(f"  Total steps: {steps}")
        print(f"  Avg reward: {np.mean([h['reward'] for h in self.training_history[-steps:]]):.2f}")
        print(f"  Avg loss: {np.mean([h['loss'] for h in self.training_history[-steps:]]):.6f}")
    
    def _print_training_step(self, step: int, metrics: TCPMetrics, actions: Dict, reward: float, loss: float):
        """Print detailed training step information"""
        print(f"\n  🔍 Actions:")
        print(f"    CWND Delta:       {actions['cwnd_delta']:>8.2f}")
        print(f"    SSThresh Delta:   {actions['ssthresh_delta']:>8.2f}")
        print(f"\n  📈 Analysis:")
        print(f"    Shannon Entropy:  {actions['entropy']:>8.4f}")
        print(f"    Noise Ratio:      {actions['noise_ratio']:>8.4f}")
        print(f"    Congestion Conf:  {actions['congestion_confidence']:>8.4f}")
        print(f"\n  💰 Learning:")
        print(f"    Reward:           {reward:>8.2f}")
        print(f"    Loss:             {loss:>8.6f}")
        print(f"\n  🧠 Model State:")
        print(f"    Plasticity:       {self.controller.avg_plasticity:>8.4f}")
        print(f"    Weight Velocity:  {self.controller.avg_weight_velocity:>8.6f}")
        print(f"    Manifold Energy:  {self.controller.avg_manifold_energy:>8.6f}")
    
    def _calculate_reward(self, metrics: TCPMetrics, target_rtt: float = 50.0) -> float:
        """Calculate reward for training"""
        throughput_reward = metrics.throughput
        rtt_penalty = abs(metrics.current_rtt - target_rtt)
        loss_penalty = metrics.packet_loss_rate * 1000.0
        reward = throughput_reward - rtt_penalty - loss_penalty
        return float(reward)
    
    def cmd_predict(self, args: List[str]):
        """Predict TCP actions for current network state"""
        print(f"\n🔮 Predicting TCP Actions")
        print(f"{'─'*70}")
        
        metrics = self.collector.collect_tcp_metrics()
        actions = self.controller.forward(metrics)
        
        print(f"\n✓ Predicted Actions:")
        print(f"  CWND Delta:              {actions['cwnd_delta']:>10.2f}")
        print(f"  SSThresh Delta:          {actions['ssthresh_delta']:>10.2f}")
        print(f"\n📊 Network Analysis:")
        print(f"  Shannon Entropy:         {actions['entropy']:>10.4f}")
        print(f"  Noise Ratio:             {actions['noise_ratio']:>10.4f}")
        print(f"  Congestion Confidence:   {actions['congestion_confidence']:>10.4f}")
        print(f"\n🎯 Interpretation:")
        
        if actions['entropy'] > 0.7:
            print(f"  → HIGH ENTROPY: Random noise detected, maintaining CWND")
        elif actions['congestion_confidence'] > 0.7:
            print(f"  → LOW ENTROPY: Real congestion detected, reducing CWND")
        else:
            print(f"  → MODERATE: Mixed signals, cautious adjustment")
    
    def cmd_monitor(self, args: List[str]):
        """Monitor network in real-time"""
        duration = int(args[0]) if args else 30
        
        print(f"\n📡 Real-time Network Monitoring ({duration} seconds)")
        print(f"{'─'*70}")
        print(f"Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        samples = []
        
        try:
            while time.time() - start_time < duration:
                metrics = self.collector.collect_tcp_metrics()
                actions = self.controller.forward(metrics)
                
                samples.append({
                    'metrics': metrics,
                    'actions': actions,
                    'timestamp': time.time()
                })
                
                # Brief summary line
                print(f"  RTT: {metrics.current_rtt:>6.2f}ms | "
                      f"Loss: {metrics.packet_loss_rate:>6.4f} | "
                      f"BW: {metrics.bandwidth_estimate:>7.2f}Mbps | "
                      f"Entropy: {actions['entropy']:>6.4f} | "
                      f"CWND: {self.controller.current_cwnd:>6.2f}")
                
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Monitoring stopped by user")
        
        # Summary
        if samples:
            print(f"\n📊 Monitoring Summary:")
            print(f"  Duration:        {len(samples)} seconds")
            print(f"  Avg RTT:         {np.mean([s['metrics'].current_rtt for s in samples]):.2f} ms")
            print(f"  Avg Bandwidth:   {np.mean([s['metrics'].bandwidth_estimate for s in samples]):.2f} Mbps")
            print(f"  Avg Entropy:     {np.mean([s['actions']['entropy'] for s in samples]):.4f}")
            print(f"  Avg Loss Rate:   {np.mean([s['metrics'].packet_loss_rate for s in samples]):.6f}")
    
    def cmd_save(self, args: List[str]):
        """Save the model"""
        name = args[0] if args else "model"
        
        metadata = {
            'training_steps': len(self.training_history),
            'has_psutil': HAS_PSUTIL,
            'platform': sys.platform,
        }
        
        filepath = self.model_manager.save_model(self.controller, name, metadata)
        print(f"✓ Model saved successfully")
    
    def cmd_load(self, args: List[str]):
        """Load a model"""
        if not args:
            models = self.model_manager.list_models()
            if not models:
                print("❌ No saved models found")
                return
            
            print("\n📁 Available models:")
            for i, model in enumerate(models[:10], 1):
                print(f"  {i}. {Path(model).name}")
            
            choice = input("\nEnter model number to load (or path): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                filepath = models[int(choice) - 1]
            else:
                filepath = choice
        else:
            filepath = args[0]
        
        self.controller, model_data = self.model_manager.load_model(filepath)
        print(f"✓ Model loaded successfully")
    
    def cmd_list(self, args: List[str]):
        """List saved models"""
        models = self.model_manager.list_models()
        
        if not models:
            print("\n📁 No saved models found")
            return
        
        print(f"\n📁 Saved Models ({len(models)} total):")
        print("─"*70)
        
        for i, model in enumerate(models[:20], 1):
            path = Path(model)
            size = path.stat().st_size / 1024  # KB
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            print(f"  {i:2d}. {path.name}")
            print(f"      Size: {size:.1f} KB | Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def cmd_stats(self, args: List[str]):
        """Show network statistics"""
        print(f"\n📊 Network Statistics")
        print(f"{'─'*70}")
        
        stats = self.collector.get_network_stats()
        
        print(f"\n  Total Bytes Sent:     {stats.get('bytes_sent', 0):>15,} bytes")
        print(f"  Total Bytes Received: {stats.get('bytes_recv', 0):>15,} bytes")
        print(f"  Total Packets Sent:   {stats.get('packets_sent', 0):>15,} packets")
        print(f"  Total Packets Recv:   {stats.get('packets_recv', 0):>15,} packets")
        print(f"\n  Errors In:            {stats.get('errin', 0):>15,}")
        print(f"  Errors Out:           {stats.get('errout', 0):>15,}")
        print(f"  Drops In:             {stats.get('dropin', 0):>15,}")
        print(f"  Drops Out:            {stats.get('dropout', 0):>15,}")
        print(f"\n  Current Send Rate:    {stats.get('bytes_sent_rate', 0):>15,.0f} B/s")
        print(f"  Current Recv Rate:    {stats.get('bytes_recv_rate', 0):>15,.0f} B/s")
        
        if self.collector.rtt_samples:
            print(f"\n  RTT Samples:          {len(self.collector.rtt_samples)}")
            print(f"  Min RTT:              {min(self.collector.rtt_samples):>15.2f} ms")
            print(f"  Max RTT:              {max(self.collector.rtt_samples):>15.2f} ms")
            print(f"  Avg RTT:              {np.mean(self.collector.rtt_samples):>15.2f} ms")
            print(f"  Std RTT:              {np.std(self.collector.rtt_samples):>15.2f} ms")
    
    def cmd_analyze(self, args: List[str]):
        """Analyze network patterns"""
        print(f"\n🔬 Network Pattern Analysis")
        print(f"{'─'*70}")
        
        # Collect samples
        print("\nCollecting samples...")
        samples = []
        for i in range(10):
            metrics = self.collector.collect_tcp_metrics()
            actions = self.controller.forward(metrics)
            samples.append({'metrics': metrics, 'actions': actions})
            time.sleep(0.5)
            print(f"  Sample {i+1}/10 collected")
        
        # Analyze entropy distribution
        entropies = [s['actions']['entropy'] for s in samples]
        noise_ratios = [s['actions']['noise_ratio'] for s in samples]
        congestion_confs = [s['actions']['congestion_confidence'] for s in samples]
        
        print(f"\n📈 Entropy Analysis:")
        print(f"  Mean Entropy:         {np.mean(entropies):>10.4f}")
        print(f"  Std Entropy:          {np.std(entropies):>10.4f}")
        print(f"  Min Entropy:          {np.min(entropies):>10.4f}")
        print(f"  Max Entropy:          {np.max(entropies):>10.4f}")
        
        print(f"\n🔊 Noise vs Congestion:")
        print(f"  Avg Noise Ratio:      {np.mean(noise_ratios):>10.4f}")
        print(f"  Avg Congestion Conf:  {np.mean(congestion_confs):>10.4f}")
        
        # Determine network condition
        avg_entropy = np.mean(entropies)
        avg_congestion = np.mean(congestion_confs)
        
        print(f"\n🎯 Network Condition Assessment:")
        if avg_entropy > 0.7:
            print(f"  → HIGH NOISE: Network experiencing random fluctuations")
        elif avg_congestion > 0.7:
            print(f"  → CONGESTION: Sustained bottleneck detected")
        elif avg_entropy < 0.3:
            print(f"  → STABLE: Low entropy indicates predictable traffic")
        else:
            print(f"  → MIXED: Variable network conditions")
    
    def cmd_reset(self, args: List[str]):
        """Reset model memory"""
        self.controller.reset_memory()
        print("\n✓ Model memory reset")
        print("  All adaptive parameters cleared")
    
    def cmd_info(self, args: List[str]):
        """Show model information"""
        print(f"\n🧠 Model Information")
        print(f"{'─'*70}")
        
        self.controller.print_info()
        
        print(f"\n📊 Current State:")
        print(f"  CWND:             {self.controller.current_cwnd:>10.2f} packets")
        print(f"  SSThresh:         {self.controller.current_ssthresh:>10.2f} packets")
        print(f"  Pacing Rate:      {self.controller.current_pacing_rate:>10.2f} Mbps")
        
        print(f"\n🔄 Neuroplasticity Metrics:")
        print(f"  Weight Velocity:  {self.controller.avg_weight_velocity:>10.6f}")
        print(f"  Plasticity:       {self.controller.avg_plasticity:>10.4f}")
        print(f"  Manifold Energy:  {self.controller.avg_manifold_energy:>10.6f}")
        print(f"  Avg Entropy:      {self.controller.avg_entropy:>10.4f}")
    
    def cmd_history(self, args: List[str]):
        """Show training history"""
        if not self.training_history:
            print("\n📊 No training history available")
            print("  Run 'train' command first")
            return
        
        limit = int(args[0]) if args else 10
        recent = self.training_history[-limit:]
        
        print(f"\n📊 Training History (last {len(recent)} steps)")
        print(f"{'─'*70}")
        print(f"{'Step':<6} {'Reward':<10} {'Loss':<12} {'Entropy':<10} {'CWND':<10}")
        print(f"{'─'*70}")
        
        for h in recent:
            print(f"{h['step']:<6} "
                  f"{h['reward']:<10.2f} "
                  f"{h['loss']:<12.6f} "
                  f"{h['actions']['entropy']:<10.4f} "
                  f"{self.controller.current_cwnd:<10.2f}")
    
    def cmd_benchmark(self, args: List[str]):
        """Run performance benchmark"""
        iterations = int(args[0]) if args else 100
        
        print(f"\n⚡ Performance Benchmark ({iterations} iterations)")
        print(f"{'─'*70}")
        
        # Benchmark data collection
        print("\n1. Network Data Collection:")
        start = time.time()
        for _ in range(iterations):
            metrics = self.collector.collect_tcp_metrics()
        collection_time = time.time() - start
        print(f"   Time: {collection_time:.4f}s ({iterations/collection_time:.1f} samples/s)")
        
        # Benchmark forward pass
        print("\n2. Forward Pass (Inference):")
        metrics = self.collector.collect_tcp_metrics()
        start = time.time()
        for _ in range(iterations):
            actions = self.controller.forward(metrics)
        forward_time = time.time() - start
        print(f"   Time: {forward_time:.4f}s ({iterations/forward_time:.1f} inferences/s)")
        
        # Benchmark training
        print("\n3. Training Step:")
        start = time.time()
        for _ in range(iterations):
            loss = self.controller.train_step(metrics, 50.0)
        training_time = time.time() - start
        print(f"   Time: {training_time:.4f}s ({iterations/training_time:.1f} updates/s)")
        
        print(f"\n✓ Benchmark complete")
    
    def cmd_config(self, args: List[str]):
        """Show or modify configuration"""
        print(f"\n⚙️  Model Configuration")
        print(f"{'─'*70}")
        print(f"  Input Size:       {self.controller.input_size}")
        print(f"  Hidden Size:      {self.controller.hidden_size}")
        print(f"  Output Size:      {self.controller.output_size}")
        print(f"  Manifold Size:    {self.controller.manifold_size}")
        print(f"  Learning Rate:    {self.controller.learning_rate}")
        print(f"\n💡 To change config, create a new controller with 'reset'")
    
    def process_command(self, command: str):
        """Process a user command"""
        parts = command.strip().split()
        if not parts:
            return
        
        cmd = parts[0].lower()
        args = parts[1:]
        
        commands = {
            'train': self.cmd_train,
            'predict': self.cmd_predict,
            'monitor': self.cmd_monitor,
            'save': self.cmd_save,
            'load': self.cmd_load,
            'list': self.cmd_list,
            'stats': self.cmd_stats,
            'analyze': self.cmd_analyze,
            'reset': self.cmd_reset,
            'info': self.cmd_info,
            'history': self.cmd_history,
            'benchmark': self.cmd_benchmark,
            'config': self.cmd_config,
            'help': lambda _: self.print_help(),
            'exit': lambda _: self.exit_app(),
            'quit': lambda _: self.exit_app(),
        }
        
        if cmd in commands:
            try:
                commands[cmd](args)
            except Exception as e:
                print(f"\n❌ Error executing command: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
        else:
            print(f"\n❌ Unknown command: {cmd}")
            print("Type 'help' for available commands")
    
    def exit_app(self):
        """Exit the application"""
        print("\n👋 Exiting NDM-TCP CLI...")
        print("Thank you for using NDM-TCP!")
        self.running = False
    
    def run(self):
        """Run the interactive CLI"""
        self.print_banner()
        
        while self.running:
            try:
                command = input("\nndm-tcp> ").strip()
                if command:
                    self.process_command(command)
            except KeyboardInterrupt:
                print("\n\n⚠️  Use 'exit' or 'quit' to exit the application")
            except EOFError:
                self.exit_app()
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='NDM-TCP Interactive CLI - Neural Network TCP Congestion Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Start interactive shell
  %(prog)s --quiet            # Start with minimal output
  %(prog)s --auto-train 50    # Auto-train for 50 steps then enter shell
  
Interactive Commands:
  train [steps]      Train with real network data
  predict            Get TCP action predictions
  monitor [sec]      Real-time network monitoring
  save [name]        Save trained model
  help               Show all commands
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Enable verbose output (default: True)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Disable verbose output')
    parser.add_argument('--auto-train', type=int, metavar='STEPS',
                        help='Auto-train for N steps before entering shell')
    
    args = parser.parse_args()
    
    # Determine verbosity (default is True unless --quiet is specified)
    verbose = not args.quiet if args.quiet else True
    
    # Create and run the application
    app = NDMTCPCLIApp(verbose=verbose)
    
    # Auto-train if requested
    if args.auto_train:
        print(f"\n🚀 Auto-training for {args.auto_train} steps...")
        app.cmd_train([str(args.auto_train)])
    
    # Run interactive shell
    app.run()


if __name__ == "__main__":
    main()