#!/usr/bin/env python3
"""
NEURAL DIFFERENTIAL MANIFOLDS FOR TCP CONGESTION CONTROL (NDM-TCP)

Entropy-Aware Traffic Shaping with Continuous Weight Evolution

Features:
- Shannon Entropy to distinguish network noise from real congestion
- Differential manifold treating TCP as a "physical pipe" that bends
- Real-time neuroplasticity adapting to network conditions
- Security: Input validation, bounds checking, rate limiting
- Hebbian learning for traffic pattern recognition

Compile C library first:
    Windows: gcc -shared -o ndm_tcp.dll ndm_tcp.c -lm -O3 -fopenmp
    Linux:   gcc -shared -fPIC -o ndm_tcp.so ndm_tcp.c -lm -O3 -fopenmp
    Mac:     gcc -shared -fPIC -o ndm_tcp.dylib ndm_tcp.c -lm -O3 -Xpreprocessor -fopenmp -lomp

Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

# ============================================================================
# TCP STATE DATA CLASS
# ============================================================================

@dataclass
class TCPMetrics:
    """TCP network metrics"""
    current_rtt: float = 50.0        # Round-trip time (ms)
    min_rtt: float = 10.0            # Minimum observed RTT
    packet_loss_rate: float = 0.0    # Loss rate [0, 1]
    bandwidth_estimate: float = 100.0 # Bandwidth (Mbps)
    queue_delay: float = 5.0         # Queuing delay (ms)
    jitter: float = 2.0              # RTT variance
    throughput: float = 50.0         # Current throughput (Mbps)
    
    # Entropy metrics (computed automatically)
    shannon_entropy: float = 0.0
    noise_ratio: float = 0.2
    congestion_confidence: float = 0.8

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'ndm_tcp.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'ndm_tcp.dylib'
    else:
        lib_name = 'ndm_tcp.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o ndm_tcp.dll ndm_tcp.c -lm -O3")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o ndm_tcp.dylib ndm_tcp.c -lm -O3 -Xpreprocessor -fopenmp -lomp")
        else:
            print("  gcc -shared -fPIC -o ndm_tcp.so ndm_tcp.c -lm -O3 -fopenmp")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

try:
    _lib = load_library()
    print(f"✓ Loaded NDM-TCP C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C STRUCTURES
# ============================================================================

class CTCPState(ctypes.Structure):
    """C structure for TCP state"""
    _fields_ = [
        ("current_rtt", ctypes.c_float),
        ("min_rtt", ctypes.c_float),
        ("packet_loss_rate", ctypes.c_float),
        ("bandwidth_estimate", ctypes.c_float),
        ("queue_delay", ctypes.c_float),
        ("jitter", ctypes.c_float),
        ("throughput", ctypes.c_float),
        ("shannon_entropy", ctypes.c_float),
        ("noise_ratio", ctypes.c_float),
        ("congestion_confidence", ctypes.c_float),
        ("rtt_history", ctypes.c_float * 100),
        ("loss_history", ctypes.c_float * 100),
        ("history_index", ctypes.c_int),
        ("history_count", ctypes.c_int),
    ]

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

_lib.create_ndm_tcp.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
]
_lib.create_ndm_tcp.restype = ctypes.c_void_p

_lib.destroy_ndm_tcp.argtypes = [ctypes.c_void_p]
_lib.destroy_ndm_tcp.restype = None

_lib.create_tcp_state.argtypes = []
_lib.create_tcp_state.restype = ctypes.POINTER(CTCPState)

_lib.destroy_tcp_state.argtypes = [ctypes.POINTER(CTCPState)]
_lib.destroy_tcp_state.restype = None

_lib.update_tcp_state_entropy.argtypes = [
    ctypes.POINTER(CTCPState), ctypes.c_float, ctypes.c_float
]
_lib.update_tcp_state_entropy.restype = None

_lib.ndm_tcp_forward.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(CTCPState), ctypes.POINTER(ctypes.c_float)
]
_lib.ndm_tcp_forward.restype = None

_lib.apply_tcp_actions.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_float
]
_lib.apply_tcp_actions.restype = None

_lib.ndm_tcp_train.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(CTCPState), ctypes.c_float
]
_lib.ndm_tcp_train.restype = ctypes.c_float

_lib.ndm_tcp_reset_memory.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_reset_memory.restype = None

_lib.ndm_tcp_get_avg_weight_velocity.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_get_avg_weight_velocity.restype = ctypes.c_float

_lib.ndm_tcp_get_avg_plasticity.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_get_avg_plasticity.restype = ctypes.c_float

_lib.ndm_tcp_get_avg_manifold_energy.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_get_avg_manifold_energy.restype = ctypes.c_float

_lib.ndm_tcp_get_avg_entropy.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_get_avg_entropy.restype = ctypes.c_float

_lib.ndm_tcp_get_current_cwnd.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_get_current_cwnd.restype = ctypes.c_float

_lib.ndm_tcp_get_current_ssthresh.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_get_current_ssthresh.restype = ctypes.c_float

_lib.ndm_tcp_get_current_pacing_rate.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_get_current_pacing_rate.restype = ctypes.c_float

_lib.ndm_tcp_print_info.argtypes = [ctypes.c_void_p]
_lib.ndm_tcp_print_info.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class NDMTCPController:
    """
    Neural Differential Manifolds for TCP Congestion Control
    
    This controller uses entropy analysis to distinguish between:
    - Network Noise: Random fluctuations (high entropy)
    - Real Congestion: Structured bottlenecks (low entropy)
    
    The network treats TCP connections as a "physical manifold" that
    bends and adapts to traffic load, maintaining low latency.
    
    Features:
    - Shannon Entropy calculation for noise detection
    - Continuous weight evolution via ODEs
    - Hebbian learning for traffic patterns
    - Adaptive plasticity based on network conditions
    - Security: Input validation and bounds checking
    
    Parameters
    ----------
    input_size : int, default=15
        Dimension of TCP state vector
    hidden_size : int, default=64
        Number of hidden neurons
    output_size : int, default=3
        Actions: [cwnd_delta, ssthresh_delta, pacing_rate_multiplier]
    manifold_size : int, default=32
        Size of associative memory for traffic patterns
    learning_rate : float, default=0.01
        Learning rate for policy updates
    
    Examples
    --------
    >>> controller = NDMTCPController(hidden_size=64)
    >>> metrics = TCPMetrics(current_rtt=60, packet_loss_rate=0.01)
    >>> actions = controller.forward(metrics)
    >>> print(f"CWND adjustment: {actions['cwnd_delta']:.2f}")
    """
    
    def __init__(self, input_size: int = 15, hidden_size: int = 64,
                 output_size: int = 3, manifold_size: int = 32,
                 learning_rate: float = 0.01):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.manifold_size = manifold_size
        
        self._net = _lib.create_ndm_tcp(
            input_size, hidden_size, output_size, manifold_size, learning_rate
        )
        
        if not self._net:
            raise RuntimeError("Failed to create NDM-TCP network")
        
        # Create internal TCP state tracker
        self._tcp_state = _lib.create_tcp_state()
        if not self._tcp_state:
            raise RuntimeError("Failed to create TCP state")
        
        # History tracking
        self._cwnd_history = []
        self._entropy_history = []
        self._reward_history = []
    
    def __del__(self):
        if hasattr(self, '_tcp_state') and self._tcp_state:
            _lib.destroy_tcp_state(self._tcp_state)
        if hasattr(self, '_net') and self._net:
            _lib.destroy_ndm_tcp(self._net)
    
    def update_state(self, metrics: TCPMetrics):
        """Update internal TCP state with new measurements"""
        # Update C structure
        self._tcp_state.contents.current_rtt = metrics.current_rtt
        self._tcp_state.contents.min_rtt = metrics.min_rtt
        self._tcp_state.contents.packet_loss_rate = metrics.packet_loss_rate
        self._tcp_state.contents.bandwidth_estimate = metrics.bandwidth_estimate
        self._tcp_state.contents.queue_delay = metrics.queue_delay
        self._tcp_state.contents.jitter = metrics.jitter
        self._tcp_state.contents.throughput = metrics.throughput
        
        # Update entropy (C function will calculate Shannon entropy)
        _lib.update_tcp_state_entropy(
            self._tcp_state,
            metrics.current_rtt,
            metrics.packet_loss_rate
        )
        
        # Read back computed entropy values
        metrics.shannon_entropy = self._tcp_state.contents.shannon_entropy
        metrics.noise_ratio = self._tcp_state.contents.noise_ratio
        metrics.congestion_confidence = self._tcp_state.contents.congestion_confidence
    
    def forward(self, metrics: TCPMetrics) -> Dict[str, float]:
        """
        Forward pass: Determine TCP actions
        
        Returns
        -------
        dict with keys:
            - cwnd_delta: Change in congestion window
            - ssthresh_delta: Change in slow start threshold
            - pacing_multiplier: Pacing rate multiplier
            - entropy: Current Shannon entropy
            - noise_ratio: Estimated noise ratio
            - congestion_confidence: Confidence in real congestion
        """
        # Update state with entropy calculation
        self.update_state(metrics)
        
        # Forward pass through network
        actions = np.zeros(3, dtype=np.float32)
        actions_ptr = actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        _lib.ndm_tcp_forward(self._net, self._tcp_state, actions_ptr)
        
        return {
            'cwnd_delta': float(actions[0]),
            'ssthresh_delta': float(actions[1]),
            'pacing_multiplier': float(actions[2]),
            'entropy': metrics.shannon_entropy,
            'noise_ratio': metrics.noise_ratio,
            'congestion_confidence': metrics.congestion_confidence,
        }
    
    def train_step(self, metrics: TCPMetrics, reward: float) -> float:
        """
        Single training step with reinforcement learning
        
        Reward should be:
        - Positive: Good throughput, low latency
        - Negative: Packet loss, high latency, low throughput
        
        Returns the absolute reward value
        """
        self.update_state(metrics)
        
        loss = _lib.ndm_tcp_train(self._net, self._tcp_state, reward)
        
        self._reward_history.append(reward)
        self._cwnd_history.append(self.current_cwnd)
        self._entropy_history.append(self.avg_entropy)
        
        return float(loss)
    
    def apply_actions(self, actions: Dict[str, float], reward: float = 0.0):
        """Apply actions to TCP state"""
        actions_array = np.array([
            actions['cwnd_delta'],
            actions['ssthresh_delta'],
            actions['pacing_multiplier']
        ], dtype=np.float32)
        
        actions_ptr = actions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        _lib.apply_tcp_actions(self._net, actions_ptr, reward)
    
    def reset_memory(self):
        """Reset memory states (but not learned weights)"""
        _lib.ndm_tcp_reset_memory(self._net)
    
    @property
    def current_cwnd(self) -> float:
        """Current congestion window"""
        return _lib.ndm_tcp_get_current_cwnd(self._net)
    
    @property
    def current_ssthresh(self) -> float:
        """Current slow start threshold"""
        return _lib.ndm_tcp_get_current_ssthresh(self._net)
    
    @property
    def current_pacing_rate(self) -> float:
        """Current pacing rate (Mbps)"""
        return _lib.ndm_tcp_get_current_pacing_rate(self._net)
    
    @property
    def avg_weight_velocity(self) -> float:
        """Average rate of weight change (neuroplasticity)"""
        return _lib.ndm_tcp_get_avg_weight_velocity(self._net)
    
    @property
    def avg_plasticity(self) -> float:
        """Average plasticity (0=rigid, 1=fluid)"""
        return _lib.ndm_tcp_get_avg_plasticity(self._net)
    
    @property
    def avg_manifold_energy(self) -> float:
        """Energy in associative memory manifold"""
        return _lib.ndm_tcp_get_avg_manifold_energy(self._net)
    
    @property
    def avg_entropy(self) -> float:
        """Average Shannon entropy"""
        return _lib.ndm_tcp_get_avg_entropy(self._net)
    
    def print_info(self):
        """Print network and TCP state information"""
        _lib.ndm_tcp_print_info(self._net)
    
    def get_cwnd_history(self) -> np.ndarray:
        """Get history of congestion window"""
        return np.array(self._cwnd_history)
    
    def get_entropy_history(self) -> np.ndarray:
        """Get history of Shannon entropy"""
        return np.array(self._entropy_history)
    
    def get_reward_history(self) -> np.ndarray:
        """Get history of rewards"""
        return np.array(self._reward_history)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def simulate_network_condition(
    base_rtt: float = 50.0,
    congestion_level: float = 0.0,
    noise_level: float = 0.1
) -> TCPMetrics:
    """
    Simulate network conditions
    
    Parameters
    ----------
    base_rtt : float
        Base RTT in milliseconds
    congestion_level : float
        Congestion severity [0, 1]
    noise_level : float
        Random noise level [0, 1]
    
    Returns
    -------
    TCPMetrics with simulated values
    """
    # Add structured congestion
    rtt_increase = congestion_level * 100.0
    
    # Add random noise
    rtt_noise = np.random.randn() * noise_level * 20.0
    
    current_rtt = base_rtt + rtt_increase + rtt_noise
    current_rtt = max(1.0, current_rtt)
    
    # Packet loss increases with congestion
    packet_loss = congestion_level * 0.1 + np.random.rand() * noise_level * 0.05
    packet_loss = np.clip(packet_loss, 0.0, 1.0)
    
    # Bandwidth decreases with congestion
    bandwidth = 100.0 * (1.0 - congestion_level * 0.5)
    bandwidth += np.random.randn() * noise_level * 10.0
    bandwidth = max(1.0, bandwidth)
    
    # Queue delay correlates with congestion
    queue_delay = congestion_level * 50.0 + np.random.randn() * noise_level * 5.0
    queue_delay = max(0.0, queue_delay)
    
    # Jitter increases with noise
    jitter = noise_level * 10.0 + np.random.rand() * 5.0
    
    # Throughput
    throughput = bandwidth * (1.0 - packet_loss)
    
    return TCPMetrics(
        current_rtt=float(current_rtt),
        min_rtt=base_rtt,
        packet_loss_rate=float(packet_loss),
        bandwidth_estimate=float(bandwidth),
        queue_delay=float(queue_delay),
        jitter=float(jitter),
        throughput=float(throughput)
    )

def calculate_reward(
    metrics: TCPMetrics,
    previous_throughput: float,
    target_rtt: float = 50.0
) -> float:
    """
    Calculate reward for reinforcement learning
    
    Reward components:
    - High throughput: positive
    - Low latency: positive
    - Low packet loss: positive
    - RTT close to target: positive
    """
    # Throughput reward (0 to 100)
    throughput_reward = metrics.throughput
    
    # Latency penalty
    rtt_penalty = abs(metrics.current_rtt - target_rtt)
    
    # Loss penalty
    loss_penalty = metrics.packet_loss_rate * 1000.0
    
    # Combined reward
    reward = throughput_reward - rtt_penalty - loss_penalty
    
    return float(reward)

# ============================================================================
# DEMONSTRATION
# ============================================================================

def demo():
    """Demonstrate NDM-TCP capabilities"""
    print("\n" + "="*70)
    print("NEURAL DIFFERENTIAL MANIFOLDS FOR TCP CONGESTION CONTROL")
    print("Entropy-Aware Traffic Shaping")
    print("="*70)
    
    np.random.seed(42)
    
    # Create controller
    controller = NDMTCPController(
        input_size=15,
        hidden_size=64,
        output_size=3,
        manifold_size=32,
        learning_rate=0.01
    )
    
    print("\nController Architecture:")
    controller.print_info()
    
    print("\n" + "="*70)
    print("SCENARIO 1: NETWORK NOISE (High Entropy)")
    print("="*70)
    print("Simulating random RTT fluctuations without real congestion\n")
    
    for step in range(5):
        # High noise, low congestion
        metrics = simulate_network_condition(
            base_rtt=50.0,
            congestion_level=0.1,
            noise_level=0.8  # High noise
        )
        
        actions = controller.forward(metrics)
        
        print(f"Step {step+1}:")
        print(f"  RTT: {metrics.current_rtt:.2f} ms")
        print(f"  Loss Rate: {metrics.packet_loss_rate:.4f}")
        print(f"  Shannon Entropy: {actions['entropy']:.4f} (HIGH → Noise detected)")
        print(f"  Noise Ratio: {actions['noise_ratio']:.4f}")
        print(f"  Congestion Confidence: {actions['congestion_confidence']:.4f}")
        print(f"  → CWND Delta: {actions['cwnd_delta']:.2f} (should be small/positive)")
        print()
    
    controller.reset_memory()
    
    print("\n" + "="*70)
    print("SCENARIO 2: REAL CONGESTION (Low Entropy)")
    print("="*70)
    print("Simulating sustained congestion at bottleneck\n")
    
    for step in range(5):
        # Low noise, high congestion
        metrics = simulate_network_condition(
            base_rtt=50.0,
            congestion_level=0.8,  # High congestion
            noise_level=0.1   # Low noise
        )
        
        actions = controller.forward(metrics)
        
        print(f"Step {step+1}:")
        print(f"  RTT: {metrics.current_rtt:.2f} ms")
        print(f"  Loss Rate: {metrics.packet_loss_rate:.4f}")
        print(f"  Shannon Entropy: {actions['entropy']:.4f} (LOW → Congestion detected)")
        print(f"  Noise Ratio: {actions['noise_ratio']:.4f}")
        print(f"  Congestion Confidence: {actions['congestion_confidence']:.4f}")
        print(f"  → CWND Delta: {actions['cwnd_delta']:.2f} (should be negative)")
        print()
    
    print("\n" + "="*70)
    print("KEY INSIGHT: ENTROPY DISTINGUISHES NOISE FROM CONGESTION")
    print("="*70)
    print("\n✓ High Entropy → Random noise → Don't reduce CWND aggressively")
    print("✓ Low Entropy  → Real congestion → Reduce CWND to avoid collapse")
    print("\nThis prevents traditional TCP's overreaction to transient noise!")
    
    print("\n" + "="*70)
    print("NEUROPLASTICITY METRICS")
    print("="*70)
    print(f"\nWeight Velocity: {controller.avg_weight_velocity:.6f}")
    print(f"Plasticity: {controller.avg_plasticity:.4f}")
    print(f"Manifold Energy: {controller.avg_manifold_energy:.6f}")
    print(f"\nCurrent TCP State:")
    print(f"  CWND: {controller.current_cwnd:.2f} packets")
    print(f"  SSThresh: {controller.current_ssthresh:.2f} packets")
    print(f"  Pacing Rate: {controller.current_pacing_rate:.2f} Mbps")

if __name__ == "__main__":
    demo()