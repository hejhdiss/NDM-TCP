# NDM-TCP GUI Application

A comprehensive PySide6 GUI application for the Neural Differential Manifolds TCP Congestion Control (NDM-TCP) system.

## Features

### 🎯 Model Management
- **Create New Models**: Configure hidden size, manifold size, and learning rate
- **Save/Load Models**: Persistent storage with metadata
- **Model Validation**: Comprehensive validation checks for loaded models
- **Real-time Statistics**: Monitor controller metrics (CWND, entropy, plasticity, etc.)

### 🧠 Training
- **Configurable Training**: Set epochs, steps, RTT ranges, noise levels, and congestion ranges
- **Real-time Progress**: Live training progress with loss and metrics display
- **Background Training**: Non-blocking training with stop capability
- **Training Logs**: Detailed logging of training progress

### 📊 Real-time Monitoring
- **Live Network Metrics**: Real-time RTT, loss rate, throughput, bandwidth
- **Entropy Analysis**: Shannon entropy, noise ratio, congestion confidence
- **TCP Parameters**: CWND, SS threshold, pacing rate
- **Historical Data**: Last 10 updates with color-coded indicators
- **Configurable Update Rate**: Adjustable monitoring interval

### 🎯 Decision Making
- **Intelligent Traffic Analysis**: Automatic decision making based on network conditions
- **Action Types**:
  - `REDUCE_RATE`: High congestion detected
  - `MAINTAIN`: Network noise detected, maintain current rate
  - `INCREASE_RATE`: Good network conditions
  - `REDUCE_WINDOW`: High latency detected
  - `NORMAL`: No action needed
- **Decision Logging**: Complete history with timestamp, action, reason, and severity
- **Export Capability**: Export decisions to JSON or CSV

### 📝 System Logging
- **Comprehensive Logging**: All system events with timestamps and severity levels
- **Color-coded Messages**: Info (blue), Success (green), Warning (orange), Error (red)
- **Auto-scroll**: Optional auto-scrolling to latest messages
- **Export Logs**: Export to text or JSON format

## Installation

### Prerequisites

1. **Python 3.8+**
2. **C Compiler** (gcc/clang)
3. **Required Python packages**

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Compile the C Library

The NDM-TCP system requires a compiled C library. Follow the instructions for your platform:

#### Linux
```bash
gcc -shared -fPIC -o ndm_tcp.so ndm_tcp.c -lm -O3 -fopenmp
```

#### macOS
```bash
gcc -shared -fPIC -o ndm_tcp.dylib ndm_tcp.c -lm -O3 -Xpreprocessor -fopenmp -lomp
```

#### Windows
```bash
gcc -shared -o ndm_tcp.dll ndm_tcp.c -lm -O3
```

### Step 3: Verify Installation

Make sure the following files are in the same directory:
- `ndm_tcp_gui.py` (this GUI application)
- `ndm_tcp.py` (Python wrapper)
- `ndm_tcp.c` (C implementation)
- `ndm_tcp.so` / `ndm_tcp.dylib` / `ndm_tcp.dll` (compiled library)

## Usage

### Starting the Application

```bash
python ndm_tcp_gui.py
```

### Quick Start Guide

#### 1. Create or Load a Model

**Option A: Create New Model**
1. Go to "Model Management" tab
2. Configure parameters:
   - Hidden Size: Number of hidden neurons (default: 64)
   - Manifold Size: Associative memory size (default: 32)
   - Learning Rate: Training step size (default: 0.01)
3. Click "Create New Model"

**Option B: Load Existing Model**
1. Click "Load Model"
2. Select a `.ndm` file
3. Model will be validated automatically

#### 2. Validate the Model

1. Click "Validate Model" to run validation checks
2. Checks include:
   - Output values are finite
   - Entropy values are in valid range [0,1]
   - TCP parameters are positive
   - Network metrics are reasonable

#### 3. Train the Model

1. Go to "Training" tab
2. Configure training parameters:
   - **Epochs**: Number of training iterations (default: 100)
   - **Steps per Epoch**: Training steps per epoch (default: 50)
   - **Base RTT**: Baseline round-trip time in ms (default: 50)
   - **Noise Range**: Random noise variation (default: 0.1 - 0.5)
   - **Congestion Range**: Congestion severity (default: 0.0 - 0.8)
3. Click "Start Training"
4. Monitor progress in real-time
5. Training can be stopped at any time with "Stop Training"

#### 4. Monitor Real-time Traffic

1. Go to "Real-Time Monitoring" tab
2. Set update interval (default: 1000ms)
3. Click "Start Monitoring"
4. Observe:
   - Current network metrics in the table
   - Color-coded indicators (green = good, yellow = ok, red = poor)
   - Recent history of last 10 updates
5. Click "Stop Monitoring" when done

#### 5. Review Decisions

1. Go to "Decision Log" tab
2. View all automated decisions made by the controller
3. Each decision shows:
   - Timestamp
   - Action taken (REDUCE_RATE, INCREASE_RATE, etc.)
   - Reason for the decision
   - Severity level
4. Export decisions using "Export Decisions" button

#### 6. Check System Logs

1. Go to "System Log" tab
2. Review all system events
3. Use "Auto-scroll" to follow latest messages
4. Export logs using "Export Log" button

#### 7. Save Your Model

1. After training, go to "Model Management" tab
2. Click "Save Model"
3. Choose location and filename (`.ndm` extension)
4. Model metadata and training history will be saved

## Understanding the Interface

### Model Management Tab

**Model Information Group**
- Shows current model status (loaded/saved/created)
- Displays file path, creation date, and parameters

**Create New Model Group**
- Configure neural network architecture
- Hidden Size: 16-256 neurons
- Manifold Size: 8-128 dimensions
- Learning Rate: 0.0001-0.1

**Current Controller Statistics**
- Real-time metrics updated every 2 seconds
- CWND: Congestion window size
- SS Threshold: Slow start threshold
- Pacing Rate: Current data transmission rate
- Avg Entropy: Shannon entropy (noise detection)
- Plasticity: Network adaptability (0=rigid, 1=fluid)
- Weight Velocity: Rate of neural weight changes

### Training Tab

**Training Configuration**
- **Epochs**: More epochs = better learning, but slower
- **Steps per Epoch**: More steps = more diverse training
- **Noise Range**: Simulates random network fluctuations
- **Congestion Range**: Simulates real network congestion

**Training Metrics**
- **Loss**: Training error (lower is better)
- **Reward**: Network performance score (higher is better)
- **CWND**: Current congestion window
- **Entropy**: Noise detection metric
- **Plasticity**: Network adaptability

### Monitoring Tab

**Network Metrics Table**
- **RTT**: Round-trip time (lower is better)
  - Green: < 60ms (good)
  - Yellow: 60-100ms (ok)
  - Red: > 100ms (poor)
- **Loss Rate**: Packet loss percentage (lower is better)
  - Green: < 1% (good)
  - Yellow: 1-5% (ok)
  - Red: > 5% (poor)
- **Throughput**: Actual data rate
- **Shannon Entropy**: Noise detection (0-1)
- **Noise Ratio**: Estimated noise vs signal
- **Congestion Confidence**: Confidence in real congestion (0-1)

### Decision Making Logic

The controller automatically makes decisions based on:

1. **High Congestion** (REDUCE_RATE)
   - Congestion confidence > 70%
   - Packet loss rate > 5%
   - Action: Reduce transmission rate

2. **Network Noise** (MAINTAIN)
   - Noise ratio > 60%
   - Action: Maintain current rate (don't overreact to noise)

3. **Good Conditions** (INCREASE_RATE)
   - Packet loss < 1%
   - RTT < 60ms
   - Action: Increase transmission rate

4. **High Latency** (REDUCE_WINDOW)
   - RTT > 100ms
   - Action: Reduce congestion window

## File Formats

### Model Files (.ndm)

Model files are Python pickle files containing:
```python
{
    'metadata': {
        'created': '2024-02-06T10:30:00',
        'hidden_size': 64,
        'manifold_size': 32,
        'learning_rate': 0.01,
        'input_size': 15,
        'output_size': 3
    },
    'history': {
        'cwnd': [...],
        'entropy': [...],
        'reward': [...]
    },
    'version': '1.0'
}
```

### Decision Exports

**JSON Format:**
```json
[
  {
    "timestamp": "10:30:45",
    "action": "REDUCE_RATE",
    "reason": "High congestion detected (confidence: 0.85, loss: 0.067)",
    "severity": "warning"
  }
]
```

**CSV Format:**
```csv
timestamp,action,reason,severity
10:30:45,REDUCE_RATE,"High congestion detected",warning
```

### Log Exports

**Text Format:**
```
[2024-02-06 10:30:45] [INFO] Model created successfully
[2024-02-06 10:31:20] [SUCCESS] Training completed: 100 epochs
[2024-02-06 10:35:10] [WARNING] High latency detected
```

**JSON Format:**
```json
[
  {
    "timestamp": "2024-02-06 10:30:45",
    "level": "INFO",
    "message": "Model created successfully"
  }
]
```

## Troubleshooting

### "NDM-TCP module is not available"
- Ensure `ndm_tcp.py` is in the same directory as `ndm_tcp_gui.py`
- Ensure the compiled C library (`.so`/`.dylib`/`.dll`) is present
- Check that the C library was compiled correctly

### "Failed to create controller"
- Check that the C library is properly compiled
- Verify that all dependencies are installed
- Check system logs for detailed error messages

### Training not starting
- Ensure a model is created or loaded
- Check that no other training is in progress
- Verify training parameters are valid

### Monitoring shows N/A values
- Start monitoring by clicking "Start Monitoring"
- Ensure a model is loaded
- Check update interval setting

### Model validation fails
- The model may be corrupted
- Try creating a new model
- Check for valid parameter ranges

## Advanced Features

### Custom Training Scenarios

You can customize training by adjusting:
- **High Noise Training**: Set noise range to 0.5-0.9 to train for noisy environments
- **High Congestion Training**: Set congestion range to 0.6-1.0 to train for congested networks
- **Balanced Training**: Use default ranges for general-purpose models

### Real-time Integration

For production use, replace the simulation in `MonitoringThread.run()` with actual network measurements:

```python
# Replace this:
metrics = simulate_network_condition(...)

# With this:
metrics = TCPMetrics(
    current_rtt=get_actual_rtt(),
    packet_loss_rate=get_actual_loss_rate(),
    bandwidth_estimate=get_actual_bandwidth(),
    # ... other actual measurements
)
```

## Technical Details

### Architecture
- **Frontend**: PySide6 (Qt for Python)
- **Backend**: NDM-TCP neural network (C implementation)
- **Threading**: QThread for non-blocking operations
- **Data Flow**: Real-time metrics → Controller → Actions → Decisions

### Performance
- Monitoring updates: 100-5000ms intervals
- Training: ~10-50 steps/second (depends on hardware)
- Model size: Typical ~50-200KB (depends on architecture)

### Security
- Input validation on all network metrics
- Bounds checking in C library
- Safe model file handling with error checking

## License

GPL V3 - See LICENSE file for details

## Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review system logs in the application
3. Check that all dependencies are correctly installed
4. Ensure the C library is properly compiled for your platform

## Credits

NDM-TCP: Neural Differential Manifolds for TCP Congestion Control
- Entropy-Aware Traffic Shaping
- Continuous Weight Evolution
- Real-time Neuroplasticity
