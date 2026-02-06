#!/usr/bin/env python3
"""
NDM-TCP GUI Application
PySide6 interface for Neural Differential Manifolds TCP Congestion Control

Features:
- Real-time network traffic monitoring
- Model training with progress tracking
- Model save/load with validation
- Decision making and traffic analysis
- Comprehensive logging system
- Network statistics visualization
"""

import sys
import os
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import numpy as np
from collections import deque

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QGroupBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QFileDialog, QMessageBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QLineEdit, QCheckBox, QSplitter,
    QStatusBar, QGridLayout
)
from PySide6.QtCore import QThread, Signal, QTimer, Qt, QMutex
from PySide6.QtGui import QFont, QColor, QPalette

# Import the NDM-TCP module
try:
    from ndm_tcp import (
        NDMTCPController, TCPMetrics, simulate_network_condition,
        calculate_reward
    )
    NDM_AVAILABLE = True
except ImportError as e:
    NDM_AVAILABLE = False
    print(f"WARNING: Could not import ndm_tcp module: {e}")
    print("Make sure ndm_tcp.py and the compiled C library are in the same directory")


# ============================================================================
# TRAINING THREAD
# ============================================================================

class TrainingThread(QThread):
    """Background thread for model training"""
    
    progress = Signal(int, float, str)  # epoch, loss, metrics
    finished = Signal(str)  # completion message
    error = Signal(str)  # error message
    
    def __init__(self, controller: 'NDMTCPController', config: Dict):
        super().__init__()
        self.controller = controller
        self.config = config
        self.running = True
        self.mutex = QMutex()
    
    def stop(self):
        """Stop training"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
    
    def run(self):
        """Run training loop"""
        try:
            epochs = self.config['epochs']
            steps_per_epoch = self.config['steps_per_epoch']
            base_rtt = self.config['base_rtt']
            noise_range = self.config['noise_range']
            congestion_range = self.config['congestion_range']
            
            total_steps = epochs * steps_per_epoch
            step_count = 0
            
            for epoch in range(epochs):
                self.mutex.lock()
                if not self.running:
                    self.mutex.unlock()
                    break
                self.mutex.unlock()
                
                epoch_losses = []
                epoch_rewards = []
                
                for step in range(steps_per_epoch):
                    # Generate random network conditions
                    noise_level = np.random.uniform(*noise_range)
                    congestion_level = np.random.uniform(*congestion_range)
                    
                    # Simulate network
                    metrics = simulate_network_condition(
                        base_rtt=base_rtt,
                        congestion_level=congestion_level,
                        noise_level=noise_level
                    )
                    
                    # Calculate reward
                    reward = calculate_reward(metrics, metrics.throughput, base_rtt)
                    
                    # Training step
                    loss = self.controller.train_step(metrics, reward)
                    
                    epoch_losses.append(loss)
                    epoch_rewards.append(reward)
                    
                    step_count += 1
                
                # Compute epoch statistics
                avg_loss = np.mean(epoch_losses)
                avg_reward = np.mean(epoch_rewards)
                
                # Build metrics string
                metrics_str = (
                    f"Loss: {avg_loss:.4f} | "
                    f"Reward: {avg_reward:.2f} | "
                    f"CWND: {self.controller.current_cwnd:.2f} | "
                    f"Entropy: {self.controller.avg_entropy:.4f} | "
                    f"Plasticity: {self.controller.avg_plasticity:.4f}"
                )
                
                # Emit progress
                progress_pct = int((epoch + 1) / epochs * 100)
                self.progress.emit(epoch + 1, avg_loss, metrics_str)
                
                # Small delay to prevent UI freezing
                self.msleep(10)
            
            self.finished.emit(f"Training completed: {epochs} epochs, {total_steps} steps")
            
        except Exception as e:
            self.error.emit(f"Training error: {str(e)}")


# ============================================================================
# MONITORING THREAD
# ============================================================================

class MonitoringThread(QThread):
    """Background thread for real-time network monitoring"""
    
    update = Signal(dict)  # Network metrics update
    decision = Signal(dict)  # Decision making result
    
    def __init__(self, controller: 'NDMTCPController', config: Dict):
        super().__init__()
        self.controller = controller
        self.config = config
        self.running = True
        self.mutex = QMutex()
    
    def stop(self):
        """Stop monitoring"""
        self.mutex.lock()
        self.running = False
        self.mutex.unlock()
    
    def run(self):
        """Run monitoring loop"""
        interval_ms = self.config.get('interval_ms', 1000)
        
        while True:
            self.mutex.lock()
            if not self.running:
                self.mutex.unlock()
                break
            self.mutex.unlock()
            
            try:
                # Simulate network conditions (in real app, get from actual network)
                noise_level = np.random.uniform(0.1, 0.3)
                congestion_level = np.random.uniform(0.0, 0.5)
                
                metrics = simulate_network_condition(
                    base_rtt=50.0,
                    congestion_level=congestion_level,
                    noise_level=noise_level
                )
                
                # Get controller actions
                actions = self.controller.forward(metrics)
                
                # Prepare data for UI
                data = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'rtt': metrics.current_rtt,
                    'loss_rate': metrics.packet_loss_rate,
                    'throughput': metrics.throughput,
                    'bandwidth': metrics.bandwidth_estimate,
                    'entropy': actions['entropy'],
                    'noise_ratio': actions['noise_ratio'],
                    'congestion_confidence': actions['congestion_confidence'],
                    'cwnd': self.controller.current_cwnd,
                    'ssthresh': self.controller.current_ssthresh,
                    'pacing_rate': self.controller.current_pacing_rate,
                    'cwnd_delta': actions['cwnd_delta'],
                    'congestion_level': congestion_level,
                }
                
                self.update.emit(data)
                
                # Decision making logic
                decision_data = self._make_decision(metrics, actions, data)
                self.decision.emit(decision_data)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
            
            self.msleep(interval_ms)
    
    def _make_decision(self, metrics: TCPMetrics, actions: Dict, data: Dict) -> Dict:
        """Make traffic management decisions"""
        decision = {
            'timestamp': data['timestamp'],
            'action': 'NORMAL',
            'reason': '',
            'severity': 'info'
        }
        
        # Check for high congestion
        if actions['congestion_confidence'] > 0.7 and metrics.packet_loss_rate > 0.05:
            decision['action'] = 'REDUCE_RATE'
            decision['reason'] = f"High congestion detected (confidence: {actions['congestion_confidence']:.2f}, loss: {metrics.packet_loss_rate:.3f})"
            decision['severity'] = 'warning'
        
        # Check for network noise
        elif actions['noise_ratio'] > 0.6:
            decision['action'] = 'MAINTAIN'
            decision['reason'] = f"Network noise detected (ratio: {actions['noise_ratio']:.2f}), maintaining current rate"
            decision['severity'] = 'info'
        
        # Check for good conditions
        elif metrics.packet_loss_rate < 0.01 and metrics.current_rtt < 60:
            decision['action'] = 'INCREASE_RATE'
            decision['reason'] = f"Good network conditions (RTT: {metrics.current_rtt:.1f}ms, loss: {metrics.packet_loss_rate:.3f})"
            decision['severity'] = 'success'
        
        # Check for high latency
        elif metrics.current_rtt > 100:
            decision['action'] = 'REDUCE_WINDOW'
            decision['reason'] = f"High latency detected (RTT: {metrics.current_rtt:.1f}ms)"
            decision['severity'] = 'warning'
        
        return decision


# ============================================================================
# MAIN WINDOW
# ============================================================================

class NDMTCPMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.controller: Optional[NDMTCPController] = None
        self.training_thread: Optional[TrainingThread] = None
        self.monitoring_thread: Optional[MonitoringThread] = None
        
        # Data storage
        self.metrics_history = deque(maxlen=100)
        self.decision_history = deque(maxlen=50)
        self.log_entries = []
        
        # Model metadata
        self.model_path: Optional[Path] = None
        self.model_metadata: Dict = {}
        
        self.init_ui()
        self.apply_stylesheet()
        
        # Initialize with default controller
        if NDM_AVAILABLE:
            self.create_new_controller()
        else:
            self.log_message("ERROR: NDM-TCP module not available", "error")
    
    def init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("NDM-TCP Network Controller")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Tab 1: Model Management
        tabs.addTab(self.create_model_tab(), "Model Management")
        
        # Tab 2: Training
        tabs.addTab(self.create_training_tab(), "Training")
        
        # Tab 3: Monitoring
        tabs.addTab(self.create_monitoring_tab(), "Real-Time Monitoring")
        
        # Tab 4: Decision Log
        tabs.addTab(self.create_decision_tab(), "Decision Log")
        
        # Tab 5: System Log
        tabs.addTab(self.create_log_tab(), "System Log")
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def create_model_tab(self) -> QWidget:
        """Create model management tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Model info group
        info_group = QGroupBox("Model Information")
        info_layout = QGridLayout()
        info_group.setLayout(info_layout)
        
        self.model_status_label = QLabel("No model loaded")
        self.model_status_label.setFont(QFont("Monospace", 10, QFont.Bold))
        info_layout.addWidget(QLabel("Status:"), 0, 0)
        info_layout.addWidget(self.model_status_label, 0, 1)
        
        self.model_path_label = QLabel("N/A")
        info_layout.addWidget(QLabel("Path:"), 1, 0)
        info_layout.addWidget(self.model_path_label, 1, 1)
        
        self.model_created_label = QLabel("N/A")
        info_layout.addWidget(QLabel("Created:"), 2, 0)
        info_layout.addWidget(self.model_created_label, 2, 1)
        
        self.model_params_label = QLabel("N/A")
        info_layout.addWidget(QLabel("Parameters:"), 3, 0)
        info_layout.addWidget(self.model_params_label, 3, 1)
        
        layout.addWidget(info_group)
        
        # New model group
        new_group = QGroupBox("Create New Model")
        new_layout = QGridLayout()
        new_group.setLayout(new_layout)
        
        new_layout.addWidget(QLabel("Hidden Size:"), 0, 0)
        self.hidden_size_spin = QSpinBox()
        self.hidden_size_spin.setRange(16, 256)
        self.hidden_size_spin.setValue(64)
        new_layout.addWidget(self.hidden_size_spin, 0, 1)
        
        new_layout.addWidget(QLabel("Manifold Size:"), 1, 0)
        self.manifold_size_spin = QSpinBox()
        self.manifold_size_spin.setRange(8, 128)
        self.manifold_size_spin.setValue(32)
        new_layout.addWidget(self.manifold_size_spin, 1, 1)
        
        new_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.0001, 0.1)
        self.learning_rate_spin.setValue(0.01)
        self.learning_rate_spin.setDecimals(4)
        self.learning_rate_spin.setSingleStep(0.001)
        new_layout.addWidget(self.learning_rate_spin, 2, 1)
        
        create_btn = QPushButton("Create New Model")
        create_btn.clicked.connect(self.create_new_controller)
        new_layout.addWidget(create_btn, 3, 0, 1, 2)
        
        layout.addWidget(new_group)
        
        # Save/Load group
        file_group = QGroupBox("Save / Load Model")
        file_layout = QHBoxLayout()
        file_group.setLayout(file_layout)
        
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self.save_model)
        file_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self.load_model)
        file_layout.addWidget(load_btn)
        
        validate_btn = QPushButton("Validate Model")
        validate_btn.clicked.connect(self.validate_model)
        file_layout.addWidget(validate_btn)
        
        layout.addWidget(file_group)
        
        # Controller stats
        stats_group = QGroupBox("Current Controller Statistics")
        stats_layout = QGridLayout()
        stats_group.setLayout(stats_layout)
        
        self.stats_labels = {}
        stats_fields = [
            ('CWND', 'cwnd'),
            ('SS Threshold', 'ssthresh'),
            ('Pacing Rate', 'pacing_rate'),
            ('Avg Entropy', 'entropy'),
            ('Plasticity', 'plasticity'),
            ('Weight Velocity', 'velocity'),
        ]
        
        for i, (label, key) in enumerate(stats_fields):
            row = i // 2
            col = (i % 2) * 2
            stats_layout.addWidget(QLabel(f"{label}:"), row, col)
            value_label = QLabel("N/A")
            value_label.setFont(QFont("Monospace", 9))
            self.stats_labels[key] = value_label
            stats_layout.addWidget(value_label, row, col + 1)
        
        layout.addWidget(stats_group)
        
        # Update timer
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_controller_stats)
        self.stats_timer.start(2000)  # Update every 2 seconds
        
        layout.addStretch()
        return widget
    
    def create_training_tab(self) -> QWidget:
        """Create training tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QGridLayout()
        config_group.setLayout(config_layout)
        
        config_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)
        config_layout.addWidget(self.epochs_spin, 0, 1)
        
        config_layout.addWidget(QLabel("Steps per Epoch:"), 1, 0)
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 1000)
        self.steps_spin.setValue(50)
        config_layout.addWidget(self.steps_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Base RTT (ms):"), 2, 0)
        self.base_rtt_spin = QDoubleSpinBox()
        self.base_rtt_spin.setRange(1.0, 500.0)
        self.base_rtt_spin.setValue(50.0)
        config_layout.addWidget(self.base_rtt_spin, 2, 1)
        
        config_layout.addWidget(QLabel("Noise Range:"), 3, 0)
        noise_layout = QHBoxLayout()
        self.noise_min_spin = QDoubleSpinBox()
        self.noise_min_spin.setRange(0.0, 1.0)
        self.noise_min_spin.setValue(0.1)
        self.noise_min_spin.setSingleStep(0.1)
        self.noise_max_spin = QDoubleSpinBox()
        self.noise_max_spin.setRange(0.0, 1.0)
        self.noise_max_spin.setValue(0.5)
        self.noise_max_spin.setSingleStep(0.1)
        noise_layout.addWidget(self.noise_min_spin)
        noise_layout.addWidget(QLabel("-"))
        noise_layout.addWidget(self.noise_max_spin)
        config_layout.addLayout(noise_layout, 3, 1)
        
        config_layout.addWidget(QLabel("Congestion Range:"), 4, 0)
        cong_layout = QHBoxLayout()
        self.cong_min_spin = QDoubleSpinBox()
        self.cong_min_spin.setRange(0.0, 1.0)
        self.cong_min_spin.setValue(0.0)
        self.cong_min_spin.setSingleStep(0.1)
        self.cong_max_spin = QDoubleSpinBox()
        self.cong_max_spin.setRange(0.0, 1.0)
        self.cong_max_spin.setValue(0.8)
        self.cong_max_spin.setSingleStep(0.1)
        cong_layout.addWidget(self.cong_min_spin)
        cong_layout.addWidget(QLabel("-"))
        cong_layout.addWidget(self.cong_max_spin)
        config_layout.addLayout(cong_layout, 4, 1)
        
        layout.addWidget(config_group)
        
        # Training controls
        control_layout = QHBoxLayout()
        
        self.start_train_btn = QPushButton("Start Training")
        self.start_train_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_train_btn)
        
        self.stop_train_btn = QPushButton("Stop Training")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        control_layout.addWidget(self.stop_train_btn)
        
        layout.addLayout(control_layout)
        
        # Progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Not started")
        progress_layout.addWidget(self.progress_label)
        
        self.metrics_label = QLabel("")
        self.metrics_label.setFont(QFont("Monospace", 9))
        progress_layout.addWidget(self.metrics_label)
        
        layout.addWidget(progress_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.training_log = QTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setFont(QFont("Monospace", 9))
        log_layout.addWidget(self.training_log)
        
        layout.addWidget(log_group)
        
        return widget
    
    def create_monitoring_tab(self) -> QWidget:
        """Create real-time monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.start_monitor_btn = QPushButton("Start Monitoring")
        self.start_monitor_btn.clicked.connect(self.start_monitoring)
        control_layout.addWidget(self.start_monitor_btn)
        
        self.stop_monitor_btn = QPushButton("Stop Monitoring")
        self.stop_monitor_btn.clicked.connect(self.stop_monitoring)
        self.stop_monitor_btn.setEnabled(False)
        control_layout.addWidget(self.stop_monitor_btn)
        
        control_layout.addWidget(QLabel("Update Interval (ms):"))
        self.monitor_interval_spin = QSpinBox()
        self.monitor_interval_spin.setRange(100, 5000)
        self.monitor_interval_spin.setValue(1000)
        control_layout.addWidget(self.monitor_interval_spin)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Metrics table
        metrics_group = QGroupBox("Current Network Metrics")
        metrics_layout = QVBoxLayout()
        metrics_group.setLayout(metrics_layout)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setRowCount(11)
        
        metric_names = [
            "Timestamp", "RTT (ms)", "Loss Rate", "Throughput (Mbps)",
            "Bandwidth (Mbps)", "Shannon Entropy", "Noise Ratio",
            "Congestion Confidence", "CWND", "SS Threshold", "Pacing Rate (Mbps)"
        ]
        
        for i, name in enumerate(metric_names):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem("N/A"))
        
        metrics_layout.addWidget(self.metrics_table)
        layout.addWidget(metrics_group)
        
        # History display
        history_group = QGroupBox("Recent History (Last 10 Updates)")
        history_layout = QVBoxLayout()
        history_group.setLayout(history_layout)
        
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setFont(QFont("Monospace", 8))
        self.history_text.setMaximumHeight(200)
        history_layout.addWidget(self.history_text)
        
        layout.addWidget(history_group)
        
        return widget
    
    def create_decision_tab(self) -> QWidget:
        """Create decision log tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        control_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Decision Log")
        clear_btn.clicked.connect(self.clear_decision_log)
        control_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("Export Decisions")
        export_btn.clicked.connect(self.export_decisions)
        control_layout.addWidget(export_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Decision table
        self.decision_table = QTableWidget()
        self.decision_table.setColumnCount(4)
        self.decision_table.setHorizontalHeaderLabels([
            "Timestamp", "Action", "Reason", "Severity"
        ])
        header = self.decision_table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        layout.addWidget(self.decision_table)
        
        return widget
    
    def create_log_tab(self) -> QWidget:
        """Create system log tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Controls
        control_layout = QHBoxLayout()
        
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.clear_log)
        control_layout.addWidget(clear_btn)
        
        export_btn = QPushButton("Export Log")
        export_btn.clicked.connect(self.export_log)
        control_layout.addWidget(export_btn)
        
        self.auto_scroll_check = QCheckBox("Auto-scroll")
        self.auto_scroll_check.setChecked(True)
        control_layout.addWidget(self.auto_scroll_check)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Log display
        self.system_log = QTextEdit()
        self.system_log.setReadOnly(True)
        self.system_log.setFont(QFont("Monospace", 9))
        layout.addWidget(self.system_log)
        
        return widget
    
    def apply_stylesheet(self):
        """Apply custom stylesheet"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
    
    # ========================================================================
    # MODEL MANAGEMENT
    # ========================================================================
    
    def create_new_controller(self):
        """Create a new NDM-TCP controller"""
        if not NDM_AVAILABLE:
            QMessageBox.critical(self, "Error", "NDM-TCP module is not available")
            return
        
        try:
            hidden_size = self.hidden_size_spin.value()
            manifold_size = self.manifold_size_spin.value()
            learning_rate = self.learning_rate_spin.value()
            
            self.controller = NDMTCPController(
                input_size=15,
                hidden_size=hidden_size,
                output_size=3,
                manifold_size=manifold_size,
                learning_rate=learning_rate
            )
            
            self.model_metadata = {
                'created': datetime.now().isoformat(),
                'hidden_size': hidden_size,
                'manifold_size': manifold_size,
                'learning_rate': learning_rate,
                'input_size': 15,
                'output_size': 3,
            }
            
            self.model_status_label.setText("Model created (not saved)")
            self.model_status_label.setStyleSheet("color: green;")
            self.model_created_label.setText(self.model_metadata['created'])
            self.model_params_label.setText(
                f"H={hidden_size}, M={manifold_size}, LR={learning_rate}"
            )
            
            self.log_message(
                f"Created new controller: hidden={hidden_size}, "
                f"manifold={manifold_size}, lr={learning_rate}",
                "success"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create controller: {str(e)}")
            self.log_message(f"Controller creation failed: {str(e)}", "error")
    
    def save_model(self):
        """Save model to file"""
        if self.controller is None:
            QMessageBox.warning(self, "Warning", "No model to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", "", "NDM-TCP Model (*.ndm);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # In a real implementation, you would serialize the C++ weights
            # For now, we save the metadata and Python state
            save_data = {
                'metadata': self.model_metadata,
                'history': {
                    'cwnd': self.controller.get_cwnd_history().tolist(),
                    'entropy': self.controller.get_entropy_history().tolist(),
                    'reward': self.controller.get_reward_history().tolist(),
                },
                'version': '1.0'
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.model_path = Path(file_path)
            self.model_path_label.setText(str(self.model_path))
            self.model_status_label.setText("Model saved")
            
            self.log_message(f"Model saved to: {file_path}", "success")
            QMessageBox.information(self, "Success", "Model saved successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")
            self.log_message(f"Model save failed: {str(e)}", "error")
    
    def load_model(self):
        """Load model from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "NDM-TCP Model (*.ndm);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # Validate file format
            if 'metadata' not in save_data or 'version' not in save_data:
                raise ValueError("Invalid model file format")
            
            metadata = save_data['metadata']
            
            # Create controller with loaded parameters
            self.controller = NDMTCPController(
                input_size=metadata.get('input_size', 15),
                hidden_size=metadata['hidden_size'],
                output_size=metadata.get('output_size', 3),
                manifold_size=metadata['manifold_size'],
                learning_rate=metadata['learning_rate']
            )
            
            self.model_metadata = metadata
            self.model_path = Path(file_path)
            
            # Update UI
            self.model_status_label.setText("Model loaded")
            self.model_status_label.setStyleSheet("color: blue;")
            self.model_path_label.setText(str(self.model_path))
            self.model_created_label.setText(metadata['created'])
            self.model_params_label.setText(
                f"H={metadata['hidden_size']}, "
                f"M={metadata['manifold_size']}, "
                f"LR={metadata['learning_rate']}"
            )
            
            self.log_message(f"Model loaded from: {file_path}", "success")
            QMessageBox.information(self, "Success", "Model loaded successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.log_message(f"Model load failed: {str(e)}", "error")
    
    def validate_model(self):
        """Validate the loaded model"""
        if self.controller is None:
            QMessageBox.warning(self, "Warning", "No model to validate")
            return
        
        try:
            # Run validation tests
            test_metrics = simulate_network_condition(50.0, 0.3, 0.2)
            actions = self.controller.forward(test_metrics)
            
            # Check if outputs are valid
            checks = []
            checks.append(("CWND delta is finite", np.isfinite(actions['cwnd_delta'])))
            checks.append(("Entropy in [0,1]", 0 <= actions['entropy'] <= 1))
            checks.append(("Noise ratio in [0,1]", 0 <= actions['noise_ratio'] <= 1))
            checks.append(("Confidence in [0,1]", 0 <= actions['congestion_confidence'] <= 1))
            checks.append(("Current CWND > 0", self.controller.current_cwnd > 0))
            checks.append(("Pacing rate > 0", self.controller.current_pacing_rate > 0))
            
            all_passed = all(check[1] for check in checks)
            
            msg = "Model Validation Results:\n\n"
            for name, passed in checks:
                status = "✓ PASS" if passed else "✗ FAIL"
                msg += f"{status}: {name}\n"
            
            if all_passed:
                msg += "\n✓ All validation checks passed!"
                self.log_message("Model validation: PASSED", "success")
                QMessageBox.information(self, "Validation Success", msg)
            else:
                msg += "\n✗ Some validation checks failed!"
                self.log_message("Model validation: FAILED", "error")
                QMessageBox.warning(self, "Validation Failed", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Validation error: {str(e)}")
            self.log_message(f"Model validation error: {str(e)}", "error")
    
    def update_controller_stats(self):
        """Update controller statistics display"""
        if self.controller is None:
            return
        
        try:
            self.stats_labels['cwnd'].setText(f"{self.controller.current_cwnd:.2f}")
            self.stats_labels['ssthresh'].setText(f"{self.controller.current_ssthresh:.2f}")
            self.stats_labels['pacing_rate'].setText(f"{self.controller.current_pacing_rate:.2f}")
            self.stats_labels['entropy'].setText(f"{self.controller.avg_entropy:.4f}")
            self.stats_labels['plasticity'].setText(f"{self.controller.avg_plasticity:.4f}")
            self.stats_labels['velocity'].setText(f"{self.controller.avg_weight_velocity:.6f}")
        except:
            pass
    
    # ========================================================================
    # TRAINING
    # ========================================================================
    
    def start_training(self):
        """Start model training"""
        if self.controller is None:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return
        
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Training already in progress")
            return
        
        config = {
            'epochs': self.epochs_spin.value(),
            'steps_per_epoch': self.steps_spin.value(),
            'base_rtt': self.base_rtt_spin.value(),
            'noise_range': (self.noise_min_spin.value(), self.noise_max_spin.value()),
            'congestion_range': (self.cong_min_spin.value(), self.cong_max_spin.value()),
        }
        
        self.training_thread = TrainingThread(self.controller, config)
        self.training_thread.progress.connect(self.on_training_progress)
        self.training_thread.finished.connect(self.on_training_finished)
        self.training_thread.error.connect(self.on_training_error)
        
        self.training_thread.start()
        
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        
        self.log_message(f"Training started: {config['epochs']} epochs", "info")
        self.training_log.append(f"\n{'='*60}\nTraining started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")
    
    def stop_training(self):
        """Stop training"""
        if self.training_thread:
            self.training_thread.stop()
            self.log_message("Training stop requested", "warning")
    
    def on_training_progress(self, epoch: int, loss: float, metrics: str):
        """Handle training progress update"""
        total_epochs = self.epochs_spin.value()
        progress = int(epoch / total_epochs * 100)
        
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Epoch {epoch}/{total_epochs}")
        self.metrics_label.setText(metrics)
        
        log_msg = f"Epoch {epoch:4d} | {metrics}"
        self.training_log.append(log_msg)
        
        if self.auto_scroll_check.isChecked():
            self.training_log.verticalScrollBar().setValue(
                self.training_log.verticalScrollBar().maximum()
            )
    
    def on_training_finished(self, message: str):
        """Handle training completion"""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        
        self.log_message(message, "success")
        self.training_log.append(f"\n{message}\n{'='*60}\n")
        
        QMessageBox.information(self, "Training Complete", message)
    
    def on_training_error(self, error: str):
        """Handle training error"""
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)
        
        self.log_message(f"Training error: {error}", "error")
        self.training_log.append(f"\nERROR: {error}\n")
        
        QMessageBox.critical(self, "Training Error", error)
    
    # ========================================================================
    # MONITORING
    # ========================================================================
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        if self.controller is None:
            QMessageBox.warning(self, "Warning", "No model loaded")
            return
        
        if self.monitoring_thread and self.monitoring_thread.isRunning():
            QMessageBox.warning(self, "Warning", "Monitoring already in progress")
            return
        
        config = {
            'interval_ms': self.monitor_interval_spin.value()
        }
        
        self.monitoring_thread = MonitoringThread(self.controller, config)
        self.monitoring_thread.update.connect(self.on_monitoring_update)
        self.monitoring_thread.decision.connect(self.on_decision_made)
        
        self.monitoring_thread.start()
        
        self.start_monitor_btn.setEnabled(False)
        self.stop_monitor_btn.setEnabled(True)
        
        self.log_message("Real-time monitoring started", "info")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if self.monitoring_thread:
            self.monitoring_thread.stop()
            self.monitoring_thread.wait()
            
            self.start_monitor_btn.setEnabled(True)
            self.stop_monitor_btn.setEnabled(False)
            
            self.log_message("Real-time monitoring stopped", "info")
    
    def on_monitoring_update(self, data: Dict):
        """Handle monitoring data update"""
        # Store in history
        self.metrics_history.append(data)
        
        # Update table
        values = [
            data['timestamp'],
            f"{data['rtt']:.2f}",
            f"{data['loss_rate']:.4f}",
            f"{data['throughput']:.2f}",
            f"{data['bandwidth']:.2f}",
            f"{data['entropy']:.4f}",
            f"{data['noise_ratio']:.4f}",
            f"{data['congestion_confidence']:.4f}",
            f"{data['cwnd']:.2f}",
            f"{data['ssthresh']:.2f}",
            f"{data['pacing_rate']:.2f}",
        ]
        
        for i, value in enumerate(values):
            self.metrics_table.item(i, 1).setText(value)
        
        # Color code based on conditions
        # RTT
        rtt_item = self.metrics_table.item(1, 1)
        if data['rtt'] < 60:
            rtt_item.setBackground(QColor(200, 255, 200))
        elif data['rtt'] > 100:
            rtt_item.setBackground(QColor(255, 200, 200))
        else:
            rtt_item.setBackground(QColor(255, 255, 200))
        
        # Loss rate
        loss_item = self.metrics_table.item(2, 1)
        if data['loss_rate'] < 0.01:
            loss_item.setBackground(QColor(200, 255, 200))
        elif data['loss_rate'] > 0.05:
            loss_item.setBackground(QColor(255, 200, 200))
        else:
            loss_item.setBackground(QColor(255, 255, 200))
        
        # Update history text
        history_lines = []
        for item in list(self.metrics_history)[-10:]:
            history_lines.append(
                f"{item['timestamp']} | RTT: {item['rtt']:6.2f}ms | "
                f"Loss: {item['loss_rate']:.4f} | "
                f"Tput: {item['throughput']:6.2f}Mbps | "
                f"Entropy: {item['entropy']:.3f}"
            )
        
        self.history_text.setText("\n".join(history_lines))
    
    def on_decision_made(self, decision: Dict):
        """Handle decision making result"""
        # Store in history
        self.decision_history.append(decision)
        
        # Add to decision table
        row = self.decision_table.rowCount()
        self.decision_table.insertRow(row)
        
        self.decision_table.setItem(row, 0, QTableWidgetItem(decision['timestamp']))
        self.decision_table.setItem(row, 1, QTableWidgetItem(decision['action']))
        self.decision_table.setItem(row, 2, QTableWidgetItem(decision['reason']))
        self.decision_table.setItem(row, 3, QTableWidgetItem(decision['severity']))
        
        # Color code by severity
        severity_colors = {
            'success': QColor(200, 255, 200),
            'info': QColor(200, 220, 255),
            'warning': QColor(255, 255, 200),
            'error': QColor(255, 200, 200),
        }
        
        color = severity_colors.get(decision['severity'], QColor(255, 255, 255))
        for col in range(4):
            self.decision_table.item(row, col).setBackground(color)
        
        # Auto-scroll to bottom
        self.decision_table.scrollToBottom()
        
        # Log important decisions
        if decision['severity'] in ['warning', 'error']:
            self.log_message(f"Decision: {decision['action']} - {decision['reason']}", decision['severity'])
    
    def clear_decision_log(self):
        """Clear decision log"""
        self.decision_table.setRowCount(0)
        self.decision_history.clear()
        self.log_message("Decision log cleared", "info")
    
    def export_decisions(self):
        """Export decision log to file"""
        if not self.decision_history:
            QMessageBox.information(self, "Info", "No decisions to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Decisions", "", "JSON Files (*.json);;CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(list(self.decision_history), f, indent=2)
            else:
                import csv
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['timestamp', 'action', 'reason', 'severity'])
                    writer.writeheader()
                    writer.writerows(self.decision_history)
            
            self.log_message(f"Decisions exported to: {file_path}", "success")
            QMessageBox.information(self, "Success", "Decisions exported successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    def log_message(self, message: str, level: str = "info"):
        """Add message to system log"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        level_colors = {
            'info': 'blue',
            'success': 'green',
            'warning': 'orange',
            'error': 'red',
        }
        
        color = level_colors.get(level, 'black')
        level_text = level.upper()
        
        log_entry = {
            'timestamp': timestamp,
            'level': level_text,
            'message': message
        }
        self.log_entries.append(log_entry)
        
        html_msg = f'<span style="color: {color};">[{timestamp}] [{level_text}] {message}</span>'
        self.system_log.append(html_msg)
        
        if self.auto_scroll_check.isChecked():
            self.system_log.verticalScrollBar().setValue(
                self.system_log.verticalScrollBar().maximum()
            )
        
        # Update status bar for important messages
        if level in ['warning', 'error']:
            self.status_bar.showMessage(f"{level_text}: {message}", 5000)
        elif level == 'success':
            self.status_bar.showMessage(message, 3000)
    
    def clear_log(self):
        """Clear system log"""
        self.system_log.clear()
        self.log_entries.clear()
        self.log_message("System log cleared", "info")
    
    def export_log(self):
        """Export system log to file"""
        if not self.log_entries:
            QMessageBox.information(self, "Info", "No log entries to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Log", "", "Text Files (*.txt);;JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(self.log_entries, f, indent=2)
            else:
                with open(file_path, 'w') as f:
                    for entry in self.log_entries:
                        f.write(f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}\n")
            
            self.log_message(f"Log exported to: {file_path}", "success")
            QMessageBox.information(self, "Success", "Log exported successfully")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export log: {str(e)}")
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop threads
        if self.monitoring_thread and self.monitoring_thread.isRunning():
            self.monitoring_thread.stop()
            self.monitoring_thread.wait()
        
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Training in Progress',
                'Training is in progress. Are you sure you want to quit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                self.training_thread.wait()
            else:
                event.ignore()
                return
        
        event.accept()


# ============================================================================
# MAIN
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NDM-TCP Network Controller")
    
    window = NDMTCPMainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()