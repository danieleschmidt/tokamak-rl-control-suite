"""
Real-time dashboard and visualization system for tokamak monitoring.

This module provides web-based dashboards, real-time plotting, and interactive
visualization tools for plasma control and performance monitoring.
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
except ImportError:
    plt = None
    animation = None
    FigureCanvasAgg = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    go = None
    px = None
    make_subplots = None

try:
    import dash
    from dash import dcc, html, Input, Output, State
except ImportError:
    dash = None
    dcc = None
    html = None
    Input = None
    Output = None
    State = None
import json
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from collections import deque
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    pd = None
import logging

from .physics import PlasmaState
from .business import PerformanceMetrics
from .analytics import AnomalyEvent, PredictionResult


@dataclass
class DashboardConfig:
    """Configuration for dashboard display."""
    update_interval: float = 1.0  # seconds
    history_length: int = 1000
    port: int = 8050
    debug: bool = False
    theme: str = "dark"
    auto_refresh: bool = True


class RealTimePlotter:
    """Real-time plotting with matplotlib for high-frequency data."""
    
    def __init__(self, figsize: tuple = (12, 8), max_points: int = 1000):
        self.figsize = figsize
        self.max_points = max_points
        
        # Data buffers
        self.time_data = deque(maxlen=max_points)
        self.plot_data = {}
        
        # Matplotlib setup
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 2, figsize=figsize)
        self.fig.suptitle('Tokamak Real-Time Monitoring', fontsize=16, color='white')
        
        # Animation
        self.animation = None
        self.is_running = False
        
        # Configure subplots
        self.axes[0, 0].set_title('Q-factor Profile', color='white')
        self.axes[0, 1].set_title('Shape Error', color='white')
        self.axes[1, 0].set_title('Plasma Beta', color='white')
        self.axes[1, 1].set_title('Control Signals', color='white')
        
        for ax in self.axes.flat:
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
    
    def add_data_point(self, timestamp: float, plasma_state: PlasmaState,
                      control_action: Optional[np.ndarray] = None) -> None:
        """Add new data point for plotting."""
        self.time_data.append(timestamp)
        
        # Store plasma state data
        if 'q_min' not in self.plot_data:
            self.plot_data['q_min'] = deque(maxlen=self.max_points)
            self.plot_data['shape_error'] = deque(maxlen=self.max_points)
            self.plot_data['beta'] = deque(maxlen=self.max_points)
            self.plot_data['disruption_prob'] = deque(maxlen=self.max_points)
            
        self.plot_data['q_min'].append(plasma_state.q_min)
        self.plot_data['shape_error'].append(plasma_state.shape_error)
        self.plot_data['beta'].append(plasma_state.plasma_beta)
        self.plot_data['disruption_prob'].append(plasma_state.disruption_probability)
        
        # Store control action data
        if control_action is not None:
            if 'control_pf_rms' not in self.plot_data:
                self.plot_data['control_pf_rms'] = deque(maxlen=self.max_points)
                self.plot_data['gas_puff'] = deque(maxlen=self.max_points)
                self.plot_data['heating'] = deque(maxlen=self.max_points)
                
            self.plot_data['control_pf_rms'].append(np.sqrt(np.mean(control_action[:6]**2)))
            self.plot_data['gas_puff'].append(control_action[6])
            self.plot_data['heating'].append(control_action[7])
    
    def update_plot(self, frame):
        """Animation update function."""
        if not self.time_data or not self.plot_data:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
        
        time_array = np.array(self.time_data)
        time_relative = time_array - time_array[0] if len(time_array) > 0 else np.array([])
        
        # Q-factor and disruption probability
        ax1 = self.axes[0, 0]
        if 'q_min' in self.plot_data:
            ax1.plot(time_relative, list(self.plot_data['q_min']), 'cyan', label='Q-min', linewidth=2)
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='q=1')
            ax1.axhline(y=2.0, color='yellow', linestyle='--', alpha=0.7, label='q=2')
            
        if 'disruption_prob' in self.plot_data:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(time_relative, list(self.plot_data['disruption_prob']), 
                         'red', alpha=0.7, label='Disruption Risk')
            ax1_twin.set_ylabel('Disruption Probability', color='red')
            ax1_twin.tick_params(colors='white')
            
        ax1.set_title('Q-factor & Disruption Risk', color='white')
        ax1.set_xlabel('Time (s)', color='white')
        ax1.set_ylabel('Q-factor', color='white')
        ax1.legend()
        
        # Shape error
        ax2 = self.axes[0, 1]
        if 'shape_error' in self.plot_data:
            ax2.plot(time_relative, list(self.plot_data['shape_error']), 'orange', linewidth=2)
            ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Limit')
            
        ax2.set_title('Shape Error', color='white')
        ax2.set_xlabel('Time (s)', color='white')
        ax2.set_ylabel('Error (cm)', color='white')
        ax2.legend()
        
        # Plasma beta
        ax3 = self.axes[1, 0]
        if 'beta' in self.plot_data:
            ax3.plot(time_relative, list(self.plot_data['beta']), 'green', linewidth=2)
            ax3.axhline(y=0.04, color='red', linestyle='--', alpha=0.7, label='Beta limit')
            
        ax3.set_title('Plasma Beta', color='white')
        ax3.set_xlabel('Time (s)', color='white')
        ax3.set_ylabel('Beta', color='white')
        ax3.legend()
        
        # Control signals
        ax4 = self.axes[1, 1]
        if 'control_pf_rms' in self.plot_data:
            ax4.plot(time_relative, list(self.plot_data['control_pf_rms']), 
                    'purple', label='PF RMS', linewidth=2)
        if 'gas_puff' in self.plot_data:
            ax4.plot(time_relative, list(self.plot_data['gas_puff']), 
                    'blue', label='Gas Puff', linewidth=2)
        if 'heating' in self.plot_data:
            heating_normalized = np.array(list(self.plot_data['heating'])) / 50.0  # Normalize for display
            ax4.plot(time_relative, heating_normalized, 
                    'red', label='Heating (norm)', linewidth=2)
            
        ax4.set_title('Control Signals', color='white')
        ax4.set_xlabel('Time (s)', color='white')
        ax4.set_ylabel('Amplitude', color='white')
        ax4.legend()
        
        plt.tight_layout()
    
    def start_animation(self, interval: int = 1000) -> None:
        """Start real-time animation."""
        if not self.is_running:
            self.animation = animation.FuncAnimation(
                self.fig, self.update_plot, interval=interval, blit=False
            )
            self.is_running = True
            plt.show(block=False)
    
    def stop_animation(self) -> None:
        """Stop real-time animation."""
        if self.animation:
            self.animation.event_source.stop()
            self.is_running = False
    
    def save_plot(self, filename: str) -> None:
        """Save current plot to file."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight',
                        facecolor='black', edgecolor='none')


if dash is not None and go is not None:
    class WebDashboard:
        """Web-based dashboard using Plotly and Dash."""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        
        # Data storage
        self.plasma_data = deque(maxlen=self.config.history_length)
        self.performance_data = deque(maxlen=self.config.history_length)
        self.anomaly_data = deque(maxlen=self.config.history_length)
        self.prediction_data = deque(maxlen=self.config.history_length)
        
        # Dash app
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        # Thread for running server
        self.server_thread = None
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
    
    def setup_layout(self) -> None:
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1("Tokamak Control Dashboard", 
                   style={'textAlign': 'center', 'color': 'white', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.Button('Start Monitoring', id='start-btn', n_clicks=0,
                           style={'marginRight': 10}),
                html.Button('Stop Monitoring', id='stop-btn', n_clicks=0,
                           style={'marginRight': 10}),
                html.Button('Reset Data', id='reset-btn', n_clicks=0),
                
                html.Div(id='status-indicator', 
                        style={'display': 'inline-block', 'marginLeft': 20,
                               'padding': 10, 'backgroundColor': 'green',
                               'color': 'white', 'borderRadius': 5},
                        children='Status: Ready')
            ], style={'textAlign': 'center', 'marginBottom': 20}),
            
            # Main monitoring plots
            html.Div([
                dcc.Graph(id='plasma-state-plot'),
                dcc.Graph(id='performance-plot'),
            ], style={'width': '100%'}),
            
            # Secondary plots
            html.Div([
                dcc.Graph(id='control-plot'),
                dcc.Graph(id='prediction-plot'),
            ], style={'width': '100%'}),
            
            # Anomaly alerts
            html.Div([
                html.H3("Recent Anomalies", style={'color': 'white'}),
                html.Div(id='anomaly-list')
            ], style={'marginTop': 20, 'padding': 20, 'backgroundColor': '#2F2F2F'}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=int(self.config.update_interval * 1000),
                n_intervals=0,
                disabled=not self.config.auto_refresh
            ),
            
            # Hidden div to store data
            html.Div(id='hidden-div', style={'display': 'none'})
        ], style={'backgroundColor': '#1E1E1E', 'padding': 20})
    
    def setup_callbacks(self) -> None:
        """Setup dashboard callbacks."""
        
        @self.app.callback(
            [Output('plasma-state-plot', 'figure'),
             Output('performance-plot', 'figure'),
             Output('control-plot', 'figure'),
             Output('prediction-plot', 'figure'),
             Output('anomaly-list', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_plots(n):
            """Update all dashboard plots."""
            
            # Plasma state plot
            plasma_fig = self.create_plasma_state_plot()
            
            # Performance metrics plot
            performance_fig = self.create_performance_plot()
            
            # Control signals plot
            control_fig = self.create_control_plot()
            
            # Prediction plot
            prediction_fig = self.create_prediction_plot()
            
            # Anomaly list
            anomaly_components = self.create_anomaly_list()
            
            return plasma_fig, performance_fig, control_fig, prediction_fig, anomaly_components
    
    def create_plasma_state_plot(self) -> go.Figure:
        """Create plasma state monitoring plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Q-factor & Disruption Risk', 'Shape Error', 
                           'Plasma Beta', 'Plasma Current'],
            specs=[[{"secondary_y": True}, {}],
                   [{}, {}]]
        )
        
        if not self.plasma_data:
            return fig
        
        # Extract data
        times = [d['timestamp'] for d in self.plasma_data]
        q_mins = [d['q_min'] for d in self.plasma_data]
        shape_errors = [d['shape_error'] for d in self.plasma_data]
        betas = [d['beta'] for d in self.plasma_data]
        currents = [d['plasma_current'] for d in self.plasma_data]
        disruption_probs = [d['disruption_probability'] for d in self.plasma_data]
        
        # Q-factor and disruption risk
        fig.add_trace(
            go.Scatter(x=times, y=q_mins, mode='lines', name='Q-min', 
                      line=dict(color='cyan', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=times, y=disruption_probs, mode='lines', name='Disruption Risk',
                      line=dict(color='red', width=2)),
            row=1, col=1, secondary_y=True
        )
        
        # Shape error
        fig.add_trace(
            go.Scatter(x=times, y=shape_errors, mode='lines', name='Shape Error',
                      line=dict(color='orange', width=2)),
            row=1, col=2
        )
        
        # Plasma beta
        fig.add_trace(
            go.Scatter(x=times, y=betas, mode='lines', name='Beta',
                      line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        # Plasma current
        fig.add_trace(
            go.Scatter(x=times, y=currents, mode='lines', name='Plasma Current',
                      line=dict(color='blue', width=2)),
            row=2, col=2
        )
        
        # Add threshold lines
        fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                     annotation_text="q=1", row=1, col=1)
        fig.add_hline(y=2.0, line_dash="dash", line_color="yellow", 
                     annotation_text="q=2", row=1, col=1)
        fig.add_hline(y=5.0, line_dash="dash", line_color="red", 
                     annotation_text="Limit", row=1, col=2)
        fig.add_hline(y=0.04, line_dash="dash", line_color="red", 
                     annotation_text="Beta Limit", row=2, col=1)
        
        fig.update_layout(
            title="Plasma State Monitoring",
            template="plotly_dark",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_performance_plot(self) -> go.Figure:
        """Create performance metrics plot."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Energy Efficiency', 'Shape Control Accuracy', 
                           'Uptime', 'Cost per Shot']
        )
        
        if not self.performance_data:
            return fig
        
        # Extract performance data
        times = [d['timestamp'] for d in self.performance_data]
        efficiency = [d['energy_efficiency'] for d in self.performance_data]
        accuracy = [d['shape_control_accuracy'] for d in self.performance_data]
        uptime = [d['uptime_percentage'] for d in self.performance_data]
        cost = [d['operational_cost_per_shot'] for d in self.performance_data]
        
        # Energy efficiency
        fig.add_trace(
            go.Scatter(x=times, y=efficiency, mode='lines+markers', name='Efficiency',
                      line=dict(color='green', width=2)),
            row=1, col=1
        )
        
        # Shape control accuracy
        fig.add_trace(
            go.Scatter(x=times, y=accuracy, mode='lines+markers', name='Accuracy',
                      line=dict(color='blue', width=2)),
            row=1, col=2
        )
        
        # Uptime
        fig.add_trace(
            go.Scatter(x=times, y=uptime, mode='lines+markers', name='Uptime',
                      line=dict(color='cyan', width=2)),
            row=2, col=1
        )
        
        # Cost per shot
        fig.add_trace(
            go.Scatter(x=times, y=cost, mode='lines+markers', name='Cost',
                      line=dict(color='orange', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Performance Metrics",
            template="plotly_dark",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_control_plot(self) -> go.Figure:
        """Create control signals plot."""
        fig = go.Figure()
        
        if not self.plasma_data:
            return fig
        
        times = [d['timestamp'] for d in self.plasma_data if 'control_action' in d]
        
        if not times:
            return fig
        
        # Extract control signals
        pf_currents = []
        gas_puffs = []
        heating = []
        
        for d in self.plasma_data:
            if 'control_action' in d:
                action = d['control_action']
                pf_currents.append(np.sqrt(np.mean(action[:6]**2)))
                gas_puffs.append(action[6])
                heating.append(action[7])
        
        # Plot control signals
        fig.add_trace(go.Scatter(x=times, y=pf_currents, mode='lines', 
                                name='PF Coil RMS', line=dict(color='purple', width=2)))
        fig.add_trace(go.Scatter(x=times, y=gas_puffs, mode='lines', 
                                name='Gas Puff Rate', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=times, y=heating, mode='lines', 
                                name='Heating Power', line=dict(color='red', width=2)))
        
        fig.update_layout(
            title="Control Signals",
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Signal Amplitude",
            height=400
        )
        
        return fig
    
    def create_prediction_plot(self) -> go.Figure:
        """Create prediction plot."""
        fig = go.Figure()
        
        if not self.prediction_data:
            return fig
        
        # Extract prediction data
        times = [d['timestamp'] for d in self.prediction_data]
        predicted_shape_error = [d['predicted_values'].get('shape_error', 0) for d in self.prediction_data]
        predicted_q_min = [d['predicted_values'].get('q_min', 0) for d in self.prediction_data]
        
        # Plot predictions
        fig.add_trace(go.Scatter(x=times, y=predicted_shape_error, mode='lines',
                                name='Predicted Shape Error', line=dict(color='orange', dash='dot')))
        fig.add_trace(go.Scatter(x=times, y=predicted_q_min, mode='lines',
                                name='Predicted Q-min', line=dict(color='cyan', dash='dot')))
        
        fig.update_layout(
            title="Performance Predictions",
            template="plotly_dark",
            xaxis_title="Time",
            yaxis_title="Predicted Values",
            height=400
        )
        
        return fig
    
    def create_anomaly_list(self) -> List[html.Div]:
        """Create anomaly alert list."""
        if not self.anomaly_data:
            return [html.P("No anomalies detected", style={'color': 'green'})]
        
        # Get recent anomalies (last 10)
        recent_anomalies = list(self.anomaly_data)[-10:]
        
        anomaly_components = []
        for anomaly in reversed(recent_anomalies):  # Most recent first
            severity_color = 'red' if anomaly['severity'] > 0.7 else 'orange' if anomaly['severity'] > 0.4 else 'yellow'
            
            anomaly_div = html.Div([
                html.Span(f"🚨 {anomaly['type']}", style={'fontWeight': 'bold', 'color': severity_color}),
                html.Span(f" | Severity: {anomaly['severity']:.2f} | ", style={'color': 'white'}),
                html.Span(f"{anomaly['description']}", style={'color': 'lightgray'}),
                html.Br(),
                html.Small(f"Time: {anomaly['timestamp']}", style={'color': 'gray'})
            ], style={'marginBottom': 10, 'padding': 10, 'backgroundColor': '#3F3F3F', 'borderRadius': 5})
            
            anomaly_components.append(anomaly_div)
        
        return anomaly_components
    
    def add_plasma_data(self, timestamp: float, plasma_state: PlasmaState,
                       control_action: Optional[np.ndarray] = None) -> None:
        """Add plasma state data point."""
        data_point = {
            'timestamp': timestamp,
            'q_min': plasma_state.q_min,
            'shape_error': plasma_state.shape_error,
            'beta': plasma_state.plasma_beta,
            'plasma_current': plasma_state.plasma_current,
            'disruption_probability': plasma_state.disruption_probability,
            'elongation': plasma_state.elongation,
            'triangularity': plasma_state.triangularity
        }
        
        if control_action is not None:
            data_point['control_action'] = control_action
        
        self.plasma_data.append(data_point)
    
    def add_performance_data(self, timestamp: float, performance: PerformanceMetrics) -> None:
        """Add performance metrics data point."""
        data_point = {
            'timestamp': timestamp,
            'energy_efficiency': performance.energy_efficiency,
            'shape_control_accuracy': performance.shape_control_accuracy,
            'uptime_percentage': performance.uptime_percentage,
            'operational_cost_per_shot': performance.operational_cost_per_shot,
            'q_factor_stability': performance.q_factor_stability,
            'target_achievement_rate': performance.target_achievement_rate
        }
        
        self.performance_data.append(data_point)
    
    def add_anomaly_data(self, anomaly: AnomalyEvent) -> None:
        """Add anomaly event."""
        data_point = {
            'timestamp': anomaly.timestamp,
            'type': anomaly.type.value,
            'severity': anomaly.severity,
            'description': anomaly.description,
            'confidence': anomaly.confidence
        }
        
        self.anomaly_data.append(data_point)
    
    def add_prediction_data(self, prediction: PredictionResult) -> None:
        """Add prediction result."""
        data_point = {
            'timestamp': prediction.timestamp,
            'predicted_values': prediction.predicted_values,
            'confidence_intervals': prediction.confidence_intervals,
            'model_accuracy': prediction.model_accuracy
        }
        
        self.prediction_data.append(data_point)
    
    def start_server(self) -> None:
        """Start dashboard server in separate thread."""
        if not self.is_running:
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            self.is_running = True
            self.logger.info(f"Dashboard server started on port {self.config.port}")
    
    def _run_server(self) -> None:
        """Run the Dash server."""
        self.app.run_server(
            host='0.0.0.0',
            port=self.config.port,
            debug=self.config.debug,
            use_reloader=False
        )
    
    def stop_server(self) -> None:
        """Stop dashboard server."""
        self.is_running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)

else:
    # Fallback class when dependencies are missing
    class WebDashboard:
        def __init__(self, config=None):
            print("Warning: Dashboard dependencies not available. Using fallback implementation.")
        
        def add_plasma_data(self, state, info=None):
            pass
        
        def start_server(self):
            pass
        
        def stop_server(self):
            pass


def create_dashboard_system(config: DashboardConfig = None) -> Dict[str, Any]:
    """
    Factory function to create complete dashboard system.
    
    Returns:
        Dictionary containing plotter and web dashboard instances
    """
    config = config or DashboardConfig()
    
    real_time_plotter = RealTimePlotter()
    web_dashboard = WebDashboard(config)
    
    return {
        'real_time_plotter': real_time_plotter,
        'web_dashboard': web_dashboard,
        'config': config
    }