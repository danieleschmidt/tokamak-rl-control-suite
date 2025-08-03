"""
Tests for dashboard and visualization components.
"""

import pytest
import numpy as np
import tempfile
import time
from unittest.mock import patch, MagicMock
from tokamak_rl.dashboard import (
    RealTimePlotter, WebDashboard, DashboardConfig,
    create_dashboard_system
)
from tokamak_rl.physics import TokamakConfig, PlasmaState
from tokamak_rl.business import PerformanceMetrics
from tokamak_rl.analytics import AnomalyEvent, AnomalyType, PredictionResult


class TestDashboardConfig:
    """Test dashboard configuration."""
    
    def test_default_config(self):
        """Test default dashboard configuration."""
        config = DashboardConfig()
        
        assert config.update_interval == 1.0
        assert config.history_length == 1000
        assert config.port == 8050
        assert config.debug == False
        assert config.theme == "dark"
        assert config.auto_refresh == True
    
    def test_custom_config(self):
        """Test custom dashboard configuration."""
        config = DashboardConfig(
            update_interval=0.5,
            history_length=500,
            port=8080,
            debug=True,
            theme="light",
            auto_refresh=False
        )
        
        assert config.update_interval == 0.5
        assert config.history_length == 500
        assert config.port == 8080
        assert config.debug == True
        assert config.theme == "light"
        assert config.auto_refresh == False


class TestRealTimePlotter:
    """Test real-time matplotlib plotting."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.plotter = RealTimePlotter(figsize=(10, 6), max_points=100)
        self.config = TokamakConfig.from_preset("ITER")
        self.plasma_state = PlasmaState(self.config)
    
    def test_plotter_initialization(self):
        """Test plotter initialization."""
        assert self.plotter.figsize == (10, 6)
        assert self.plotter.max_points == 100
        assert len(self.plotter.time_data) == 0
        assert len(self.plotter.plot_data) == 0
        assert not self.plotter.is_running
        assert self.plotter.fig is not None
        assert self.plotter.axes is not None
    
    def test_data_point_addition(self):
        """Test adding data points."""
        timestamp = time.time()
        control_action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.5, 10.0])
        
        self.plotter.add_data_point(timestamp, self.plasma_state, control_action)
        
        assert len(self.plotter.time_data) == 1
        assert self.plotter.time_data[0] == timestamp
        
        # Check plasma data was stored
        assert 'q_min' in self.plotter.plot_data
        assert 'shape_error' in self.plotter.plot_data
        assert 'beta' in self.plotter.plot_data
        assert 'disruption_prob' in self.plotter.plot_data
        
        assert len(self.plotter.plot_data['q_min']) == 1
        assert self.plotter.plot_data['q_min'][0] == self.plasma_state.q_min
        
        # Check control data was stored
        assert 'control_pf_rms' in self.plotter.plot_data
        assert 'gas_puff' in self.plotter.plot_data
        assert 'heating' in self.plotter.plot_data
        
        assert len(self.plotter.plot_data['gas_puff']) == 1
        assert self.plotter.plot_data['gas_puff'][0] == control_action[6]
    
    def test_data_point_without_control(self):
        """Test adding data point without control action."""
        timestamp = time.time()
        
        self.plotter.add_data_point(timestamp, self.plasma_state)
        
        assert len(self.plotter.time_data) == 1
        assert 'q_min' in self.plotter.plot_data
        assert 'control_pf_rms' not in self.plotter.plot_data
    
    def test_max_points_limit(self):
        """Test that data respects max points limit."""
        # Add more data than max_points
        for i in range(150):
            timestamp = time.time() + i
            self.plotter.add_data_point(timestamp, self.plasma_state)
        
        assert len(self.plotter.time_data) == self.plotter.max_points
        
        # Check that plot_data also respects limit
        for key in self.plotter.plot_data:
            assert len(self.plotter.plot_data[key]) == self.plotter.max_points
    
    @patch('matplotlib.pyplot.show')
    def test_animation_start_stop(self, mock_show):
        """Test starting and stopping animation."""
        # Add some data first
        for i in range(10):
            timestamp = time.time() + i
            self.plotter.add_data_point(timestamp, self.plasma_state)
        
        # Start animation
        self.plotter.start_animation(interval=100)
        assert self.plotter.is_running
        assert self.plotter.animation is not None
        
        # Stop animation
        self.plotter.stop_animation()
        assert not self.plotter.is_running
    
    def test_plot_update_with_no_data(self):
        """Test plot update with no data."""
        # Should not raise error even with no data
        try:
            self.plotter.update_plot(0)
        except Exception as e:
            pytest.fail(f"update_plot failed with no data: {e}")
    
    def test_plot_update_with_data(self):
        """Test plot update with data."""
        # Add some test data
        for i in range(20):
            timestamp = time.time() + i * 0.1
            # Vary plasma state for interesting plots
            self.plasma_state.q_min = 1.5 + 0.5 * np.sin(i * 0.2)
            self.plasma_state.shape_error = 2.0 + np.sin(i * 0.3)
            self.plasma_state.plasma_beta = 0.02 + 0.01 * np.sin(i * 0.1)
            self.plasma_state.disruption_probability = 0.05 + 0.03 * np.sin(i * 0.4)
            
            control_action = np.array([0.1 * np.sin(i * 0.2)] * 6 + [0.5, 10.0])
            self.plotter.add_data_point(timestamp, self.plasma_state, control_action)
        
        # Update plot should not raise error
        try:
            self.plotter.update_plot(0)
        except Exception as e:
            pytest.fail(f"update_plot failed with data: {e}")
    
    def test_save_plot(self):
        """Test saving plot to file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            filename = f.name
        
        try:
            # Add some data
            for i in range(5):
                timestamp = time.time() + i
                self.plotter.add_data_point(timestamp, self.plasma_state)
            
            # Update plot
            self.plotter.update_plot(0)
            
            # Save plot
            self.plotter.save_plot(filename)
            
            # Check file exists
            import os
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
            
        finally:
            import os
            if os.path.exists(filename):
                os.unlink(filename)


class TestWebDashboard:
    """Test web-based dashboard."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = DashboardConfig(port=8051, auto_refresh=False)  # Different port to avoid conflicts
        self.dashboard = WebDashboard(self.config)
        self.plasma_config = TokamakConfig.from_preset("SPARC")
        self.plasma_state = PlasmaState(self.plasma_config)
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization."""
        assert self.dashboard.config == self.config
        assert len(self.dashboard.plasma_data) == 0
        assert len(self.dashboard.performance_data) == 0
        assert len(self.dashboard.anomaly_data) == 0
        assert len(self.dashboard.prediction_data) == 0
        assert not self.dashboard.is_running
        assert self.dashboard.app is not None
    
    def test_plasma_data_addition(self):
        """Test adding plasma data."""
        timestamp = time.time()
        control_action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.5, 10.0])
        
        self.dashboard.add_plasma_data(timestamp, self.plasma_state, control_action)
        
        assert len(self.dashboard.plasma_data) == 1
        
        data_point = self.dashboard.plasma_data[0]
        assert data_point['timestamp'] == timestamp
        assert data_point['q_min'] == self.plasma_state.q_min
        assert data_point['shape_error'] == self.plasma_state.shape_error
        assert data_point['beta'] == self.plasma_state.plasma_beta
        assert np.array_equal(data_point['control_action'], control_action)
    
    def test_plasma_data_without_control(self):
        """Test adding plasma data without control action."""
        timestamp = time.time()
        
        self.dashboard.add_plasma_data(timestamp, self.plasma_state)
        
        assert len(self.dashboard.plasma_data) == 1
        data_point = self.dashboard.plasma_data[0]
        assert 'control_action' not in data_point
    
    def test_performance_data_addition(self):
        """Test adding performance data."""
        timestamp = time.time()
        performance = PerformanceMetrics(
            energy_efficiency=0.85,
            shape_control_accuracy=2.5,
            uptime_percentage=95.0,
            q_factor_stability=0.9
        )
        
        self.dashboard.add_performance_data(timestamp, performance)
        
        assert len(self.dashboard.performance_data) == 1
        
        data_point = self.dashboard.performance_data[0]
        assert data_point['timestamp'] == timestamp
        assert data_point['energy_efficiency'] == 0.85
        assert data_point['shape_control_accuracy'] == 2.5
        assert data_point['uptime_percentage'] == 95.0
    
    def test_anomaly_data_addition(self):
        """Test adding anomaly data."""
        anomaly = AnomalyEvent(
            type=AnomalyType.PERFORMANCE_DEGRADATION,
            severity=0.7,
            timestamp=time.time(),
            features={'q_min': 1.2},
            description="Test anomaly",
            confidence=0.8
        )
        
        self.dashboard.add_anomaly_data(anomaly)
        
        assert len(self.dashboard.anomaly_data) == 1
        
        data_point = self.dashboard.anomaly_data[0]
        assert data_point['type'] == 'performance_degradation'
        assert data_point['severity'] == 0.7
        assert data_point['description'] == "Test anomaly"
        assert data_point['confidence'] == 0.8
    
    def test_prediction_data_addition(self):
        """Test adding prediction data."""
        prediction = PredictionResult(
            predicted_values={'shape_error': 2.5, 'q_min': 2.0},
            confidence_intervals={'shape_error': (2.0, 3.0)},
            forecast_horizon=300.0,
            model_accuracy=0.85,
            feature_importance={'q_min': 0.3},
            timestamp=time.time()
        )
        
        self.dashboard.add_prediction_data(prediction)
        
        assert len(self.dashboard.prediction_data) == 1
        
        data_point = self.dashboard.prediction_data[0]
        assert data_point['predicted_values']['shape_error'] == 2.5
        assert data_point['model_accuracy'] == 0.85
    
    def test_data_history_limit(self):
        """Test that data history respects limit."""
        # Set small limit for testing
        self.dashboard.config.history_length = 5
        self.dashboard.plasma_data = self.dashboard.plasma_data.__class__(maxlen=5)
        
        # Add more data than limit
        for i in range(10):
            timestamp = time.time() + i
            self.dashboard.add_plasma_data(timestamp, self.plasma_state)
        
        assert len(self.dashboard.plasma_data) == 5
    
    def test_plasma_state_plot_creation(self):
        """Test plasma state plot creation."""
        # Add sample data
        for i in range(10):
            timestamp = time.time() + i
            # Vary plasma parameters
            self.plasma_state.q_min = 1.8 + 0.4 * np.sin(i * 0.2)
            self.plasma_state.shape_error = 2.0 + np.sin(i * 0.3)
            self.plasma_state.plasma_beta = 0.02 + 0.01 * np.sin(i * 0.1)
            self.plasma_state.disruption_probability = 0.05 + 0.02 * np.sin(i * 0.4)
            
            self.dashboard.add_plasma_data(timestamp, self.plasma_state)
        
        # Create plot
        fig = self.dashboard.create_plasma_state_plot()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        # Should have multiple traces for different parameters
        assert len(fig.data) > 0
    
    def test_performance_plot_creation(self):
        """Test performance plot creation."""
        # Add sample performance data
        for i in range(10):
            timestamp = time.time() + i
            performance = PerformanceMetrics(
                energy_efficiency=0.8 + 0.1 * np.sin(i * 0.2),
                shape_control_accuracy=2.0 + 0.5 * np.sin(i * 0.3),
                uptime_percentage=90 + 5 * np.sin(i * 0.1),
                operational_cost_per_shot=100 + 20 * np.sin(i * 0.4),
                timestamp=timestamp
            )
            self.dashboard.add_performance_data(timestamp, performance)
        
        # Create plot
        fig = self.dashboard.create_performance_plot()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_control_plot_creation(self):
        """Test control signals plot creation."""
        # Add sample data with control actions
        for i in range(10):
            timestamp = time.time() + i
            control_action = np.array([
                0.1 * np.sin(i * 0.2), -0.1 * np.sin(i * 0.2),
                0.05 * np.sin(i * 0.3), -0.05 * np.sin(i * 0.3),
                0.02 * np.sin(i * 0.1), -0.02 * np.sin(i * 0.1),
                0.5 + 0.1 * np.sin(i * 0.4),
                10.0 + 2.0 * np.sin(i * 0.5)
            ])
            
            self.dashboard.add_plasma_data(timestamp, self.plasma_state, control_action)
        
        # Create plot
        fig = self.dashboard.create_control_plot()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        # Should have traces for different control signals
        assert len(fig.data) >= 3  # PF, gas puff, heating
    
    def test_prediction_plot_creation(self):
        """Test prediction plot creation."""
        # Add sample prediction data
        for i in range(10):
            timestamp = time.time() + i
            prediction = PredictionResult(
                predicted_values={
                    'shape_error': 2.0 + 0.5 * np.sin(i * 0.2),
                    'q_min': 1.8 + 0.2 * np.sin(i * 0.3)
                },
                confidence_intervals={},
                forecast_horizon=300.0,
                model_accuracy=0.85,
                feature_importance={},
                timestamp=timestamp
            )
            
            self.dashboard.add_prediction_data(prediction)
        
        # Create plot
        fig = self.dashboard.create_prediction_plot()
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) >= 2  # shape_error and q_min predictions
    
    def test_anomaly_list_creation(self):
        """Test anomaly list creation."""
        # Add sample anomalies
        anomalies = [
            AnomalyEvent(
                type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=0.7, timestamp=time.time(),
                features={}, description="Test anomaly 1", confidence=0.8
            ),
            AnomalyEvent(
                type=AnomalyType.SHAPE_DEVIATION,
                severity=0.5, timestamp=time.time() + 100,
                features={}, description="Test anomaly 2", confidence=0.9
            )
        ]
        
        for anomaly in anomalies:
            self.dashboard.add_anomaly_data(anomaly)
        
        # Create anomaly list
        anomaly_components = self.dashboard.create_anomaly_list()
        
        assert isinstance(anomaly_components, list)
        assert len(anomaly_components) == 2
    
    def test_empty_data_plots(self):
        """Test plot creation with empty data."""
        # Should not raise errors even with no data
        try:
            plasma_fig = self.dashboard.create_plasma_state_plot()
            performance_fig = self.dashboard.create_performance_plot()
            control_fig = self.dashboard.create_control_plot()
            prediction_fig = self.dashboard.create_prediction_plot()
            anomaly_list = self.dashboard.create_anomaly_list()
            
            assert plasma_fig is not None
            assert performance_fig is not None
            assert control_fig is not None
            assert prediction_fig is not None
            assert isinstance(anomaly_list, list)
            
        except Exception as e:
            pytest.fail(f"Plot creation failed with empty data: {e}")
    
    def test_layout_setup(self):
        """Test dashboard layout setup."""
        # Layout should be set up during initialization
        assert self.dashboard.app.layout is not None
        
        # Check that layout contains expected components
        layout_children = self.dashboard.app.layout.children
        assert len(layout_children) > 0
        
        # Should contain graphs and controls
        graph_components = [child for child in layout_children 
                          if hasattr(child, 'children') and 
                          any(hasattr(gc, 'id') and 'plot' in str(getattr(gc, 'id', '')) 
                              for gc in getattr(child, 'children', []))]
        assert len(graph_components) > 0


class TestDashboardSystemIntegration:
    """Test dashboard system integration."""
    
    def test_dashboard_system_factory(self):
        """Test dashboard system factory function."""
        config = DashboardConfig(port=8052)
        dashboard_system = create_dashboard_system(config)
        
        assert isinstance(dashboard_system, dict)
        assert 'real_time_plotter' in dashboard_system
        assert 'web_dashboard' in dashboard_system
        assert 'config' in dashboard_system
        
        assert isinstance(dashboard_system['real_time_plotter'], RealTimePlotter)
        assert isinstance(dashboard_system['web_dashboard'], WebDashboard)
        assert dashboard_system['config'] == config
    
    def test_integrated_dashboard_workflow(self):
        """Test integrated dashboard workflow."""
        config = DashboardConfig(port=8053, auto_refresh=False)
        dashboard_system = create_dashboard_system(config)
        
        plotter = dashboard_system['real_time_plotter']
        web_dashboard = dashboard_system['web_dashboard']
        
        # Setup test data
        plasma_config = TokamakConfig.from_preset("ITER")
        plasma_state = PlasmaState(plasma_config)
        
        # Add data to both dashboard components
        timestamp = time.time()
        control_action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.5, 10.0])
        
        # Add to real-time plotter
        plotter.add_data_point(timestamp, plasma_state, control_action)
        
        # Add to web dashboard
        web_dashboard.add_plasma_data(timestamp, plasma_state, control_action)
        
        # Verify data was added
        assert len(plotter.time_data) == 1
        assert len(web_dashboard.plasma_data) == 1
        
        # Test plot updates
        try:
            plotter.update_plot(0)
            plasma_fig = web_dashboard.create_plasma_state_plot()
            
            assert plasma_fig is not None
            
        except Exception as e:
            pytest.fail(f"Integrated dashboard workflow failed: {e}")
    
    def test_dashboard_with_all_data_types(self):
        """Test dashboard with all types of data."""
        config = DashboardConfig(port=8054, auto_refresh=False)
        dashboard_system = create_dashboard_system(config)
        
        web_dashboard = dashboard_system['web_dashboard']
        
        # Add plasma data
        plasma_config = TokamakConfig.from_preset("SPARC")
        plasma_state = PlasmaState(plasma_config)
        timestamp = time.time()
        control_action = np.array([0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.5, 10.0])
        
        web_dashboard.add_plasma_data(timestamp, plasma_state, control_action)
        
        # Add performance data
        performance = PerformanceMetrics(
            energy_efficiency=0.85,
            shape_control_accuracy=2.5,
            uptime_percentage=95.0
        )
        web_dashboard.add_performance_data(timestamp, performance)
        
        # Add anomaly data
        anomaly = AnomalyEvent(
            type=AnomalyType.PERFORMANCE_DEGRADATION,
            severity=0.7, timestamp=timestamp,
            features={}, description="Test anomaly", confidence=0.8
        )
        web_dashboard.add_anomaly_data(anomaly)
        
        # Add prediction data
        prediction = PredictionResult(
            predicted_values={'shape_error': 2.5},
            confidence_intervals={},
            forecast_horizon=300.0,
            model_accuracy=0.85,
            feature_importance={},
            timestamp=timestamp
        )
        web_dashboard.add_prediction_data(prediction)
        
        # Test that all plots can be created
        try:
            plasma_fig = web_dashboard.create_plasma_state_plot()
            performance_fig = web_dashboard.create_performance_plot()
            control_fig = web_dashboard.create_control_plot()
            prediction_fig = web_dashboard.create_prediction_plot()
            anomaly_list = web_dashboard.create_anomaly_list()
            
            assert plasma_fig is not None
            assert performance_fig is not None
            assert control_fig is not None
            assert prediction_fig is not None
            assert len(anomaly_list) > 0
            
        except Exception as e:
            pytest.fail(f"Dashboard with all data types failed: {e}")
    
    @patch('threading.Thread')
    def test_dashboard_server_management(self, mock_thread):
        """Test dashboard server start/stop functionality."""
        config = DashboardConfig(port=8055)
        web_dashboard = WebDashboard(config)
        
        # Mock thread for testing
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        # Test starting server
        web_dashboard.start_server()
        
        assert web_dashboard.is_running
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()
        
        # Test stopping server
        web_dashboard.stop_server()
        
        assert not web_dashboard.is_running