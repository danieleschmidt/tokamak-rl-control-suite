"""
Tests for command-line interface.
"""

import pytest
import argparse
import tempfile
import os
from unittest.mock import patch, MagicMock
from tokamak_rl.cli import train_command, evaluate_command, main


class TestCLIArguments:
    """Test CLI argument parsing."""
    
    def test_main_help(self):
        """Test main help message."""
        with patch('sys.argv', ['tokamak-rl', '--help']):
            with pytest.raises(SystemExit):
                main()
                
    def test_train_subcommand_help(self):
        """Test train subcommand help."""
        with patch('sys.argv', ['tokamak-rl', 'train', '--help']):
            with pytest.raises(SystemExit):
                main()
                
    def test_eval_subcommand_help(self):
        """Test eval subcommand help."""
        with patch('sys.argv', ['tokamak-rl', 'eval', '--help']):
            with pytest.raises(SystemExit):
                main()


class TestTrainingCommand:
    """Test training command functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_args = argparse.Namespace(
            agent='SAC',
            tokamak='ITER',
            episodes=5,
            lr=3e-4,
            frequency=100,
            safety=True,
            device='cpu',
            save_path=None,
            log_dir='./test_logs',
            log_interval=2,
            train_freq=1
        )
        
    @patch('tokamak_rl.cli.make_tokamak_env')
    @patch('tokamak_rl.cli.create_agent')
    @patch('tokamak_rl.cli.create_monitoring_system')
    def test_train_command_basic(self, mock_monitoring, mock_agent, mock_env):
        """Test basic training command execution."""
        # Mock environment
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = (mock_obs := MagicMock(), {})
        mock_env_instance.step.return_value = (mock_obs, 1.0, False, False, {})
        mock_env_instance.get_episode_metrics.return_value = {'mean_shape_error': 2.5}
        mock_env_instance.plasma_state = MagicMock()
        mock_env.return_value = mock_env_instance
        
        # Mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.act.return_value = [0.1, -0.1, 0.0, 0.2, -0.2, 0.1, 0.5, 0.3]
        mock_agent_instance.learn.return_value = {'critic_loss': 0.5, 'actor_loss': 0.3}
        mock_agent.return_value = mock_agent_instance
        
        # Mock monitoring
        mock_monitoring_dict = {
            'monitor': MagicMock(),
            'logger': MagicMock()
        }
        mock_monitoring.return_value = mock_monitoring_dict
        
        # Run training (should not raise exceptions)
        try:
            train_command(self.test_args)
        except Exception as e:
            pytest.fail(f"Training command failed: {e}")
            
        # Verify calls
        mock_env.assert_called_once()
        mock_agent.assert_called_once()
        mock_monitoring.assert_called_once()
        
    @patch('tokamak_rl.cli.make_tokamak_env')
    @patch('tokamak_rl.cli.create_agent')
    @patch('tokamak_rl.cli.create_monitoring_system')
    def test_train_command_with_save(self, mock_monitoring, mock_agent, mock_env):
        """Test training command with model saving."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
            
        try:
            # Set save path
            self.test_args.save_path = save_path
            
            # Mock components
            mock_env_instance = MagicMock()
            mock_env_instance.reset.return_value = (MagicMock(), {})
            mock_env_instance.step.return_value = (MagicMock(), 5.0, True, False, {})  # High reward
            mock_env_instance.get_episode_metrics.return_value = {'mean_shape_error': 1.0}
            mock_env_instance.plasma_state = MagicMock()
            mock_env.return_value = mock_env_instance
            
            mock_agent_instance = MagicMock()
            mock_agent_instance.act.return_value = [0.1] * 8
            mock_agent_instance.learn.return_value = {}
            mock_agent.return_value = mock_agent_instance
            
            mock_monitoring.return_value = {'monitor': MagicMock(), 'logger': MagicMock()}
            
            # Run training
            train_command(self.test_args)
            
            # Verify save was called
            mock_agent_instance.save.assert_called()
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
                
    def test_train_command_parameter_validation(self):
        """Test training command parameter validation."""
        # Test invalid agent type
        invalid_args = argparse.Namespace(**vars(self.test_args))
        invalid_args.agent = 'INVALID'
        
        with patch('tokamak_rl.cli.make_tokamak_env'):
            with patch('tokamak_rl.cli.create_agent') as mock_agent:
                mock_agent.side_effect = ValueError("Unknown agent type")
                
                with pytest.raises(ValueError):
                    train_command(invalid_args)


class TestEvaluationCommand:
    """Test evaluation command functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_args = argparse.Namespace(
            agent='SAC',
            model_path=None,
            tokamak='ITER',
            episodes=3,
            frequency=100,
            safety=True,
            device='cpu',
            render=False,
            output='./test_results.json',
            log_dir='./test_eval_logs'
        )
        
    @patch('tokamak_rl.cli.make_tokamak_env')
    @patch('tokamak_rl.cli.create_agent')
    @patch('tokamak_rl.cli.create_monitoring_system')
    def test_evaluate_command_basic(self, mock_monitoring, mock_agent, mock_env):
        """Test basic evaluation command execution."""
        # Mock environment
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = (MagicMock(), {})
        mock_env_instance.step.return_value = (MagicMock(), 1.0, True, False, {})
        mock_env_instance.get_episode_metrics.return_value = {
            'mean_shape_error': 2.0,
            'final_shape_error': 1.8,
            'episode_length': 100
        }
        mock_env_instance.plasma_state = MagicMock()
        mock_env_instance.plasma_state.shape_error = 2.0
        mock_env_instance.plasma_state.q_min = 2.5
        mock_env_instance.plasma_state.disruption_probability = 0.05
        mock_env.return_value = mock_env_instance
        
        # Mock agent
        mock_agent_instance = MagicMock()
        mock_agent_instance.act.return_value = [0.1] * 8
        mock_agent.return_value = mock_agent_instance
        
        # Mock monitoring
        mock_monitoring.return_value = {
            'monitor': MagicMock(),
            'renderer': MagicMock()
        }
        
        # Run evaluation
        try:
            evaluate_command(self.test_args)
        except Exception as e:
            pytest.fail(f"Evaluation command failed: {e}")
            
        # Verify calls
        mock_env.assert_called_once()
        mock_agent.assert_called_once()
        mock_monitoring.assert_called_once()
        
    @patch('tokamak_rl.cli.make_tokamak_env')
    @patch('tokamak_rl.cli.create_agent')
    @patch('tokamak_rl.cli.create_monitoring_system')
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_evaluate_command_with_output(self, mock_json_dump, mock_open, 
                                        mock_monitoring, mock_agent, mock_env):
        """Test evaluation command with results output."""
        # Mock components
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = (MagicMock(), {})
        mock_env_instance.step.return_value = (MagicMock(), 2.0, True, False, {})
        mock_env_instance.get_episode_metrics.return_value = {'mean_shape_error': 1.5}
        mock_env_instance.plasma_state = MagicMock()
        mock_env_instance.plasma_state.shape_error = 1.5
        mock_env_instance.plasma_state.q_min = 3.0
        mock_env_instance.plasma_state.disruption_probability = 0.02
        mock_env.return_value = mock_env_instance
        
        mock_agent_instance = MagicMock()
        mock_agent_instance.act.return_value = [0.05] * 8
        mock_agent.return_value = mock_agent_instance
        
        mock_monitoring.return_value = {'monitor': MagicMock(), 'renderer': MagicMock()}
        
        # Run evaluation
        evaluate_command(self.test_args)
        
        # Verify output file was written
        mock_open.assert_called_with('./test_results.json', 'w')
        mock_json_dump.assert_called_once()
        
    @patch('tokamak_rl.cli.make_tokamak_env')
    @patch('tokamak_rl.cli.create_agent')
    @patch('tokamak_rl.cli.create_monitoring_system')
    def test_evaluate_command_with_model_loading(self, mock_monitoring, mock_agent, mock_env):
        """Test evaluation command with model loading."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            model_path = f.name
            
        try:
            # Set model path
            self.test_args.model_path = model_path
            
            # Mock components
            mock_env_instance = MagicMock()
            mock_env_instance.reset.return_value = (MagicMock(), {})
            mock_env_instance.step.return_value = (MagicMock(), 1.0, True, False, {})
            mock_env_instance.get_episode_metrics.return_value = {'mean_shape_error': 2.5}
            mock_env_instance.plasma_state = MagicMock()
            mock_env_instance.plasma_state.shape_error = 2.5
            mock_env_instance.plasma_state.q_min = 2.0
            mock_env_instance.plasma_state.disruption_probability = 0.1
            mock_env.return_value = mock_env_instance
            
            mock_agent_instance = MagicMock()
            mock_agent_instance.act.return_value = [0.2] * 8
            mock_agent.return_value = mock_agent_instance
            
            mock_monitoring.return_value = {'monitor': MagicMock(), 'renderer': MagicMock()}
            
            # Run evaluation
            evaluate_command(self.test_args)
            
            # Verify load was called
            mock_agent_instance.load.assert_called_with(model_path)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestCLIIntegration:
    """Test CLI integration and main function."""
    
    def test_main_no_command(self):
        """Test main function with no command."""
        with patch('sys.argv', ['tokamak-rl']):
            with pytest.raises(SystemExit):
                main()
                
    @patch('tokamak_rl.cli.train_command')
    def test_main_train_command(self, mock_train):
        """Test main function with train command."""
        with patch('sys.argv', [
            'tokamak-rl', 'train', 
            '--agent', 'SAC',
            '--episodes', '5',
            '--device', 'cpu'
        ]):
            main()
            
        mock_train.assert_called_once()
        
    @patch('tokamak_rl.cli.evaluate_command')
    def test_main_eval_command(self, mock_eval):
        """Test main function with eval command."""
        with patch('sys.argv', [
            'tokamak-rl', 'eval',
            '--agent', 'SAC', 
            '--episodes', '3',
            '--device', 'cpu'
        ]):
            main()
            
        mock_eval.assert_called_once()
        
    def test_device_auto_selection(self):
        """Test automatic device selection."""
        with patch('sys.argv', [
            'tokamak-rl', 'train',
            '--device', 'auto'
        ]):
            with patch('tokamak_rl.cli.train_command') as mock_train:
                with patch('torch.cuda.is_available', return_value=False):
                    main()
                    
                # Check that device was set to cpu
                args = mock_train.call_args[0][0]
                assert args.device == 'cpu'
                
    @patch('torch.cuda.is_available', return_value=True)
    def test_device_auto_selection_cuda(self, mock_cuda):
        """Test automatic device selection with CUDA available."""
        with patch('sys.argv', [
            'tokamak-rl', 'train',
            '--device', 'auto'
        ]):
            with patch('tokamak_rl.cli.train_command') as mock_train:
                main()
                
                # Check that device was set to cuda
                args = mock_train.call_args[0][0]
                assert args.device == 'cuda'
                
    def test_entry_points(self):
        """Test CLI entry point functions."""
        from tokamak_rl.cli import train, evaluate
        
        # Test train entry point
        with patch('sys.argv', ['tokamak-train', '--help']):
            with pytest.raises(SystemExit):
                train()
                
        # Test evaluate entry point  
        with patch('sys.argv', ['tokamak-eval', '--help']):
            with pytest.raises(SystemExit):
                evaluate()


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    @patch('tokamak_rl.cli.make_tokamak_env')
    def test_training_keyboard_interrupt(self, mock_env):
        """Test training graceful handling of keyboard interrupt."""
        mock_env.side_effect = KeyboardInterrupt()
        
        args = argparse.Namespace(
            agent='SAC', tokamak='ITER', episodes=10,
            lr=3e-4, frequency=100, safety=True, device='cpu',
            save_path=None, log_dir='./logs', log_interval=5, train_freq=1
        )
        
        # Should handle interrupt gracefully
        try:
            train_command(args)
        except KeyboardInterrupt:
            pytest.fail("KeyboardInterrupt not handled gracefully")
            
    @patch('tokamak_rl.cli.create_agent')
    def test_invalid_agent_type_error(self, mock_agent):
        """Test handling of invalid agent type."""
        mock_agent.side_effect = ValueError("Unknown agent type")
        
        args = argparse.Namespace(
            agent='INVALID', tokamak='ITER', episodes=1,
            lr=3e-4, frequency=100, safety=True, device='cpu',
            save_path=None, log_dir='./logs', log_interval=1, train_freq=1
        )
        
        with pytest.raises(ValueError):
            train_command(args)
            
    def test_missing_model_file_warning(self):
        """Test warning for missing model file in evaluation."""
        with patch('tokamak_rl.cli.make_tokamak_env') as mock_env:
            with patch('tokamak_rl.cli.create_agent') as mock_agent:
                with patch('tokamak_rl.cli.create_monitoring_system') as mock_monitoring:
                    # Mock components
                    mock_env.return_value = MagicMock()
                    mock_env.return_value.reset.return_value = (MagicMock(), {})
                    mock_env.return_value.step.return_value = (MagicMock(), 1.0, True, False, {})
                    mock_env.return_value.get_episode_metrics.return_value = {}
                    mock_env.return_value.plasma_state = MagicMock()
                    
                    mock_agent_instance = MagicMock()
                    mock_agent_instance.act.return_value = [0.1] * 8
                    mock_agent.return_value = mock_agent_instance
                    
                    mock_monitoring.return_value = {'monitor': MagicMock(), 'renderer': MagicMock()}
                    
                    args = argparse.Namespace(
                        agent='SAC', model_path=None, tokamak='ITER', episodes=1,
                        frequency=100, safety=True, device='cpu', render=False,
                        output='./results.json', log_dir='./logs'
                    )
                    
                    # Should run without error (using random agent)
                    evaluate_command(args)