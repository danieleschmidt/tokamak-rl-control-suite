"""
Command-line interface for tokamak RL training and evaluation.

This module provides CLI commands for training agents, running evaluations,
and managing tokamak plasma control experiments.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from .environment import make_tokamak_env
from .agents import create_agent
from .monitoring import create_monitoring_system
from .physics import TokamakConfig


def train_command(args: argparse.Namespace) -> None:
    """Execute training command."""
    print(f"🚀 Starting tokamak RL training with {args.agent} agent")
    print(f"Tokamak: {args.tokamak}")
    print(f"Episodes: {args.episodes}")
    print(f"Safety: {'Enabled' if args.safety else 'Disabled'}")
    
    # Create environment
    env = make_tokamak_env(
        tokamak_config=args.tokamak,
        control_frequency=args.frequency,
        enable_safety=args.safety
    )
    
    # Create agent
    agent = create_agent(
        args.agent,
        env.observation_space,
        env.action_space,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Create monitoring system
    monitoring = create_monitoring_system(
        log_dir=args.log_dir,
        enable_tensorboard=True,
        enable_alerts=True
    )
    
    monitor = monitoring['monitor']
    logger = monitoring['logger']
    
    print(f"📊 Logging to: {args.log_dir}")
    print("=" * 50)
    
    # Training loop
    episode_rewards = []
    best_reward = float('-inf')
    
    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            while True:
                # Select action
                action = agent.act(obs, deterministic=False)
                
                # Environment step
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                # Store experience
                if hasattr(agent, 'add_experience'):
                    agent.add_experience(obs, action, reward, next_obs, terminated)
                
                # Learn
                if hasattr(agent, 'learn') and episode_steps % args.train_freq == 0:
                    train_metrics = agent.learn()
                    if logger and train_metrics:
                        logger.log_training_metrics(train_metrics, episode * 1000 + episode_steps)
                
                # Log step
                if monitor:
                    monitor.log_step(env.plasma_state, action, reward, info)
                
                # Log plasma state
                if logger:
                    logger.log_plasma_state(env.plasma_state, episode * 1000 + episode_steps)
                
                # Update state
                obs = next_obs
                episode_reward += reward
                episode_steps += 1
                
                # Check termination
                if terminated or truncated:
                    break
            
            # Episode completed
            episode_rewards.append(episode_reward)
            episode_metrics = env.get_episode_metrics()
            
            # Log episode
            if monitor:
                monitor.log_episode_end(episode_metrics)
            if logger:
                logger.log_episode_metrics(episode_metrics, episode)
                
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                if args.save_path:
                    save_path = Path(args.save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    agent.save(str(save_path))
                    print(f"💾 Saved best model (reward: {episode_reward:.2f})")
            
            # Progress report
            if episode % args.log_interval == 0:
                recent_rewards = episode_rewards[-args.log_interval:]
                avg_reward = np.mean(recent_rewards)
                avg_shape_error = episode_metrics.get('mean_shape_error', 0)
                avg_q_min = np.mean([env.plasma_state.q_min])  # Last episode q_min
                
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | "
                      f"Shape Error: {avg_shape_error:5.2f} cm | "
                      f"Q-min: {avg_q_min:4.2f}")
                      
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    # Final save
    if args.save_path:
        final_path = str(Path(args.save_path).with_suffix('.final.pt'))
        agent.save(final_path)
        print(f"💾 Saved final model to {final_path}")
    
    # Close monitoring
    if logger:
        logger.close()
        
    print(f"✅ Training completed! Best reward: {best_reward:.2f}")


def evaluate_command(args: argparse.Namespace) -> None:
    """Execute evaluation command."""
    print(f"🔬 Evaluating agent on {args.tokamak}")
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    
    # Create environment
    env = make_tokamak_env(
        tokamak_config=args.tokamak,
        control_frequency=args.frequency,
        enable_safety=args.safety
    )
    
    # Create and load agent
    agent = create_agent(
        args.agent,
        env.observation_space,
        env.action_space,
        device=args.device
    )
    
    if args.model_path:
        agent.load(args.model_path)
        print(f"📁 Loaded model from {args.model_path}")
    else:
        print("⚠️  No model specified, using random agent")
    
    # Create monitoring
    monitoring = create_monitoring_system(
        log_dir=args.log_dir,
        enable_tensorboard=False,
        enable_alerts=True
    )
    
    monitor = monitoring['monitor']
    renderer = monitoring['renderer']
    
    # Evaluation loop
    episode_metrics = []
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        episode_data = {
            'rewards': [],
            'shape_errors': [],
            'q_mins': [],
            'disruption_risks': []
        }
        
        while True:
            # Select action (deterministic for evaluation)
            action = agent.act(obs, deterministic=True)
            
            # Environment step
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Record data
            episode_data['rewards'].append(reward)
            episode_data['shape_errors'].append(env.plasma_state.shape_error)
            episode_data['q_mins'].append(env.plasma_state.q_min)
            episode_data['disruption_risks'].append(env.plasma_state.disruption_probability)
            
            # Log step
            if monitor:
                monitor.log_step(env.plasma_state, action, reward, info)
            
            # Render if requested
            if args.render and episode == 0:  # Only render first episode
                if episode_steps % 10 == 0:  # Render every 10 steps
                    env.render("human")
                    
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            if terminated or truncated:
                break
        
        # Episode statistics
        env_metrics = env.get_episode_metrics()
        episode_metrics.append(env_metrics)
        
        # Log episode
        if monitor:
            monitor.log_episode_end(env_metrics)
        
        print(f"Episode {episode+1:2d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Steps: {episode_steps:4d} | "
              f"Shape Error: {env_metrics.get('mean_shape_error', 0):5.2f} cm | "
              f"Final Q-min: {episode_data['q_mins'][-1]:4.2f}")
    
    # Compute overall statistics
    if episode_metrics:
        avg_metrics = {}
        for key in episode_metrics[0].keys():
            values = [m[key] for m in episode_metrics if key in m]
            if values:
                avg_metrics[f'avg_{key}'] = np.mean(values)
                avg_metrics[f'std_{key}'] = np.std(values)
    
        print("\n📊 Evaluation Results:")
        print("=" * 50)
        for key, value in avg_metrics.items():
            print(f"{key:25s}: {value:8.3f}")
            
        # Save results
        if args.output:
            results = {
                'evaluation_config': {
                    'tokamak': args.tokamak,
                    'episodes': args.episodes,
                    'agent': args.agent,
                    'model_path': args.model_path
                },
                'episode_metrics': episode_metrics,
                'summary_metrics': avg_metrics
            }
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to {args.output}")
    
    print("✅ Evaluation completed!")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Tokamak RL Control Suite CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--agent', choices=['SAC', 'DREAMER'], default='SAC',
                             help='RL algorithm to use')
    train_parser.add_argument('--tokamak', choices=['ITER', 'SPARC', 'NSTX', 'DIII-D'], 
                             default='ITER', help='Tokamak configuration')
    train_parser.add_argument('--episodes', type=int, default=1000,
                             help='Number of training episodes')
    train_parser.add_argument('--lr', type=float, default=3e-4,
                             help='Learning rate')
    train_parser.add_argument('--frequency', type=int, default=100,
                             help='Control frequency (Hz)')
    train_parser.add_argument('--safety', action='store_true', default=True,
                             help='Enable safety shield')
    train_parser.add_argument('--no-safety', dest='safety', action='store_false',
                             help='Disable safety shield')
    train_parser.add_argument('--device', default='auto',
                             help='Device to use (cpu/cuda/auto)')
    train_parser.add_argument('--save-path', default='./models/tokamak_agent.pt',
                             help='Path to save trained model')
    train_parser.add_argument('--log-dir', default='./logs',
                             help='Directory for logs and tensorboard')
    train_parser.add_argument('--log-interval', type=int, default=10,
                             help='Episodes between progress reports')
    train_parser.add_argument('--train-freq', type=int, default=1,
                             help='Steps between training updates')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained agent')
    eval_parser.add_argument('--agent', choices=['SAC', 'DREAMER'], default='SAC',
                            help='RL algorithm type')
    eval_parser.add_argument('--model-path', 
                            help='Path to trained model')
    eval_parser.add_argument('--tokamak', choices=['ITER', 'SPARC', 'NSTX', 'DIII-D'],
                            default='ITER', help='Tokamak configuration')
    eval_parser.add_argument('--episodes', type=int, default=10,
                            help='Number of evaluation episodes')
    eval_parser.add_argument('--frequency', type=int, default=100,
                            help='Control frequency (Hz)')
    eval_parser.add_argument('--safety', action='store_true', default=True,
                            help='Enable safety shield')
    eval_parser.add_argument('--no-safety', dest='safety', action='store_false',
                            help='Disable safety shield')
    eval_parser.add_argument('--device', default='auto',
                            help='Device to use (cpu/cuda/auto)')
    eval_parser.add_argument('--render', action='store_true',
                            help='Render plasma visualization')
    eval_parser.add_argument('--output', default='./evaluation_results.json',
                            help='Path to save evaluation results')
    eval_parser.add_argument('--log-dir', default='./eval_logs',
                            help='Directory for evaluation logs')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'
    
    # Execute command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'eval':
        evaluate_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def train() -> None:
    """Entry point for tokamak-train command."""
    sys.argv = ['tokamak-train'] + sys.argv[1:]
    sys.argv.insert(1, 'train')
    main()


def evaluate() -> None:
    """Entry point for tokamak-eval command."""
    sys.argv = ['tokamak-eval'] + sys.argv[1:]
    sys.argv.insert(1, 'eval')
    main()


if __name__ == '__main__':
    main()