"""
RL Trainer for the framework.
"""

import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime


class RLTrainer:
    """
    Trainer for RL agents in the medical research environment.
    
    Handles training loop, logging, and checkpointing.
    """
    
    def __init__(
        self,
        agent: Any,
        environment: Any,
        reward_function: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            agent: RL agent to train
            environment: Environment to train in
            reward_function: Optional custom reward function
            config: Training configuration
        """
        self.agent = agent
        self.environment = environment
        self.reward_function = reward_function
        
        # Training configuration
        self.config = config or {}
        self.num_episodes = self.config.get("num_episodes", 100)
        self.max_steps_per_episode = self.config.get("max_steps_per_episode", 50)
        self.save_frequency = self.config.get("save_frequency", 10)
        self.log_frequency = self.config.get("log_frequency", 1)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        
        # Training state
        self.episode = 0
        self.total_steps = 0
        self.training_history: List[Dict] = []
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self) -> Dict[str, Any]:
        """
        Run the training loop.
        
        Returns:
            Training statistics
        """
        print(f"Starting training for {self.num_episodes} episodes...")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print("="*60)
        
        for episode_num in range(self.num_episodes):
            self.episode = episode_num
            episode_stats = self._run_episode()
            
            self.training_history.append(episode_stats)
            
            # Logging
            if (episode_num + 1) % self.log_frequency == 0:
                self._log_progress(episode_num, episode_stats)
            
            # Checkpointing
            if (episode_num + 1) % self.save_frequency == 0:
                self._save_checkpoint(episode_num)
        
        print("\nTraining completed!")
        print("="*60)
        
        # Final checkpoint
        self._save_checkpoint(self.num_episodes - 1, final=True)
        
        return self._get_training_summary()
    
    def _run_episode(self) -> Dict[str, Any]:
        """
        Run a single training episode.
        
        Returns:
            Episode statistics
        """
        observation = self.environment.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False
        
        actions_taken = []
        rewards_received = []
        
        while not done and episode_steps < self.max_steps_per_episode:
            # Agent selects action
            action = self.agent.select_action(observation)
            actions_taken.append(action.get("action_type", "unknown"))
            
            # Environment step
            next_observation, env_reward, done, info = self.environment.step(action)
            
            # Calculate reward using custom reward function if provided
            if self.reward_function:
                reward = self.reward_function.calculate_reward(
                    action, observation, next_observation
                )
            else:
                reward = env_reward
            
            rewards_received.append(reward)
            episode_reward += reward
            
            # Agent update
            experience = (observation, action, reward, next_observation, done)
            update_metrics = self.agent.update(experience)
            
            # Move to next state
            observation = next_observation
            episode_steps += 1
            self.total_steps += 1
        
        # Episode statistics
        episode_stats = {
            "episode": self.episode,
            "steps": episode_steps,
            "total_reward": episode_reward,
            "average_reward": episode_reward / episode_steps if episode_steps > 0 else 0,
            "actions": actions_taken,
            "rewards": rewards_received,
            "done": done
        }
        
        return episode_stats
    
    def _log_progress(self, episode_num: int, episode_stats: Dict) -> None:
        """Log training progress."""
        print(f"\nEpisode {episode_num + 1}/{self.num_episodes}")
        print(f"  Steps: {episode_stats['steps']}")
        print(f"  Total Reward: {episode_stats['total_reward']:.2f}")
        print(f"  Average Reward: {episode_stats['average_reward']:.2f}")
        print(f"  Actions: {', '.join(episode_stats['actions'][:5])}")
        
        # Show moving average
        if len(self.training_history) >= 10:
            recent_rewards = [h['total_reward'] for h in self.training_history[-10:]]
            avg_recent = sum(recent_rewards) / len(recent_rewards)
            print(f"  Moving Avg (10 eps): {avg_recent:.2f}")
    
    def _save_checkpoint(self, episode_num: int, final: bool = False) -> None:
        """Save training checkpoint."""
        suffix = "final" if final else f"ep{episode_num + 1}"
        
        # Save agent
        agent_path = os.path.join(self.checkpoint_dir, f"agent_{suffix}.json")
        self.agent.save(agent_path)
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, f"history_{suffix}.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"\n✓ Checkpoint saved: {suffix}")
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training run."""
        total_rewards = [h['total_reward'] for h in self.training_history]
        avg_rewards = [h['average_reward'] for h in self.training_history]
        
        summary = {
            "total_episodes": len(self.training_history),
            "total_steps": self.total_steps,
            "total_reward": sum(total_rewards),
            "average_episode_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0,
            "best_episode_reward": max(total_rewards) if total_rewards else 0,
            "worst_episode_reward": min(total_rewards) if total_rewards else 0,
            "final_10_episode_average": sum(total_rewards[-10:]) / min(10, len(total_rewards)) if total_rewards else 0,
        }
        
        return summary
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, Any]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render episodes
            
        Returns:
            Evaluation statistics
        """
        print(f"\nEvaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_steps = []
        
        for ep in range(num_episodes):
            observation = self.environment.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < self.max_steps_per_episode:
                if render:
                    self.environment.render()
                
                action = self.agent.select_action(observation)
                next_observation, reward, done, info = self.environment.step(action)
                
                episode_reward += reward
                observation = next_observation
                episode_steps += 1
            
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_steps)
            
            print(f"  Episode {ep + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        
        eval_summary = {
            "num_episodes": num_episodes,
            "average_reward": sum(eval_rewards) / len(eval_rewards),
            "std_reward": self._std(eval_rewards),
            "average_steps": sum(eval_steps) / len(eval_steps),
            "min_reward": min(eval_rewards),
            "max_reward": max(eval_rewards)
        }
        
        print("\nEvaluation Summary:")
        print(f"  Average Reward: {eval_summary['average_reward']:.2f} ± {eval_summary['std_reward']:.2f}")
        print(f"  Average Steps: {eval_summary['average_steps']:.1f}")
        
        return eval_summary
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def load_checkpoint(self, checkpoint_name: str) -> None:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint to load (e.g., "ep50" or "final")
        """
        agent_path = os.path.join(self.checkpoint_dir, f"agent_{checkpoint_name}.json")
        history_path = os.path.join(self.checkpoint_dir, f"history_{checkpoint_name}.json")
        
        # Load agent
        if os.path.exists(agent_path):
            self.agent.load(agent_path)
            print(f"✓ Agent loaded from {agent_path}")
        
        # Load history
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
            print(f"✓ Training history loaded from {history_path}")
    
    def export_results(self, output_path: str) -> None:
        """
        Export training results to a file.
        
        Args:
            output_path: Path to save results
        """
        results = {
            "config": self.config,
            "summary": self._get_training_summary(),
            "history": self.training_history,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results exported to {output_path}")
