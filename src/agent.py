import os
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer, PPOConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.policy.policy import Policy
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID, AgentID, EnvConfigDict, RolloutWorkerConfigDict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _default_policy_mapping(agent_id: AgentID, episode: MultiAgentEpisode, **kwargs) -> PolicyID:
    """
    Simple policy mapping function that assigns all agents to a shared policy.
    This can be overridden by the user when constructing the agent.
    """
    return "shared_policy"


class MultiAgentPPOAgent:
    """
    Wrapper around Ray RLlib's PPOTrainer for multi‑agent traffic‑light control.
    Provides a clean interface for training, evaluation, checkpointing and
    visualisation.

    Parameters
    ----------
    env_name : str
        Name of the Gymnasium environment to register with RLlib. The environment
        must inherit from ``gymnasium.Env`` or ``ray.rllib.env.MultiAgentEnv``.
    env_config : dict, optional
        Configuration dictionary passed to the environment constructor.
    policy_mapping_fn : callable, optional
        Function mapping ``agent_id`` to a policy ID. Defaults to a shared policy.
    config_overrides : dict, optional
        Dictionary that will be merged into the default PPO configuration.
    """

    def __init__(
        self,
        env_name: str,
        env_config: Optional[EnvConfigDict] = None,
        policy_mapping_fn: Optional[Callable[[AgentID, MultiAgentEpisode], PolicyID]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Initialise Ray (no‑op if already started)
        ray.init(ignore_reinit_error=True, log_to_driver=False)

        self.env_name = env_name
        self.env_config = env_config or {}
        self.policy_mapping_fn = policy_mapping_fn or _default_policy_mapping

        # Register the environment with RLlib if not already registered.
        if env_name not in gym.envs.registry:
            # The user is expected to have imported the environment module elsewhere.
            raise ValueError(
                f"Environment '{env_name}' is not registered with Gymnasium. "
                "Make sure the environment is imported before creating the agent."
            )

        # Build the default PPO config
        self.config = (
            PPOConfig()
            .environment(env=self.env_name, env_config=self.env_config)
            .framework("torch")
            .rollouts(num_rollout_workers=1)
            .training(
                train_batch_size=4000,
                sgd_minibatch_size=128,
                num_sgd_iter=10,
                lr=5e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                vf_loss_coeff=1.0,
                entropy_coeff=0.01,
            )
            .resources(num_gpus=0)
            .multi_agent(
                policies=self._generate_policy_dict(),
                policy_mapping_fn=self.policy_mapping_fn,
            )
        )

        # Apply user overrides
        if config_overrides:
            self.config = self.config.update_from_dict(config_overrides)

        # Create the trainer
        self.trainer = PPOTrainer(config=self.config.to_dict())
        logger.info("MultiAgentPPOAgent initialized with environment '%s'.", env_name)

    def _generate_policy_dict(self) -> Dict[PolicyID, Tuple[Optional[str], gym.Space, gym.Space, Dict]]:
        """
        Generates a policy dictionary for RLlib. By default we create a single shared
        policy that uses the observation and action spaces of the environment.
        """
        # Create a temporary env instance to query spaces.
        temp_env = gym.make(self.env_name, **self.env_config)
        if isinstance(temp_env, MultiAgentEnv):
            # MultiAgentEnv provides observation_space and action_space per agent.
            # We assume all agents share the same spaces.
            sample_obs = next(iter(temp_env.observation_spaces.values()))
            sample_act = next(iter(temp_env.action_spaces.values()))
        else:
            # Single‑agent fallback (should not happen for traffic control).
            sample_obs = temp_env.observation_space
            sample_act = temp_env.action_space

        policy_spec = {
            "shared_policy": (
                None,  # Use default Torch policy class
                sample_obs,
                sample_act,
                {},  # No extra config
            )
        }
        temp_env.close()
        return policy_spec

    def train(self, num_iterations: int = 10, checkpoint_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Run the PPO training loop.

        Parameters
        ----------
        num_iterations : int
            Number of training iterations to perform.
        checkpoint_dir : str or Path, optional
            Directory where checkpoints will be saved after each iteration.

        Returns
        -------
        results : list of dict
            List containing the training result dict for each iteration.
        """
        results = []
        for i in range(1, num_iterations + 1):
            result = self.trainer.train()
            results.append(result)
            logger.info(
                "Iteration %d/%d - episode_reward_mean: %.2f",
                i,
                num_iterations,
                result["episode_reward_mean"],
            )
            if checkpoint_dir:
                ckpt_path = self.trainer.save(checkpoint_dir=str(checkpoint_dir))
                logger.info("Checkpoint saved to %s", ckpt_path)
        return results

    def evaluate(
        self,
        env: Union[gym.Env, MultiAgentEnv],
        num_episodes: int = 5,
        render: bool = False,
    ) -> Tuple[float, List[Dict[AgentID, float]]]:
        """
        Evaluate the current policy on a given environment.

        Parameters
        ----------
        env : gym.Env or MultiAgentEnv
            Environment instance used for evaluation.
        num_episodes : int
            Number of episodes to run.
        render : bool
            Whether to call ``env.render()`` after each step.

        Returns
        -------
        mean_reward : float
            Mean cumulative reward across all evaluated episodes.
        episode_rewards : list of dict
            Per‑episode cumulative rewards per agent.
        """
        policy = self.trainer.get_policy("shared_policy")
        episode_rewards = []
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = {"__all__": False}
            cumulative_rewards: Dict[AgentID, float] = {}
            while not done["__all__"]:
                actions = {}
                for agent_id, agent_obs in obs.items():
                    action = policy.compute_single_action(agent_obs, explore=False)
                    actions[agent_id] = action
                obs, rew, done, trunc, info = env.step(actions)
                for agent_id, r in rew.items():
                    cumulative_rewards[agent_id] = cumulative_rewards.get(agent_id, 0.0) + r
                if render:
                    env.render()
            episode_rewards.append(cumulative_rewards)
            logger.info("Eval episode %d reward per agent: %s", ep + 1, cumulative_rewards)

        # Compute mean reward across agents and episodes
        all_rewards = [sum(agent_rewards.values()) for agent_rewards in episode_rewards]
        mean_reward = float(np.mean(all_rewards))
        logger.info("Evaluation completed – mean cumulative reward: %.2f", mean_reward)
        return mean_reward, episode_rewards

    def get_action(self, observation: Any, explore: bool = True) -> Any:
        """
        Compute an action for a single observation using the shared policy.

        Parameters
        ----------
        observation : any
            Observation from the environment for a particular agent.
        explore : bool
            Whether to use the exploration policy (True) or deterministic (False).

        Returns
        -------
        action : any
            Action chosen by the policy.
        """
        policy = self.trainer.get_policy("shared_policy")
        return policy.compute_single_action(observation, explore=explore)

    def save(self, checkpoint_path: Union[str, Path]) -> str:
        """
        Save the trainer state to a checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Directory where the checkpoint will be stored.

        Returns
        -------
        checkpoint_path : str
            Full path to the saved checkpoint.
        """
        checkpoint_path = self.trainer.save(checkpoint_dir=str(checkpoint_path))
        logger.info("Trainer checkpoint saved to %s", checkpoint_path)
        return checkpoint_path

    def load(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Restore the trainer from a checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the checkpoint file.
        """
        self.trainer.restore(str(checkpoint_path))
        logger.info("Trainer restored from checkpoint %s", checkpoint_path)

    def plot_training_metrics(self, training_results: List[Dict], metric: str = "episode_reward_mean") -> None:
        """
        Plot a training metric over iterations using Matplotlib.

        Parameters
        ----------
        training_results : list of dict
            Results returned by :meth:`train`.
        metric : str
            Key in the result dict to plot (default: ``episode_reward_mean``).
        """
        if not training_results:
            logger.warning("No training results to plot.")
            return

        values = [res.get(metric, np.nan) for res in training_results]
        iterations = list(range(1, len(values) + 1))

        plt.figure(figsize=(8, 4))
        plt.plot(iterations, values, marker="o")
        plt.title(f"Training {metric.replace('_', ' ').title()}")
        plt.xlabel("Iteration")
        plt.ylabel(metric.replace("_", " ").title())
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def shutdown(self) -> None:
        """
        Gracefully shutdown Ray. Should be called when the agent is no longer needed.
        """
        ray.shutdown()
        logger.info("Ray shutdown complete.")