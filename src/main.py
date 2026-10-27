#!/usr/bin/env python3
"""
src/main.py
Entry point for the Adaptive Traffic Light Control with Multi‑Agent RL project.

This script parses command‑line arguments, configures Ray/RLlib, registers the
traffic‑light environment (or a fallback Gymnasium environment), and launches
training using a PPO multi‑agent trainer.

Typical usage
-------------
    python -m src.main \
        --env TrafficLightEnv-v0 \
        --num-workers 4 \
        --train-iterations 2000 \
        --checkpoint-dir ./checkpoints \
        --log-dir ./logs \
        --seed 42

All arguments are optional; sensible defaults are provided.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.logger import pretty_print

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command‑line arguments.

    Parameters
    ----------
    argv: list[str] | None
        Argument list to parse. If ``None`` (default), ``sys.argv[1:]`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a multi‑agent RL traffic‑light controller using Ray RLlib."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment ID. Use a custom TrafficLightEnv if registered.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of parallel RolloutWorkers for sampling.",
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=1000,
        help="Number of training iterations (calls to trainer.train()).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory where checkpoints will be saved.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for Ray Tune logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=None,
        help="Optional early‑stop when the mean episode reward reaches this value.",
    )
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Path to a JSON file containing additional trainer config overrides.",
    )
    parser.add_argument(
        "--no-ray-init",
        action="store_true",
        help="If set, Ray will not be automatically initialized (useful for tests).",
    )
    return parser.parse_args(argv)


def load_json_config(path: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file.

    Parameters
    ----------
    path: str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def register_custom_env(env_name: str) -> None:
    """
    Register a custom environment with Gymnasium if it is not already known.

    The function attempts to import a module named ``src.env`` that should expose
    a class ``TrafficLightEnv`` (sub‑classing ``gymnasium.Env`` or ``MultiAgentEnv``).
    If successful, the environment is registered under ``env_name``. If the import
    fails, the function silently does nothing – the caller can still use a standard
    Gymnasium environment.

    Parameters
    ----------
    env_name: str
        The identifier under which the environment will be registered.
    """
    try:
        from src.env import TrafficLightEnv  # type: ignore
    except Exception as exc:  # pragma: no cover
        log.debug("Custom TrafficLightEnv not found: %s", exc)
        return

    if not isinstance(TrafficLightEnv, type):
        log.warning("TrafficLightEnv is not a class; skipping registration.")
        return

    gym.register(
        id=env_name,
        entry_point=TrafficLightEnv,
        max_episode_steps=500,  # reasonable default; can be overridden later
    )
    log.info("Registered custom environment '%s' using TrafficLightEnv.", env_name)


def make_multi_agent_env(env_id: str) -> MultiAgentEnv:
    """
    Factory that creates a multi‑agent environment instance.

    If the environment ID corresponds to a standard Gymnasium env, it will be
    wrapped into a ``MultiAgentEnv`` compatible interface using a simple
    conversion where each agent shares the same observation and action spaces.

    Parameters
    ----------
    env_id: str
        Gymnasium environment ID.

    Returns
    -------
    MultiAgentEnv
        An instantiated multi‑agent environment.
    """
    # Import inside the function to avoid heavy imports at module load time.
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    # First try to create the env directly; if it already implements MultiAgentEnv,
    # we just return it.
    env = gym.make(env_id)
    if isinstance(env.unwrapped, MultiAgentEnv):
        return env.unwrapped  # type: ignore[return-value]

    # Otherwise, we wrap a single‑agent env into a dummy multi‑agent env where a
    # single agent named "agent_0" interacts with the environment.
    class DummyMultiAgentEnv(MultiAgentEnv):
        def __init__(self):
            self._env = env
            self.agents = ["agent_0"]
            self.observation_spaces = {"agent_0": self._env.observation_space}
            self.action_spaces = {"agent_0": self._env.action_space}

        def reset(self, *, seed=None, options=None):
            obs, info = self._env.reset(seed=seed, options=options)
            return {"agent_0": obs}, {"agent_0": info}

        def step(self, action_dict):
            action = action_dict["agent_0"]
            obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated
            return {"agent_0": obs}, {"agent_0": reward}, {"agent_0": done}, {"agent_0": info}

        def render(self):
            return self._env.render()

        def close(self):
            self._env.close()

    return DummyMultiAgentEnv()


def get_trainer_config(
    env_id: str,
    num_workers: int,
    seed: int,
    extra_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Assemble the RLlib trainer configuration dictionary.

    Parameters
    ----------
    env_id: str
        Environment identifier.
    num_workers: int
        Number of rollout workers.
    seed: int
        Random seed.
    extra_config: dict | None
        Optional dictionary with user‑provided overrides.

    Returns
    -------
    dict
        RLlib trainer configuration.
    """
    config: Dict[str, Any] = {
        "env": env_id,
        "env_config": {},  # placeholder for any env‑specific kwargs
        "num_workers": num_workers,
        "framework": "torch",
        "seed": seed,
        "log_level": "INFO",
        "model": {
            # Example model config; can be overridden via extra_config
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
        },
        "multiagent": {
            # By default we assume a single shared policy for all agents.
            "policies": {
                "shared_policy": (
                    None,  # use default policy class (PPO)
                    gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),  # placeholder
                    gym.spaces.Discrete(2),  # placeholder
                    {},
                )
            },
            "policy_mapping_fn": lambda agent_id, episode, **kwargs: "shared_policy",
        },
        "evaluation_config": {
            "explore": False,
        },
        "evaluation_interval": 5,
        "evaluation_num_episodes": 10,
    }

    # If the env is a MultiAgentEnv, we can infer observation/action spaces from a sample instance.
    sample_env = make_multi_agent_env(env_id)
    sample_obs_space = sample_env.observation_spaces["agent_0"]
    sample_act_space = sample_env.action_spaces["agent_0"]
    config["multiagent"]["policies"]["shared_policy"] = (
        None,
        sample_obs_space,
        sample_act_space,
        {},
    )

    if extra_config:
        # Deep merge of dictionaries – simple implementation sufficient for our use‑case.
        for k, v in extra_config.items():
            if isinstance(v, dict) and k in config and isinstance(config[k], dict):
                config[k].update(v)
            else:
                config[k] = v

    return config


def train(
    env_id: str,
    num_workers: int,
    train_iterations: int,
    checkpoint_dir: str,
    log_dir: str,
    seed: int,
    stop_reward: float | None = None,
    extra_config: Dict[str, Any] | None = None,
) -> PPOTrainer:
    """
    Initialise Ray, configure the trainer, and run the training loop.

    Parameters
    ----------
    env_id: str
        Environment ID.
    num_workers: int
        Number of rollout workers.
    train_iterations: int
        Number of training iterations.
    checkpoint_dir: str
        Directory where checkpoints will be stored.
    log_dir: str
        Directory for Ray Tune logs.
    seed: int
        Random seed.
    stop_reward: float | None
        Optional early‑stop threshold based on mean episode reward.
    extra_config: dict | None
        Additional trainer configuration overrides.

    Returns
    -------
    PPOTrainer
        The trained RLlib trainer instance.
    """
    # Ensure directories exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Initialise Ray (skip if already initialised)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    # Register the environment with RLlib if it is not a standard gym env.
    if env_id not in gym.envs.registry:
        register_custom_env(env_id)

    config = get_trainer_config(
        env_id=env_id,
        num_workers=num_workers,
        seed=seed,
        extra_config=extra_config,
    )

    trainer = PPOTrainer(config=config, logger_creator=lambda cfg: tune.logger.TBXLogger(cfg, log_dir))

    # Training loop with optional early stopping.
    for i in range(1, train_iterations + 1):
        result = trainer.train()
        log.info("Iteration %d - reward_mean: %.3f", i, result["episode_reward_mean"])
        if stop_reward is not None and result["episode_reward_mean"] >= stop_reward:
            log.info(
                "Early stopping triggered at iteration %d (reward_mean=%.3f >= %.3f)",
                i,
                result["episode_reward_mean"],
                stop_reward,
            )
            break

        # Periodic checkpointing
        if i % 10 == 0:
            ckpt_path = trainer.save(checkpoint_dir)
            log.info("Checkpoint saved at %s", ckpt_path)

    # Final checkpoint
    final_ckpt = trainer.save(checkpoint_dir)
    log.info("Training completed. Final checkpoint at %s", final_ckpt)

    return trainer


def main(argv: list[str] | None = None) -> None:
    """
    Main entry point: parse arguments, optionally load a JSON config, and start training.
    """
    args = parse_args(argv)

    # Load optional JSON config overrides
    json_overrides: Dict[str, Any] = {}
    if args.config_json:
        json_overrides = load_json_config(args.config_json)
        log.info("Loaded JSON config overrides from %s", args.config_json)

    # Combine command‑line args with JSON overrides (command line wins)
    trainer_config_overrides = json_overrides.get("trainer_config", {})

    # Run training unless user explicitly disabled Ray init (useful for unit tests)
    if args.no_ray_init:
        log.info("Ray initialization disabled via --no-ray-init; exiting.")
        return

    train(
        env_id=args.env,
        num_workers=args.num_workers,
        train_iterations=args.train_iterations,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        seed=args.seed,
        stop_reward=args.stop_reward,
        extra_config=trainer_config_overrides,
    )


if __name__ == "__main__":
    main()