import pytest
import numpy as np

# The environment should be importable from the package. Adjust the import path if necessary.
try:
    # Common import patterns for custom Gymnasium environments
    from adaptive_traffic.env import TrafficLightEnv  # type: ignore
except ImportError:
    try:
        from adaptive_traffic.envs import TrafficLightEnv  # type: ignore
    except ImportError:
        try:
            from adaptive_traffic.envs.traffic_light import TrafficLightEnv  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Could not import TrafficLightEnv. Ensure the project defines the environment "
                "class and that the import path matches one of the attempted locations."
            ) from exc

# Gymnasium is the successor of OpenAI Gym
import gymnasium as gym


@pytest.fixture(scope="function")
def env():
    """
    Create a fresh instance of the environment for each test.
    The environment is closed after the test finishes.
    """
    # Prefer using gymnasium.make if the env is registered; otherwise instantiate directly.
    try:
        env_instance = gym.make("AdaptiveTrafficLight-v0")
    except gym.error.Error:
        env_instance = TrafficLightEnv()
    yield env_instance
    env_instance.close()


def test_observation_and_action_spaces(env):
    """
    Verify that the environment defines valid observation and action spaces.
    """
    # Observation space should be a gymnasium Space (Box, Dict, etc.)
    assert isinstance(env.observation_space, gym.spaces.Space), "observation_space must be a gymnasium Space"
    # Action space should also be a gymnasium Space
    assert isinstance(env.action_space, gym.spaces.Space), "action_space must be a gymnasium Space"

    # Basic sanity checks for Box spaces
    if isinstance(env.observation_space, gym.spaces.Box):
        # Ensure bounds are finite
        assert np.all(np.isfinite(env.observation_space.low)), "observation_space.low must be finite"
        assert np.all(np.isfinite(env.observation_space.high)), "observation_space.high must be finite"
        # Shape must be defined
        assert len(env.observation_space.shape) > 0, "observation_space must have a non‑empty shape"

    if isinstance(env.action_space, gym.spaces.Box):
        # Discrete actions are more common for traffic lights, but Box is also allowed
        assert np.all(np.isfinite(env.action_space.low)), "action_space.low must be finite"
        assert np.all(np.isfinite(env.action_space.high)), "action_space.high must be finite"


def test_reset_returns_valid_observation(env):
    """
    Ensure that ``reset`` returns an observation that conforms to the declared observation space.
    """
    reset_result = env.reset()
    # Gymnasium reset returns (obs, info); older versions may return only obs.
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}

    # Observation must be compatible with the observation space
    assert env.observation_space.contains(obs), "Reset observation is not within observation_space"

    # Info should be a dict (Gymnasium standard)
    assert isinstance(info, dict), "Reset info must be a dict"


def test_step_dynamics(env):
    """
    Perform a single step with a random action and verify the output types and consistency.
    """
    # Reset the environment first
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    # Sample a random action from the action space
    action = env.action_space.sample()

    # Perform a step
    step_result = env.step(action)

    # Gymnasium step returns (obs, reward, terminated, truncated, info)
    assert isinstance(step_result, tuple) and len(step_result) == 5, "step must return 5 values"
    next_obs, reward, terminated, truncated, info = step_result

    # Observation must be valid
    assert env.observation_space.contains(next_obs), "Step observation is not within observation_space"

    # Reward must be a scalar numeric type
    assert isinstance(reward, (int, float, np.number)), f"Reward must be numeric, got {type(reward)}"

    # terminated and truncated must be booleans
    assert isinstance(terminated, bool), "terminated flag must be bool"
    assert isinstance(truncated, bool), "truncated flag must be bool"

    # Info must be a dict
    assert isinstance(info, dict), "info must be a dict"

    # Ensure that at least one of terminated or truncated is False (environment cannot be both finished)
    assert not (terminated and truncated), "Both terminated and truncated cannot be True simultaneously"


def test_observation_bounds_respected(env):
    """
    Run a short rollout and assert that all observations stay within the defined bounds.
    """
    obs, _ = env.reset()
    for _ in range(10):
        assert env.observation_space.contains(obs), "Observation out of bounds during rollout"
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def test_multi_agent_structure_if_dict_observation(env):
    """
    If the environment uses a dict observation (common for multi‑agent setups),
    verify that the keys match between observation and action spaces.
    """
    if isinstance(env.observation_space, gym.spaces.Dict):
        # Ensure action space is also a Dict for a multi‑agent env
        assert isinstance(env.action_space, gym.spaces.Dict), (
            "When observation_space is a Dict, action_space should also be a Dict."
        )
        # Keys must match
        obs_keys = set(env.observation_space.spaces.keys())
        act_keys = set(env.action_space.spaces.keys())
        assert obs_keys == act_keys, "Observation and action space keys must match for multi‑agent env"

        # Test a single step with dict actions
        obs, _ = env.reset()
        assert isinstance(obs, dict), "Reset observation should be a dict for multi‑agent env"
        action = {k: space.sample() for k, space in env.action_space.spaces.items()}
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Validate each component
        assert isinstance(next_obs, dict), "Step observation should be a dict for multi‑agent env"
        for k in obs_keys:
            assert env.observation_space.spaces[k].contains(next_obs[k]), f"Observation for agent {k} out of bounds"
            assert isinstance(reward, (int, float, np.number)), "Reward must be numeric"
            # terminated/truncated can be per‑agent or global; we only check they are bools
            assert isinstance(terminated, bool) and isinstance(truncated, bool), "Flags must be bools"
