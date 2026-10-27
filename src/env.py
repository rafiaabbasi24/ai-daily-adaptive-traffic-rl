import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple, Optional


class TrafficIntersectionEnv(gym.Env):
    """
    A simple custom Gymnasium environment that simulates a single traffic
    intersection with four incoming lanes (North, East, South, West).

    The environment models vehicle arrivals as a Poisson process and
    departures according to the current traffic‑light phase.  The agent
    chooses between two phases:

    * ``0`` – North‑South green (vehicles on the North and South lanes may
      depart).
    * ``1`` – East‑West green (vehicles on the East and West lanes may
      depart).

    The observation is a ``Box`` containing the current queue length on each
    lane.  The reward is the negative total queue length, encouraging the
    agent to minimise waiting vehicles.

    The environment is deterministic given a fixed random seed and can be
    rendered as a simple bar plot of the queues.

    Example
    -------
    >>> import gymnasium as gym
    >>> env = TrafficIntersectionEnv(seed=42)
    >>> obs, info = env.reset()
    >>> for _ in range(10):
    ...     action = env.action_space.sample()
    ...     obs, reward, terminated, truncated, info = env.step(action)
    ...     if terminated or truncated:
    ...         break
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    # Mapping from action to lanes that receive service (green)
    _PHASE_MAP = {
        0: (0, 2),  # North (0) and South (2) green
        1: (1, 3),  # East (1) and West (3) green
    }

    def __init__(
        self,
        *,
        max_steps: int = 200,
        arrival_rate: float = 0.2,
        service_rate: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialise the traffic‑intersection environment.

        Parameters
        ----------
        max_steps: int, optional
            Maximum number of steps per episode before termination.
        arrival_rate: float, optional
            Mean number of vehicles arriving per lane per step (Poisson λ).
        service_rate: int, optional
            Maximum number of vehicles that can depart from a green lane each step.
        seed: int, optional
            Random seed for reproducibility.
        """
        super().__init__()

        self.max_steps = int(max_steps)
        self.arrival_rate = float(arrival_rate)
        self.service_rate = int(service_rate)

        # Observation: queue length on each of the four lanes (non‑negative ints)
        self.observation_space = spaces.Box(
            low=0,
            high=np.iinfo(np.int32).max,
            shape=(4,),
            dtype=np.int32,
        )

        # Action: choose traffic‑light phase (0 = NS green, 1 = EW green)
        self.action_space = spaces.Discrete(2)

        # Random number generator – will be seeded in ``reset``
        self.np_random: np.random.Generator = np.random.default_rng(seed)

        # Internal state
        self.queues: np.ndarray = np.zeros(4, dtype=np.int32)  # N, E, S, W
        self.current_step: int = 0
        self.last_action: Optional[int] = None

        # Rendering utilities
        self._fig: Optional[plt.Figure] = None
        self._ax: Optional[plt.Axes] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.

        Parameters
        ----------
        seed: int, optional
            Seed for the internal RNG.  If ``None`` the existing RNG state is kept.
        options: dict, optional
            Additional reset options (ignored).

        Returns
        -------
        observation: np.ndarray
            Initial queue lengths (all zeros).
        info: dict
            Empty dictionary (placeholder for compatibility).
        """
        super().reset(seed=seed)

        # Re‑initialise the RNG with the provided seed (if any)
        self.np_random = np.random.default_rng(seed)

        self.queues = np.zeros(4, dtype=np.int32)
        self.current_step = 0
        self.last_action = None

        return self.queues.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step of the environment's dynamics.

        Parameters
        ----------
        action: int
            Traffic‑light phase chosen by the agent (0 or 1).

        Returns
        -------
        observation: np.ndarray
            Updated queue lengths after arrivals and departures.
        reward: float
            Negative total queue length (the smaller the queues, the higher the reward).
        terminated: bool
            ``True`` if ``max_steps`` have been reached.
        truncated: bool
            Always ``False`` (no external truncation logic).
        info: dict
            Additional diagnostic information (e.g., arrivals, departures).
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.last_action = action

        # 1. Vehicle arrivals (Poisson process)
        arrivals = self.np_random.poisson(lam=self.arrival_rate, size=4).astype(np.int32)

        # 2. Vehicle departures according to the selected phase
        green_lanes = self._PHASE_MAP[action]
        departures = np.zeros(4, dtype=np.int32)
        for lane in green_lanes:
            # Number of vehicles that can leave is limited by the queue length
            # and the service capacity.
            departures[lane] = min(self.queues[lane], self.service_rate)

        # 3. Update queues
        self.queues = self.queues + arrivals - departures
        # Ensure no negative queue lengths (numerical safety)
        self.queues = np.maximum(self.queues, 0)

        # 4. Compute reward (negative total waiting vehicles)
        reward = -float(self.queues.sum())

        # 5. Episode termination logic
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False  # No external truncation condition

        # 6. Assemble info dict
        info = {
            "step": self.current_step,
            "action": action,
            "arrivals": arrivals,
            "departures": departures,
            "queues": self.queues.copy(),
        }

        return self.queues.copy(), reward, terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Render the current state of the environment.

        In ``human`` mode a Matplotlib window with a bar chart of the queue
        lengths is displayed.  The method returns ``None`` in this mode.
        For ``rgb_array`` mode a NumPy array representing the image is returned.

        Parameters
        ----------
        mode: str
            Rendering mode – either ``"human"`` or ``"rgb_array"``.

        Returns
        -------
        image: Optional[np.ndarray]
            RGB image array when ``mode == "rgb_array"``, otherwise ``None``.
        """
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {mode}")

        lane_labels = ["North", "East", "South", "West"]
        queue_vals = self.queues

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 4))
        else:
            self._ax.clear()

        self._ax.bar(lane_labels, queue_vals, color="tab:blue")
        self._ax.set_ylim(0, max(10, queue_vals.max() + 5))
        self._ax.set_ylabel("Queue length")
        self._ax.set_title(f"Step {self.current_step}")

        plt.tight_layout()

        if mode == "human":
            plt.pause(0.001)  # Required for interactive backends
            return None
        elif mode == "rgb_array":
            self._fig.canvas.draw()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self) -> None:
        """Close the rendering window and release resources."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_steps={self.max_steps}, "
            f"arrival_rate={self.arrival_rate}, service_rate={self.service_rate})"
        )