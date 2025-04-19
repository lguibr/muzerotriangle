# File: muzerotriangle/nn/network.py
import logging
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from ..config import EnvConfig, ModelConfig, TrainConfig
from ..environment import GameState
from ..features import extract_state_features
from ..utils.types import ActionType, PolicyValueOutput, StateType
from .model import MuZeroNet  # Import MuZeroNet

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


class NetworkEvaluationError(Exception):
    """Custom exception for errors during network evaluation."""

    pass


class NeuralNetwork:
    """
    Wrapper for the MuZeroNet model providing methods for representation,
    dynamics, and prediction, as well as initial inference.
    Handles distributional value/reward heads.
    Optionally compiles the model using torch.compile().
    """

    def __init__(
        self,
        model_config: ModelConfig,
        env_config: EnvConfig,
        train_config: TrainConfig,
        device: torch.device,
    ):
        self.model_config = model_config
        self.env_config = env_config
        self.train_config = train_config
        self.device = device
        self.model = MuZeroNet(model_config, env_config).to(device)
        self.action_dim = env_config.ACTION_DIM
        self.model.eval()

        # Distributional Value Head parameters
        self.num_value_atoms = model_config.NUM_VALUE_ATOMS
        self.v_min = model_config.VALUE_MIN
        self.v_max = model_config.VALUE_MAX
        if self.num_value_atoms <= 1:
            raise ValueError("NUM_VALUE_ATOMS must be greater than 1")
        self.delta_z = (self.v_max - self.v_min) / (self.num_value_atoms - 1)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_value_atoms, device=self.device
        )

        # Distributional Reward Head parameters (assuming symmetric support around 0)
        self.num_reward_atoms = model_config.REWARD_SUPPORT_SIZE
        if self.num_reward_atoms <= 1:
            raise ValueError("REWARD_SUPPORT_SIZE must be greater than 1")
        # Calculate reward min/max based on support size (e.g., size 21 -> -10 to 10)
        self.r_max = float((self.num_reward_atoms - 1) // 2)
        self.r_min = -self.r_max
        self.delta_r = 1.0  # Assuming integer steps for reward support
        self.reward_support = torch.linspace(
            self.r_min, self.r_max, self.num_reward_atoms, device=self.device
        )

        # Compile model if requested and compatible
        self._try_compile_model()

    def _try_compile_model(self):
        """Attempts to compile the model if configured and compatible."""
        if not self.train_config.COMPILE_MODEL:
            logger.info("Model compilation skipped (COMPILE_MODEL=False).")
            return

        if sys.platform == "win32":
            logger.warning("Model compilation skipped on Windows (Triton dependency).")
            return
        if self.device.type == "mps":
            logger.warning("Model compilation skipped on MPS (compatibility issues).")
            return
        if not hasattr(torch, "compile"):
            logger.warning(
                "Model compilation skipped (torch.compile not available)."
            )
            return

        try:
            logger.info(
                f"Attempting to compile model with torch.compile() on device '{self.device}'..."
            )
            # Compile the underlying MuZeroNet instance
            self.model = torch.compile(self.model)  # type: ignore
            logger.info(f"Model compiled successfully on device '{self.device}'.")
        except Exception as e:
            logger.warning(
                f"torch.compile() failed on device '{self.device}': {e}. Proceeding without compilation.",
                exc_info=False,
            )

    def _state_to_tensors(self, state: GameState) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from GameState and converts them to tensors."""
        state_dict: StateType = extract_state_features(state, self.model_config)
        grid_tensor = torch.from_numpy(state_dict["grid"]).unsqueeze(0).to(self.device)
        other_features_tensor = (
            torch.from_numpy(state_dict["other_features"]).unsqueeze(0).to(self.device)
        )
        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError("Non-finite values in input grid_tensor")
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values in input other_features_tensor"
            )
        return grid_tensor, other_features_tensor

    def _batch_states_to_tensors(
        self, states: list[GameState]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts features from a batch of GameStates and converts to batched tensors."""
        if not states:
            grid_shape = (
                0,
                self.model_config.GRID_INPUT_CHANNELS,
                self.env_config.ROWS,
                self.env_config.COLS,
            )
            other_shape = (0, self.model_config.OTHER_NN_INPUT_FEATURES_DIM)
            return torch.empty(grid_shape, device=self.device), torch.empty(
                other_shape, device=self.device
            )

        batch_grid = []
        batch_other = []
        for state in states:
            state_dict: StateType = extract_state_features(state, self.model_config)
            batch_grid.append(state_dict["grid"])
            batch_other.append(state_dict["other_features"])

        grid_tensor = torch.from_numpy(np.stack(batch_grid)).to(self.device)
        other_features_tensor = torch.from_numpy(np.stack(batch_other)).to(self.device)

        if not torch.all(torch.isfinite(grid_tensor)):
            raise NetworkEvaluationError("Non-finite values in batched grid_tensor")
        if not torch.all(torch.isfinite(other_features_tensor)):
            raise NetworkEvaluationError(
                "Non-finite values in batched other_features_tensor"
            )
        return grid_tensor, other_features_tensor

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts logits to probabilities using softmax."""
        return F.softmax(logits, dim=-1)

    def _logits_to_scalar(
        self, logits: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the expected scalar value from distribution logits."""
        probs = self._logits_to_probs(logits)
        # Expand support to match batch size if needed
        support_expanded = support.expand_as(probs)
        scalar = torch.sum(probs * support_expanded, dim=-1)
        return scalar

    @torch.inference_mode()
    def initial_inference(
        self, observation: StateType
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the initial inference h(o) -> s_0 and f(s_0) -> p_0, v_0.
        Args:
            observation: The StateType dictionary from feature extraction.
        Returns:
            Tuple: (policy_logits, value_logits, reward_logits (dummy), initial_hidden_state)
                   Reward logits are dummy here as they come from dynamics.
        """
        self.model.eval()
        grid_tensor = (
            torch.from_numpy(observation["grid"]).unsqueeze(0).to(self.device)
        )
        other_features_tensor = (
            torch.from_numpy(observation["other_features"])
            .unsqueeze(0)
            .to(self.device)
        )

        policy_logits, value_logits, initial_hidden_state = self.model(
            grid_tensor, other_features_tensor
        )

        # Create dummy reward logits (batch_size, num_reward_atoms)
        # Initial state doesn't have a predicted reward from dynamics
        dummy_reward_logits = torch.zeros(
            (1, self.num_reward_atoms), device=self.device
        )

        return policy_logits, value_logits, dummy_reward_logits, initial_hidden_state

    @torch.inference_mode()
    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: ActionType | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs one step of recurrent inference:
        g(s_{k-1}, a_k) -> s_k, r_k
        f(s_k) -> p_k, v_k
        Args:
            hidden_state: The previous hidden state (s_{k-1}).
            action: The action taken (a_k).
        Returns:
            Tuple: (policy_logits, value_logits, reward_logits, next_hidden_state)
        """
        self.model.eval()
        next_hidden_state, reward_logits = self.model.dynamics(hidden_state, action)
        policy_logits, value_logits = self.model.predict(next_hidden_state)
        return policy_logits, value_logits, reward_logits, next_hidden_state

    # --- Compatibility methods for MCTS/Workers expecting PolicyValueOutput ---
    # These now perform initial inference.

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> PolicyValueOutput:
        """
        Evaluates a single state using initial inference (h + f).
        Returns policy mapping and EXPECTED scalar value from the distribution.
        """
        self.model.eval()
        try:
            # 1. Feature Extraction
            state_dict: StateType = extract_state_features(state, self.model_config)
            # 2. Initial Inference
            policy_logits, value_logits, _, _ = self.initial_inference(state_dict)

            # 3. Process Outputs
            policy_probs_tensor = self._logits_to_probs(policy_logits)
            expected_value_tensor = self._logits_to_scalar(value_logits, self.support)

            # Validate and normalize policy probabilities
            policy_probs = policy_probs_tensor.squeeze(0).cpu().numpy()
            if not np.all(np.isfinite(policy_probs)):
                raise NetworkEvaluationError(
                    f"Non-finite policy probabilities AFTER softmax for state {state.current_step}."
                )
            policy_probs = np.maximum(policy_probs, 0)
            prob_sum = np.sum(policy_probs)
            if abs(prob_sum - 1.0) > 1e-5:
                logger.warning(
                    f"Evaluate: Policy probabilities sum to {prob_sum:.6f}. Re-normalizing."
                )
                if prob_sum <= 1e-9:
                    policy_probs.fill(1.0 / len(policy_probs))
                else:
                    policy_probs /= prob_sum

            # Convert to expected output format
            action_policy: Mapping[ActionType, float] = {
                i: float(p) for i, p in enumerate(policy_probs)
            }
            expected_value_scalar = expected_value_tensor.item()

            return action_policy, expected_value_scalar

        except Exception as e:
            logger.error(f"Exception during single evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(
                f"Evaluation failed for state {state}: {e}"
            ) from e

    @torch.inference_mode()
    def evaluate_batch(self, states: list[GameState]) -> list[PolicyValueOutput]:
        """
        Evaluates a batch of states using initial inference (h + f).
        Returns a list of (policy mapping, EXPECTED scalar value).
        """
        if not states:
            return []
        self.model.eval()
        try:
            # 1. Batch Feature Extraction
            grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)

            # 2. Batch Initial Inference (using model's forward)
            policy_logits, value_logits, _ = self.model(
                grid_tensor, other_features_tensor
            )

            # 3. Batch Process Outputs
            policy_probs_tensor = self._logits_to_probs(policy_logits)
            expected_values_tensor = self._logits_to_scalar(value_logits, self.support)

            # Validate and normalize policies
            policy_probs = policy_probs_tensor.cpu().numpy()
            expected_values = expected_values_tensor.cpu().numpy()

            results: list[PolicyValueOutput] = []
            for batch_idx in range(len(states)):
                probs_i = policy_probs[batch_idx]
                if not np.all(np.isfinite(probs_i)):
                    raise NetworkEvaluationError(
                        f"Non-finite policy probabilities AFTER softmax for batch item {batch_idx}."
                    )
                probs_i = np.maximum(probs_i, 0)
                prob_sum_i = np.sum(probs_i)
                if abs(prob_sum_i - 1.0) > 1e-5:
                    logger.warning(
                        f"EvaluateBatch: Policy probs sum to {prob_sum_i:.6f} for item {batch_idx}. Re-normalizing."
                    )
                    if prob_sum_i <= 1e-9:
                        probs_i.fill(1.0 / len(probs_i))
                    else:
                        probs_i /= prob_sum_i

                policy_i: Mapping[ActionType, float] = {
                    i: float(p) for i, p in enumerate(probs_i)
                }
                value_i = float(expected_values[batch_idx])
                results.append((policy_i, value_i))

            return results

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}", exc_info=True)
            raise NetworkEvaluationError(f"Batch evaluation failed: {e}") from e

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Returns the model's state dictionary, moved to CPU."""
        model_to_save = getattr(self.model, "_orig_mod", self.model)
        return {k: v.cpu() for k, v in model_to_save.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]):
        """Loads the model's state dictionary from the provided weights."""
        try:
            weights_on_device = {k: v.to(self.device) for k, v in weights.items()}
            model_to_load = getattr(self.model, "_orig_mod", self.model)
            model_to_load.load_state_dict(weights_on_device)
            self.model.eval()
            logger.debug("NN weights set successfully.")
        except Exception as e:
            logger.error(f"Error setting weights on NN instance: {e}", exc_info=True)
            raise