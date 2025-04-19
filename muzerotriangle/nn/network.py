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
from .model import MuZeroNet

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

        self.num_value_atoms = model_config.NUM_VALUE_ATOMS
        self.v_min = model_config.VALUE_MIN
        self.v_max = model_config.VALUE_MAX
        if self.num_value_atoms <= 1:
            raise ValueError("NUM_VALUE_ATOMS must be greater than 1")
        self.delta_z = (self.v_max - self.v_min) / (self.num_value_atoms - 1)
        self.support = torch.linspace(
            self.v_min, self.v_max, self.num_value_atoms, device=self.device
        )

        self.num_reward_atoms = model_config.REWARD_SUPPORT_SIZE
        if self.num_reward_atoms <= 1:
            raise ValueError("REWARD_SUPPORT_SIZE must be greater than 1")
        self.r_max = float((self.num_reward_atoms - 1) // 2)
        self.r_min = -self.r_max
        self.delta_r = 1.0
        self.reward_support = torch.linspace(
            self.r_min, self.r_max, self.num_reward_atoms, device=self.device
        )

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
            logger.warning("Model compilation skipped (torch.compile not available).")
            return

        try:
            logger.info(
                f"Attempting to compile model with torch.compile() on device '{self.device}'..."
            )
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
        grid_tensor = torch.as_tensor(
            observation["grid"], dtype=torch.float32, device=self.device
        )
        other_features_tensor = torch.as_tensor(
            observation["other_features"], dtype=torch.float32, device=self.device
        )

        if grid_tensor.dim() == 3:
            grid_tensor = grid_tensor.unsqueeze(0)
        if other_features_tensor.dim() == 1:
            other_features_tensor = other_features_tensor.unsqueeze(0)

        policy_logits, value_logits, initial_hidden_state = self.model(
            grid_tensor, other_features_tensor
        )

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
            hidden_state: The previous hidden state (s_{k-1}). Shape [B, H] or [H].
            action: The action taken (a_k). Can be int or Tensor.
        Returns:
            Tuple: (policy_logits, value_logits, reward_logits, next_hidden_state)
                   All tensors will have a batch dimension.
        """
        self.model.eval()
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        if isinstance(action, int):
            action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)
        elif isinstance(action, torch.Tensor):
            if action.dim() == 0:
                action_tensor = action.unsqueeze(0).to(self.device)
            elif action.dim() == 1:
                action_tensor = action.to(self.device)
            else:
                raise ValueError(f"Unsupported action tensor shape: {action.shape}")
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        if action_tensor.shape[0] != hidden_state.shape[0]:
            if hidden_state.shape[0] == 1 and action_tensor.shape[0] > 1:
                hidden_state = hidden_state.expand(action_tensor.shape[0], -1)
            elif action_tensor.shape[0] == 1 and hidden_state.shape[0] > 1:
                action_tensor = action_tensor.expand(hidden_state.shape[0])
            else:
                raise ValueError(
                    f"Batch size mismatch between hidden_state ({hidden_state.shape[0]}) and action ({action_tensor.shape[0]})"
                )

        # hs_finite = torch.all(torch.isfinite(hidden_state)).item() # Removed log
        # logger.debug(f"[Recurrent Inference] Input HS shape: {hidden_state.shape}, isfinite: {hs_finite}") # Removed log
        # if not hs_finite: logger.warning("[Recurrent Inference] Input hidden_state contains non-finite values!") # Removed log

        next_hidden_state, reward_logits = self.model.dynamics(
            hidden_state, action_tensor
        )
        policy_logits, value_logits = self.model.predict(next_hidden_state)

        # nhs_finite = torch.all(torch.isfinite(next_hidden_state)).item() # Removed log
        # rl_finite = torch.all(torch.isfinite(reward_logits)).item() # Removed log
        # pl_finite = torch.all(torch.isfinite(policy_logits)).item() # Removed log
        # vl_finite = torch.all(torch.isfinite(value_logits)).item() # Removed log
        # logger.debug(f"[Recurrent Inference] Output NHS shape: {next_hidden_state.shape}, isfinite: {nhs_finite}") # Removed log
        # logger.debug(f"[Recurrent Inference] Output RewardLogits shape: {reward_logits.shape}, isfinite: {rl_finite}") # Removed log
        # logger.debug(f"[Recurrent Inference] Output PolicyLogits shape: {policy_logits.shape}, isfinite: {pl_finite}") # Removed log
        # logger.debug(f"[Recurrent Inference] Output ValueLogits shape: {value_logits.shape}, isfinite: {vl_finite}") # Removed log
        # if not (nhs_finite and rl_finite and pl_finite and vl_finite): # Removed log
        #      logger.warning("[Recurrent Inference] Output contains non-finite values!") # Removed log

        return policy_logits, value_logits, reward_logits, next_hidden_state

    @torch.inference_mode()
    def evaluate(self, state: GameState) -> PolicyValueOutput:
        """
        Evaluates a single state using initial inference (h + f).
        Returns policy mapping and EXPECTED scalar value from the distribution.
        """
        self.model.eval()
        try:
            state_dict: StateType = extract_state_features(state, self.model_config)
            policy_logits, value_logits, _, _ = self.initial_inference(state_dict)

            policy_probs_tensor = self._logits_to_probs(policy_logits)
            expected_value_tensor = self._logits_to_scalar(value_logits, self.support)

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
            grid_tensor, other_features_tensor = self._batch_states_to_tensors(states)

            policy_logits, value_logits, _ = self.model(
                grid_tensor, other_features_tensor
            )

            policy_probs_tensor = self._logits_to_probs(policy_logits)
            expected_values_tensor = self._logits_to_scalar(value_logits, self.support)

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
