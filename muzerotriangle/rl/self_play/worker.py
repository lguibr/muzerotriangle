# File: muzerotriangle/rl/self_play/worker.py
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import ray

from ...environment import GameState
from ...features import extract_state_features
from ...mcts import (
    MCTSExecutionError,
    Node,
    get_policy_target,
    run_mcts_simulations,
    select_action_based_on_visits,
)
from ...nn import NeuralNetwork
from ...utils import get_device, set_random_seeds
from ..types import SelfPlayResult

if TYPE_CHECKING:
    import torch

    from ...utils.types import (
        StepInfo,
        Trajectory,
        TrajectoryStep,
    )

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """MuZero self-play worker."""

    # ... (init, set_weights, set_current_trainer_step, _report_current_state, _log_step_stats_async remain the same) ...
    def __init__(
        self,
        actor_id,
        env_config,
        mcts_config,
        model_config,
        train_config,
        stats_collector_actor,
        initial_weights,
        seed,
        worker_device_str,
    ):
        self.actor_id = actor_id
        self.env_config = env_config
        self.mcts_config = mcts_config
        self.model_config = model_config
        self.train_config = train_config
        self.stats_collector_actor = stats_collector_actor
        self.seed = seed if seed is not None else random.randint(0, 1_000_000)
        self.worker_device_str = worker_device_str
        self.current_trainer_step = 0
        worker_log_level = logging.INFO
        log_format = (
            f"%(asctime)s [%(levelname)s] [W{self.actor_id}] %(name)s: %(message)s"
        )
        logging.basicConfig(level=worker_log_level, format=log_format, force=True)
        global logger
        logger = logging.getLogger(__name__)
        logging.getLogger("muzerotriangle.mcts").setLevel(logging.WARNING)
        logging.getLogger("muzerotriangle.nn").setLevel(logging.WARNING)
        set_random_seeds(self.seed)
        self.device = get_device(self.worker_device_str)
        self.nn_evaluator = NeuralNetwork(
            model_config, env_config, train_config, self.device
        )
        if initial_weights:
            self.set_weights(initial_weights)
        else:
            self.nn_evaluator.model.eval()
        logger.info(
            f"Worker {actor_id} initialized on device {self.device}. Seed: {self.seed}."
        )

    def set_weights(self, weights):
        try:
            self.nn_evaluator.set_weights(weights)
            logger.debug(f"W{self.actor_id}: Weights updated.")
        except Exception as e:
            logger.error(f"W{self.actor_id}: Failed set weights: {e}", exc_info=True)

    def set_current_trainer_step(self, global_step):
        self.current_trainer_step = global_step
        logger.debug(f"W{self.actor_id}: Trainer step set {global_step}")

    def _report_current_state(self, game_state):
        if self.stats_collector_actor:
            try:
                state_copy = game_state.copy()
                self.stats_collector_actor.update_worker_game_state.remote(
                    self.actor_id, state_copy
                )  # type: ignore
            except Exception as e:
                logger.error(f"W{self.actor_id}: Failed report state: {e}")

    def _log_step_stats_async(self, game_step, mcts_visits, mcts_depth, step_reward):
        if self.stats_collector_actor:
            try:
                step_info: StepInfo = {
                    "game_step_index": game_step,
                    "global_step": self.current_trainer_step,
                }
                step_stats: dict[str, tuple[float, StepInfo]] = {
                    "MCTS/Step_Visits": (float(mcts_visits), step_info),
                    "MCTS/Step_Depth": (float(mcts_depth), step_info),
                    "RL/Step_Reward": (step_reward, step_info),
                }
                self.stats_collector_actor.log_batch.remote(step_stats)  # type: ignore
            except Exception as e:
                logger.error(f"W{self.actor_id}: Failed log step stats: {e}")

    def run_episode(self) -> SelfPlayResult:
        """Runs a single MuZero self-play episode."""
        self.nn_evaluator.model.eval()
        episode_seed = self.seed + random.randint(0, 1000)
        game = GameState(self.env_config, initial_seed=episode_seed)
        trajectory: Trajectory = []
        step_root_visits: list[int] = []
        step_tree_depths: list[int] = []
        step_simulations: list[int] = []
        current_hidden_state: torch.Tensor | None = None

        logger.info(f"W{self.actor_id}: Starting episode seed {episode_seed}")
        self._report_current_state(game)

        while not game.is_over():
            game_step = game.current_step
            try:
                observation = extract_state_features(game, self.model_config)
                valid_actions = game.valid_actions()
                if not valid_actions:
                    logger.warning(
                        f"W{self.actor_id}: No valid actions step {game_step}"
                    )
                    break
            except Exception as e:
                logger.error(
                    f"W{self.actor_id}: Feat/Action error step {game_step}: {e}",
                    exc_info=True,
                )
                break

            if current_hidden_state is None:
                _, _, _, hidden_state_tensor = self.nn_evaluator.initial_inference(
                    observation
                )
                current_hidden_state = hidden_state_tensor.squeeze(0)
                root_node = Node(
                    hidden_state=current_hidden_state, initial_game_state=game.copy()
                )
            else:
                root_node = Node(
                    hidden_state=current_hidden_state, initial_game_state=game.copy()
                )

            mcts_max_depth = 0
            try:
                mcts_max_depth = run_mcts_simulations(
                    root_node, self.mcts_config, self.nn_evaluator, valid_actions
                )
                step_root_visits.append(root_node.visit_count)
                step_tree_depths.append(mcts_max_depth)
                step_simulations.append(self.mcts_config.num_simulations)
            except MCTSExecutionError as mcts_err:
                logger.error(
                    f"W{self.actor_id} Step {game_step}: MCTS failed: {mcts_err}"
                )
                break

            try:
                temp = (
                    self.mcts_config.temperature_initial
                    if game_step < self.mcts_config.temperature_anneal_steps
                    else self.mcts_config.temperature_final
                )
                action = select_action_based_on_visits(root_node, temperature=temp)
                policy_target = get_policy_target(root_node, temperature=1.0)
                value_target = root_node.value_estimate
            except Exception as policy_err:
                logger.error(
                    f"W{self.actor_id} Step {game_step}: Policy/Action failed: {policy_err}",
                    exc_info=True,
                )
                break

            step_data_partial = {
                "observation": observation,
                "action": action,
                "policy_target": policy_target,
                "value_target": value_target,
            }
            real_reward, done = game.step(action)

            step_data_complete: TrajectoryStep = {
                **step_data_partial,
                "reward": real_reward,
                "hidden_state": current_hidden_state.cpu().numpy()
                if current_hidden_state is not None
                else None,
            }  # type: ignore
            trajectory.append(step_data_complete)

            if not done:
                try:
                    if current_hidden_state is not None:
                        # --- Call dynamics via underlying model ---
                        next_hidden_state_tensor, _ = self.nn_evaluator.model.dynamics(
                            current_hidden_state.unsqueeze(0), action
                        )
                        current_hidden_state = next_hidden_state_tensor.squeeze(0)
                    else:
                        logger.error(
                            f"W{self.actor_id} Step {game_step}: hidden_state is None"
                        )
                        break
                except Exception as dyn_err:
                    logger.error(
                        f"W{self.actor_id} Step {game_step}: Dynamics error: {dyn_err}",
                        exc_info=True,
                    )
                    break
            else:
                current_hidden_state = None

            self._report_current_state(game)
            self._log_step_stats_async(
                game_step, root_node.visit_count, mcts_max_depth, real_reward
            )
            if done:
                break
        # --- Episode End ---
        final_score = game.game_score
        total_steps_episode = game.current_step
        logger.info(
            f"W{self.actor_id}: Episode finished. Score: {final_score:.2f}, Steps: {total_steps_episode}"
        )
        total_sims = sum(step_simulations)
        avg_visits = np.mean(step_root_visits) if step_root_visits else 0.0
        avg_depth = np.mean(step_tree_depths) if step_tree_depths else 0.0
        if not trajectory:
            logger.warning(f"W{self.actor_id}: Episode finished empty trajectory.")
        return SelfPlayResult(
            trajectory=trajectory,
            final_score=final_score,
            episode_steps=total_steps_episode,
            total_simulations=total_sims,
            avg_root_visits=float(avg_visits),
            avg_tree_depth=float(avg_depth),
        )
