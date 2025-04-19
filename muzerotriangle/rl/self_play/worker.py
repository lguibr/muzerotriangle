# File: muzerotriangle/rl/self_play/worker.py
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import ray
import torch  # Keep the import

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
    # import torch # Already imported above

    from ...utils.types import (
        ActionType,  # Import ActionType
        PolicyTargetMapping,  # Import PolicyTargetMapping
        StateType,  # Import StateType
        StepInfo,
        Trajectory,
        TrajectoryStep,
    )

logger = logging.getLogger(__name__)


@ray.remote
class SelfPlayWorker:
    """MuZero self-play worker."""

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
        # Ensure logger is configured for the actor process
        worker_log_level = logging.INFO  # Revert to INFO unless debugging worker
        log_format = (
            f"%(asctime)s [%(levelname)s] [W{self.actor_id}] %(name)s: %(message)s"
        )
        logging.basicConfig(level=worker_log_level, format=log_format, force=True)
        global logger
        logger = logging.getLogger(__name__)
        # Keep MCTS/NN logs less verbose unless needed
        # Set MCTS logger level based on main logger level for debugging
        mcts_log_level = logging.getLogger().level
        logging.getLogger("muzerotriangle.mcts").setLevel(mcts_log_level)
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
                )
            except Exception as e:
                logger.error(f"W{self.actor_id}: Failed report state: {e}")

    def _log_step_stats_async(
        self, game_step, mcts_visits, mcts_depth, step_reward, current_score
    ):
        # Reverted to log_batch
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
                    "RL/Current_Score": (current_score, step_info),
                }
                self.stats_collector_actor.log_batch.remote(step_stats)
                logger.debug(f"W{self.actor_id}: Sent batch log for step {game_step}")
            except Exception as e:
                logger.error(f"W{self.actor_id}: Failed log step stats: {e}")

    def _calculate_n_step_targets(self, trajectory_raw: list[dict]):
        """Calculates N-step reward targets and returns a completed Trajectory."""
        n_steps = self.train_config.N_STEP_RETURNS
        discount = self.train_config.DISCOUNT
        traj_len = len(trajectory_raw)
        completed_trajectory: Trajectory = []

        for t in range(traj_len):
            n_step_reward_target = 0.0
            for i in range(n_steps):
                step_index = t + i
                if step_index < traj_len:
                    n_step_reward_target += (
                        discount**i * trajectory_raw[step_index]["reward"]
                    )
                else:
                    # Bootstrap with the value of the last *actual* state if episode ended early
                    if traj_len > 0:
                        last_step_value = trajectory_raw[-1]["value_target"]
                        n_step_reward_target += discount**i * last_step_value
                    break  # Stop accumulating reward if we are past the end

            # Add the final bootstrap value if the n-step horizon extends beyond the trajectory
            bootstrap_index = t + n_steps
            if (
                bootstrap_index >= traj_len and traj_len > 0
            ):  # Check if bootstrap is needed
                # If the bootstrap index is exactly at the end or beyond, use the last state's value
                last_step_value = trajectory_raw[-1]["value_target"]
                # The power should be n_steps, as it represents the value after n steps
                n_step_reward_target += discount**n_steps * last_step_value
            elif bootstrap_index < traj_len:  # If bootstrap index is within trajectory
                n_step_reward_target += (
                    discount**n_steps * trajectory_raw[bootstrap_index]["value_target"]
                )
            # If trajectory is empty (should not happen if called), target remains 0

            # Create the final TrajectoryStep dict
            step_data: TrajectoryStep = {
                "observation": trajectory_raw[t]["observation"],
                "action": trajectory_raw[t]["action"],
                "reward": trajectory_raw[t]["reward"],
                "policy_target": trajectory_raw[t]["policy_target"],
                "value_target": trajectory_raw[t]["value_target"],
                "n_step_reward_target": n_step_reward_target,  # Add the calculated target
                "hidden_state": trajectory_raw[t]["hidden_state"],
            }
            completed_trajectory.append(step_data)

        return completed_trajectory

    def run_episode(self) -> SelfPlayResult:
        """Runs a single MuZero self-play episode."""
        self.nn_evaluator.model.eval()
        episode_seed = self.seed + random.randint(0, 1000)
        game = GameState(self.env_config, initial_seed=episode_seed)
        trajectory_raw: list[dict] = []  # Store raw data before adding n-step target
        step_root_visits: list[int] = []
        step_tree_depths: list[int] = []
        step_simulations: list[int] = []
        current_hidden_state: torch.Tensor | None = None

        logger.info(f"W{self.actor_id}: Starting episode seed {episode_seed}")
        self._report_current_state(game)

        while not game.is_over():
            game_step = game.current_step
            logger.debug(f"W{self.actor_id} Step {game_step}: Starting step...")
            try:
                observation: StateType = extract_state_features(game, self.model_config)
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
                logger.debug(f"W{self.actor_id} Step {game_step}: Initial inference...")
                try:
                    _, _, _, hidden_state_tensor = self.nn_evaluator.initial_inference(
                        observation
                    )
                    current_hidden_state = hidden_state_tensor.squeeze(0)
                    root_node = Node(
                        hidden_state=current_hidden_state,
                        initial_game_state=game.copy(),
                    )
                    logger.debug(
                        f"W{self.actor_id} Step {game_step}: Initial inference successful."
                    )
                except Exception as inf_err:
                    logger.error(
                        f"W{self.actor_id} Step {game_step}: Initial Inference failed: {inf_err}",
                        exc_info=True,
                    )
                    break  # Stop episode if inference fails
            else:
                logger.debug(
                    f"W{self.actor_id} Step {game_step}: Using existing hidden state."
                )
                root_node = Node(
                    hidden_state=current_hidden_state, initial_game_state=game.copy()
                )

            mcts_max_depth = 0
            logger.debug(f"W{self.actor_id} Step {game_step}: Starting MCTS...")
            try:
                mcts_max_depth = run_mcts_simulations(
                    root_node, self.mcts_config, self.nn_evaluator, valid_actions
                )
                step_root_visits.append(root_node.visit_count)
                step_tree_depths.append(mcts_max_depth)
                step_simulations.append(self.mcts_config.num_simulations)
                logger.info(
                    f"W{self.actor_id} Step {game_step}: MCTS finished. Root visits: {root_node.visit_count}, Max Depth Reached: {mcts_max_depth}, Configured sims: {self.mcts_config.num_simulations}"
                )
            except MCTSExecutionError as mcts_err:
                logger.error(
                    f"W{self.actor_id} Step {game_step}: MCTS failed: {mcts_err}"
                )
                break  # Stop episode if MCTS fails

            logger.debug(f"W{self.actor_id} Step {game_step}: Selecting action...")
            try:
                temp = (
                    self.mcts_config.temperature_initial
                    if game_step < self.mcts_config.temperature_anneal_steps
                    else self.mcts_config.temperature_final
                )
                action: ActionType = select_action_based_on_visits(
                    root_node, temperature=temp
                )
                policy_target: PolicyTargetMapping = get_policy_target(
                    root_node, temperature=1.0
                )
                # Use predicted value from MCTS root for value target
                value_target: float = root_node.value_estimate
                logger.debug(
                    f"W{self.actor_id} Step {game_step}: Action selected: {action}, Value target: {value_target:.3f}"
                )
            except Exception as policy_err:
                logger.error(
                    f"W{self.actor_id} Step {game_step}: Policy/Action failed: {policy_err}",
                    exc_info=True,
                )
                break  # Stop episode if policy selection fails

            real_reward, done = game.step(action)
            logger.debug(
                f"W{self.actor_id} Step {game_step}: Game stepped. Action={action}, Reward={real_reward:.3f}, Done={done}"
            )

            # Store raw step data (n_step_reward_target will be added later)
            step_data_raw: dict = {
                "observation": observation,
                "action": action,
                "reward": real_reward,
                "policy_target": policy_target,
                "value_target": value_target,  # Store MCTS value estimate
                "hidden_state": (
                    current_hidden_state.detach().cpu().numpy()
                    if current_hidden_state is not None
                    else None
                ),
            }
            trajectory_raw.append(step_data_raw)

            if not done:
                logger.debug(
                    f"W{self.actor_id} Step {game_step}: Calling dynamics for next state..."
                )
                try:
                    if current_hidden_state is not None:
                        # Ensure action is compatible with dynamics function
                        action_tensor = torch.tensor(
                            [action], dtype=torch.long, device=self.device
                        )  # Use torch here
                        hs_batch = current_hidden_state.to(self.device).unsqueeze(0)
                        next_hidden_state_tensor, _ = self.nn_evaluator.model.dynamics(
                            hs_batch,
                            action_tensor,  # Pass action tensor
                        )
                        current_hidden_state = next_hidden_state_tensor.squeeze(0)
                        logger.debug(
                            f"W{self.actor_id} Step {game_step}: Dynamics successful."
                        )
                    else:
                        # This case should ideally not be reached if game logic is correct
                        logger.error(
                            f"W{self.actor_id} Step {game_step}: hidden_state is None before dynamics call, but game not done."
                        )
                except Exception as dyn_err:
                    logger.error(
                        f"W{self.actor_id} Step {game_step}: Dynamics error: {dyn_err}",
                        exc_info=True,
                    )
                    break  # Stop episode if dynamics fails
            else:
                current_hidden_state = None  # Reset hidden state for next episode

            self._report_current_state(game)
            # Log stats including current score
            self._log_step_stats_async(
                game_step,
                root_node.visit_count,
                mcts_max_depth,
                real_reward,
                game.game_score,  # Pass current score
            )
            if done:
                logger.debug(f"W{self.actor_id} Step {game_step}: Game ended.")
                break

        # --- Episode End ---
        final_score = game.game_score
        total_steps_episode = game.current_step
        logger.info(
            f"W{self.actor_id}: Episode finished. Score: {final_score:.2f}, Steps: {total_steps_episode}"
        )

        # --- Calculate N-step targets and create final Trajectory ---
        trajectory: Trajectory = self._calculate_n_step_targets(trajectory_raw)
        # ---

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
