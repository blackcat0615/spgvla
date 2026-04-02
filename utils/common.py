import argparse
from typing import TYPE_CHECKING
import numpy as np
import torch

from isaaclab.app import AppLauncher
from isaacsim.simulation_app import SimulationApp

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

SINGLE_ARM_HOME_POSITION = np.array(
    [
        -1.0363,  # shoulder_pan
        -1.7135,  # shoulder_lift
        1.4979,  # elbow_flex
        1.0534,  # wrist_flex
        -0.085,  # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)

# Left arm uses standard home position
LEFT_ARM_HOME_POSITION = np.array(
    [
        -1.2363,  # shoulder_pan
        -1.7135,  # shoulder_lift
        1.4979,  # elbow_flex
        1.0534,  # wrist_flex
        -0.085,  # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)
# Right arm with symmetric shoulder_pan
RIGHT_ARM_HOME_POSITION = np.array(
    [
        1.2363,  # shoulder_pan
        -1.7135,  # shoulder_lift
        1.4979,  # elbow_flex
        1.0534,  # wrist_flex
        -0.085,  # wrist_roll
        -0.01176,  # gripper
    ],
    dtype=np.float32,
)
DUAL_ARM_HOME_POSITION = np.concatenate(
    [LEFT_ARM_HOME_POSITION, RIGHT_ARM_HOME_POSITION]
)


def launch_app(parser: argparse.ArgumentParser) -> SimulationApp:
    """Launch Isaac Sim app from parser (parses args internally).

    Use this when you haven't parsed arguments yet.
    """
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return launch_app_from_args(args)


def launch_app_from_args(args: argparse.Namespace) -> SimulationApp:
    """Launch Isaac Sim app from already parsed arguments.

    Use this when arguments are already parsed (e.g., in subcommand handlers).

    Args:
        args: Already parsed command-line arguments (must include AppLauncher args).

    Returns:
        SimulationApp instance.
    """
    args.kit_args = (
        "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    )
    app_launcher = AppLauncher(vars(args))
    simulation_app = app_launcher.app
    return simulation_app


def close_app(simulation_app: SimulationApp) -> None:
    """Close Isaac Sim app."""
    simulation_app.close()


def stabilize_garment_after_reset(
    env: "DirectRLEnv",
    args: argparse.Namespace,
    num_steps: int = 20,
) -> None:
    """Stabilize garment after environment reset by running physics steps.

    Moves robot to home position and lets garment settle naturally after reset,
    preventing floating or clipping. This is critical for garment physics to
    initialize properly, especially when using CUDA device.

    Args:
        env: Environment instance.
        args: Command-line arguments containing task name.
        num_steps: Number of stabilization steps to run.
    """
    if num_steps <= 0:
        return

    is_bimanual = "Bi" in args.task or "bi" in args.task.lower()

    try:
        initial_obs = env._get_observations()
        action_dim = (
            len(initial_obs["observation.state"])
            if "observation.state" in initial_obs
            else (12 if is_bimanual else 6)
        )
    except Exception:
        action_dim = 12 if is_bimanual else 6

    home_joints = DUAL_ARM_HOME_POSITION if is_bimanual else SINGLE_ARM_HOME_POSITION

    if len(home_joints) != action_dim:
        # Use warning from logger if available, otherwise print
        try:
            from lehome.utils.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"Home position dimension mismatch: got {len(home_joints)}, "
                f"expected {action_dim}. Using zeros."
            )
        except Exception:
            pass
        home_action = torch.zeros(1, action_dim, dtype=torch.float32, device=env.device)
    else:
        home_action = torch.from_numpy(home_joints).float().to(env.device).unsqueeze(0)

    for step_idx in range(num_steps):
        env.step(home_action)
        if (step_idx + 1) % 10 == 0 or step_idx == num_steps - 1:
            env.render()
