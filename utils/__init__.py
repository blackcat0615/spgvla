"""Utility functions for LeHome scripts."""

from . import common
from . import dataset_inspection
from . import dataset_processing
from . import parser

# Note: evaluation, dataset_record and dataset_replay are not imported at module level
# to avoid importing Isaac Sim modules before SimulationApp is launched.
# They should be imported lazily when needed (after SimulationApp is launched).

# Export commonly used functions for convenience
from .parser import (
    setup_record_parser,
    setup_replay_parser,
    setup_inspect_parser,
    setup_read_parser,
    setup_augment_parser,
    setup_merge_parser,
    setup_eval_parser,
)
from .common import launch_app, launch_app_from_args, close_app
from .dataset_inspection import inspect, read_states
from .dataset_processing import (
    augment_ee_pose,
    merge_datasets,
    merge_garment_info,
)

# Note: evaluation functions are not imported at module level to avoid
# importing Isaac Sim modules before SimulationApp is launched.
# Import them lazily when needed: from .utils.evaluation import <function>

__all__ = [
    "setup_record_parser",
    "setup_replay_parser",
    "setup_inspect_parser",
    "setup_read_parser",
    "setup_augment_parser",
    "setup_merge_parser",
    "setup_eval_parser",
    "launch_app",
    "launch_app_from_args",
    "close_app",
    "inspect",
    "read_states",
    "augment_ee_pose",
    "merge_datasets",
    "merge_garment_info",
    # Note: evaluation functions, "replay" and "record_dataset" are not exported
    # at module level to avoid importing Isaac Sim modules before SimulationApp
    # is launched. Import them lazily when needed:
    #   from .utils.evaluation import <function>
    #   from .utils import dataset_replay, dataset_record
]
