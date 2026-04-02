"""
Dataset management tool for operations requiring Isaac Sim (SimulationApp).

This script handles dataset operations that require the Isaac Sim application to be running:
- record: Record teleoperation data
- replay: Replay dataset in simulation
"""

import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

from utils import common, setup_record_parser, setup_replay_parser
from lehome.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main entry point for dataset operations requiring Isaac Sim."""
    isaac_args_parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(isaac_args_parser)

    parser = argparse.ArgumentParser(
        description="LeHome dataset management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    # Setup all subcommand parsers first
    setup_record_parser(subparsers, [isaac_args_parser])
    setup_replay_parser(subparsers, [isaac_args_parser])

    args = parser.parse_args()
    simulation_app = common.launch_app_from_args(args)

    try:
        import lehome.tasks.bedroom
        from utils import dataset_record, dataset_replay

        if args.command == "record":
            dataset_record.record_dataset(args, simulation_app)
        elif args.command == "replay":
            dataset_replay.replay(args)
    except Exception as e:
        logger.error(f"Error during dataset {args.command}: {e}", exc_info=True)
        raise
    finally:
        common.close_app(simulation_app)


if __name__ == "__main__":
    main()
