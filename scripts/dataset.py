"""
Dataset management tool for operations NOT requiring Isaac Sim (SimulationApp).

This script handles dataset operations that do not require the Isaac Sim application:
- inspect: Inspect dataset metadata
- read: Read dataset states
- augment: Add end-effector pose to dataset
- merge: Merge multiple datasets

For operations requiring Isaac Sim (record, replay), use dataset_sim.py instead."""

import argparse
from pathlib import Path

from utils import (
    dataset_inspection,
    dataset_processing,
    setup_inspect_parser,
    setup_read_parser,
    setup_augment_parser,
    setup_merge_parser,
)


def main():
    """Main entry point for dataset management tool (no Isaac Sim required)."""
    parser = argparse.ArgumentParser(
        description="LeHome dataset management tool (no Isaac Sim required)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Note: For operations requiring Isaac Sim (record, replay), use dataset_sim.py instead.",
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    setup_inspect_parser(subparsers)
    setup_read_parser(subparsers)
    setup_augment_parser(subparsers)
    setup_merge_parser(subparsers)

    args = parser.parse_args()

    if args.command == "inspect":
        dataset_inspection.inspect(
            Path(args.dataset_root),
            show_frames=args.show_frames,
            show_stats=args.show_stats,
        )
    elif args.command == "read":
        dataset_inspection.read_states(
            Path(args.dataset_root),
            num_frames=args.num_frames,
            episode=args.episode,
            output_csv=args.output_csv,
            show_stats=args.show_stats,
        )
    elif args.command == "augment":
        dataset_processing.augment_ee_pose(
            Path(args.dataset_root).resolve(),
            Path(args.urdf_path).resolve(),
            state_unit=args.state_unit,
            output_root=Path(args.output_root).resolve() if args.output_root else None,
            overwrite=args.overwrite,
        )
    elif args.command == "merge":
        import ast

        source_roots = [Path(p) for p in ast.literal_eval(args.source_roots)]
        dataset_processing.merge_datasets(
            source_roots,
            Path(args.output_root),
            output_repo_id=args.output_repo_id,
            merge_custom_meta=args.merge_custom_meta,
        )


if __name__ == "__main__":
    main()
