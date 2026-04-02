# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Conditional import: register tasks and UI only in Isaac Sim environment
# This allows utils module to be used independently in plain Python environment
try:
    import omni.kit.app
    # In Isaac Sim environment, register Gym environments and UI extensions
    from .tasks import *
    from .ui_extension_example import *
except ImportError:
    # Not in Isaac Sim environment, skip tasks/UI registration
    # utils module can still be imported and used normally
    pass
