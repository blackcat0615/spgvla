from pathlib import Path

from lehome.utils.constant import ASSETS_ROOT

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg


"""Configuration for the Kitchen Scene"""
SCENES_ROOT = Path(ASSETS_ROOT) / "scenes"


MARBLE_BEDROOM_USD_PATH = str(SCENES_ROOT / "marble" / "Scene_00_Apartment.usd")

MARBLE_BEDROOM_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=MARBLE_BEDROOM_USD_PATH,
    )
)
