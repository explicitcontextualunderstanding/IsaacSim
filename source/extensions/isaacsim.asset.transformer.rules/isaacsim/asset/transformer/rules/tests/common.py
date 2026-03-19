"""Shared test data paths for rules tests."""

import os

_TEST_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),
    "data",
    "tests",
)

_UR10E_USD = os.path.join(_TEST_DATA_DIR, "ur10e", "ur10e.usd")
_UR10E_SHOULDER_USD = os.path.join(_TEST_DATA_DIR, "ur10e_shoulder", "ur10e.usda")
_TEST_ADVANCED_USD = os.path.join(_TEST_DATA_DIR, "test_advanced", "usdex", "test_advanced.usda")
_TEST_COLLISION_FROM_VISUALS_USD = os.path.join(
    _TEST_DATA_DIR, "test_collision_from_visuals", "test_collision_from_visuals.usda"
)
_INSPIRE_HAND_DIR = os.path.join(_TEST_DATA_DIR, "inspire_hand")
_INSPIRE_HAND_MATERIALS_USDA = os.path.join(_INSPIRE_HAND_DIR, "inspire_hand_materials.usda")
