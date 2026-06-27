"""
ALFWorld predicates (Route D, gemini-3.1-pro-preview)
"""

import re
from typing import Callable

# Predicate functions (ordered by expected progression)

def check_observes_items(observation: str) -> bool:
    """P1: Detects when the agent observes a location containing items (non-empty receptacle).
    This proves the agent is successfully exploring and finding objects, rather than wandering between empty locations.
    """
    return bool(re.search(r"you see a ", observation.lower()))

def check_object_acquired(observation: str) -> bool:
    """P2: Detects when the agent successfully picks up an object.
    This is a universal milestone for all task types and proves the agent has acquired the target item.
    """
    return bool(re.search(r"you pick up the", observation.lower()))

def check_advanced_interaction(observation: str) -> bool:
    """P3: Detects when the agent performs an advanced interaction (opening a receptacle, turning on a light, or modifying an object).
    This rewards progress in tasks more complex than simple Pick & Place.
    """
    return bool(re.search(r"you open the|you turn on the|you clean the|you cool the|you heat the", observation.lower()))

def check_state_modification(observation: str) -> bool:
    """P4: Detects when the agent modifies the state of an object (cleaning, cooling, or heating).
    This rewards the core requirement for Clean & Place, Cool & Place, and Heat & Place tasks.
    """
    return bool(re.search(r"you clean the|you cool the|you heat the", observation.lower()))

def check_thermal_modification(observation: str) -> bool:
    """P5: Detects when the agent performs a thermal modification (cooling or heating).
    This rewards progress in the most complex task types, which require locating specific appliances.
    """
    return bool(re.search(r"you cool the|you heat the", observation.lower()))


# Ordered list of all checkpoints
CHECKPOINTS: list[Callable[[str], bool]] = [
    check_observes_items,
    check_object_acquired,
    check_advanced_interaction,
    check_state_modification,
    check_thermal_modification
]

CHECKPOINT_DESCRIPTIONS: list[str] = [
    "P1: Agent observes a non-empty receptacle containing items.",
    "P2: Agent successfully picks up an object.",
    "P3: Agent performs an advanced environment interaction (open, turn on, or modify).",
    "P4: Agent modifies the state of an object (clean, cool, or heat).",
    "P5: Agent performs a complex thermal modification (cool or heat)."
]

def get_checkpoints() -> list[Callable[[str], bool]]:
    return CHECKPOINTS.copy()

def describe_checkpoints() -> list[tuple[str, str]]:
    return list(zip([f.__name__ for f in CHECKPOINTS], CHECKPOINT_DESCRIPTIONS))
