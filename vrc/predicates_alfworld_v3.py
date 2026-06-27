"""
ALFWorld predicates (Route D, gemini-3.1-pro-preview)
"""

import re
from typing import Callable

# Predicate functions (ordered by expected progression)

def check_pickup_object(observation: str) -> bool:
    """
    P1: Detects when the agent successfully picks up an object.
    This is the first universal milestone across all ALFWorld tasks.
    It is highly discriminative as it shows the agent has found and retrieved an item,
    moving beyond aimless navigation.
    """
    return bool(re.search(r"you pick up the ", observation.lower()))

def check_arrive_modifier(observation: str) -> bool:
    """
    P2: Detects when the agent navigates to a receptacle capable of modifying an object's state.
    For Clean, Cool, Heat, and Examine tasks, the agent must visit a specific modifier 
    (sinkbasin, bathtubbasin, fridge, microwave, stoveburner, toaster, or a location with a desklamp).
    This milestone rewards finding the necessary tool for the task.
    """
    pattern = r"you arrive at loc .*(sinkbasin|bathtubbasin|fridge|microwave|stoveburner|toaster|desklamp)"
    return bool(re.search(pattern, observation.lower()))

def check_modify_object(observation: str) -> bool:
    """
    P3: Detects when the agent successfully modifies the state of an object or turns on a device.
    This captures the irreversible state changes required in Clean, Cool, Heat, and Examine tasks.
    It is 100% discriminative in the provided data (failed agents never successfully modify objects).
    """
    pattern = r"you (clean|cool|heat) the .* using the |you turn on the desklamp"
    return bool(re.search(pattern, observation.lower()))

def check_verify_modification(observation: str) -> bool:
    """
    P4: Detects when the agent examines an object and verifies its newly modified state.
    While optional, successful agents sometimes verify the object is 'clean', 'cold', or 'hot' 
    before placing it. This represents the final visible state of progress before the terminal placement.
    """
    pattern = r"this is a (clean|cold|hot) "
    return bool(re.search(pattern, observation.lower()))


# Ordered list of all checkpoints
CHECKPOINTS: list[Callable[[str], bool]] = [
    check_pickup_object,
    check_arrive_modifier,
    check_modify_object,
    check_verify_modification
]

CHECKPOINT_DESCRIPTIONS: list[str] = [
    "P1: Agent picks up an object.",
    "P2: Agent arrives at a modifying receptacle (sink, fridge, microwave, desklamp, etc.).",
    "P3: Agent modifies the object (clean/cool/heat) or turns on the desklamp.",
    "P4: Agent verifies the modified state of the object (clean/cold/hot)."
]

def get_checkpoints() -> list[Callable[[str], bool]]:
    return CHECKPOINTS.copy()

def describe_checkpoints() -> list[tuple[str, str]]:
    return list(zip([f.__name__ for f in CHECKPOINTS], CHECKPOINT_DESCRIPTIONS))
