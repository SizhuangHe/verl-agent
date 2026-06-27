"""
ALFWorld predicates (Route D, gemini-3.1-pro-preview, attempt 5: 10s+10f)
"""

import re
from typing import Callable

# Predicate functions (ordered by expected progression)

def check_object_located(observation: str) -> bool:
    """P1: Agent found a receptacle containing at least one object (not empty)."""
    return bool(re.search(r"(on the .*, you see a |in it, you see a )", observation.lower()))

def check_object_acquired(observation: str) -> bool:
    """P2: Agent successfully picked up an object from a receptacle."""
    return bool(re.search(r"you pick up the [a-z]+ \d+ from the", observation.lower()))

def check_object_examined_or_receptacle_opened(observation: str) -> bool:
    """P3: Agent examined an object or opened a closed receptacle (advanced interaction)."""
    pattern = r"(this is a normal|there's nothing special about|you open the [a-z]+ \d+\. the [a-z]+ \d+ is open)"
    return bool(re.search(pattern, observation.lower()))

def check_intermediate_placement_or_modification_prep(observation: str) -> bool:
    """P4: Agent placed an object or used a modifier appliance (clean/cool/heat/turn on)."""
    pattern = r"(you put the .* in/on the|you (clean|cool|heat) the|you turn on the)"
    return bool(re.search(pattern, observation.lower()))

def check_object_state_modified(observation: str) -> bool:
    """P5: Environment confirms object state is modified (clean/cold/hot)."""
    pattern = r"this is a (clean|cold|hot)"
    return bool(re.search(pattern, observation.lower()))


# Ordered list of all checkpoints
CHECKPOINTS: list[Callable[[str], bool]] = [
    check_object_located,
    check_object_acquired,
    check_object_examined_or_receptacle_opened,
    check_intermediate_placement_or_modification_prep,
    check_object_state_modified
]

CHECKPOINT_DESCRIPTIONS: list[str] = [
    "P1: Agent located a receptacle containing at least one object.",
    "P2: Agent successfully acquired an object from a receptacle.",
    "P3: Agent performed advanced interaction (examined object or opened receptacle).",
    "P4: Agent performed late-stage interaction (placed object or used modifier appliance).",
    "P5: Agent verified modified physical state of object (clean/cold/hot)."
]

def get_checkpoints() -> list[Callable[[str], bool]]:
    return CHECKPOINTS.copy()

def describe_checkpoints() -> list[tuple[str, str]]:
    return list(zip([f.__name__ for f in CHECKPOINTS], CHECKPOINT_DESCRIPTIONS))
