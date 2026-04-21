"""
Prompt registry — one module per prompt version.

Why this exists (thesis Chapter 5):
  RQ2 asks whether iterative prompt engineering improves factual accuracy and
  behavioural alignment. To answer that, we need (a) every prompt version
  preserved verbatim, (b) the ability to run the same query bank through each
  version without code changes, and (c) a run log tying {version → input →
  output} together for the results table.

Registry shape:
  AGENT_PROMPTS       = {"v1": str, "v2": str, "v3": str, "v4": str}
  CURRICULUM_PROMPTS  = {"v1": str, "v2": str}

Use:
  from app.prompts import get_agent_prompt, get_curriculum_prompt
  system_prompt = get_agent_prompt("v4")      # defaults to latest
  parse_prompt  = get_curriculum_prompt("v2") # defaults to latest
"""

from .agent_v1 import PROMPT as AGENT_V1
from .agent_v2 import PROMPT as AGENT_V2
from .agent_v3 import PROMPT as AGENT_V3
from .agent_v4 import PROMPT as AGENT_V4
from .curriculum_v1 import PROMPT as CURRICULUM_V1
from .curriculum_v2 import PROMPT as CURRICULUM_V2

AGENT_PROMPTS = {
    "v1": AGENT_V1,
    "v2": AGENT_V2,
    "v3": AGENT_V3,
    "v4": AGENT_V4,
}

CURRICULUM_PROMPTS = {
    "v1": CURRICULUM_V1,
    "v2": CURRICULUM_V2,
}

AGENT_LATEST      = "v4"
CURRICULUM_LATEST = "v2"


def get_agent_prompt(version: str | None = None) -> str:
    """Return the agent system prompt for a version, falling back to latest."""
    v = version or AGENT_LATEST
    if v not in AGENT_PROMPTS:
        raise ValueError(
            f"unknown agent prompt version '{v}'. "
            f"available: {list(AGENT_PROMPTS.keys())}"
        )
    return AGENT_PROMPTS[v]


def get_curriculum_prompt(version: str | None = None) -> str:
    """Return the curriculum-parser prompt for a version, falling back to latest."""
    v = version or CURRICULUM_LATEST
    if v not in CURRICULUM_PROMPTS:
        raise ValueError(
            f"unknown curriculum prompt version '{v}'. "
            f"available: {list(CURRICULUM_PROMPTS.keys())}"
        )
    return CURRICULUM_PROMPTS[v]


__all__ = [
    "AGENT_PROMPTS", "CURRICULUM_PROMPTS",
    "AGENT_LATEST", "CURRICULUM_LATEST",
    "get_agent_prompt", "get_curriculum_prompt",
]
