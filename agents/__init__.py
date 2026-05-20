from typing import Type, cast

from dotenv import load_dotenv

from .agent import Agent, Playback
from .recorder import Recorder
from .swarm import Swarm
from .templates.langgraph_functional_agent import LangGraphFunc, LangGraphTextOnly
from .templates.langgraph_random_agent import LangGraphRandom
# NOTE: langgraph_thinking broken after Pillow 11->12 upgrade
# (uses ImageDraw.Coords which was removed). We don't use this template
# anyway. Comment out to keep auto-discovery working.
# from .templates.langgraph_thinking import LangGraphThinking
from .templates.llm_agents import LLM, FastLLM, GuidedLLM, ReasoningLLM
from .templates.multimodal import MultiModalLLM
from .templates.nh_arc_baseline_v0 import NhArcBaselineV0  # operator addition
from .templates.nh_arc_official_v0 import NhArcOfficialV0  # operator addition
from .templates.nh_arc_hierarchical_v0 import NhArcHierarchicalV0  # operator addition
from .templates.nh_arc_vision_v0 import NhArcVisionV0  # operator addition: per-action Claude multimodal
from .templates.random_agent import Random
from .templates.reasoning_agent import ReasoningAgent
from .templates.smolagents import SmolCodingAgent, SmolVisionAgent

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    cls.__name__.lower(): cast(Type[Agent], cls)
    for cls in Agent.__subclasses__()
    if cls.__name__ != "Playback"
}

# add all the recording files as valid agent names
for rec in Recorder.list():
    AVAILABLE_AGENTS[rec] = Playback

# update the agent dictionary to include subclasses of LLM class
AVAILABLE_AGENTS["reasoningagent"] = ReasoningAgent

__all__ = [
    "Swarm",
    "Random",
    "LangGraphFunc",
    "LangGraphTextOnly",
    "LangGraphThinking",
    "LangGraphRandom",
    "LLM",
    "FastLLM",
    "ReasoningLLM",
    "GuidedLLM",
    "ReasoningAgent",
    "SmolCodingAgent",
    "SmolVisionAgent",
    "Agent",
    "Recorder",
    "Playback",
    "AVAILABLE_AGENTS",
    "MultiModalLLM",
]
