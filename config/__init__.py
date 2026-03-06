"""Config — Pydantic schemas, JSON loading, defaults."""
from .schema import BanditConfig, AgentConfig, ExperimentConfig, FullConfig
from .loader import load_config
