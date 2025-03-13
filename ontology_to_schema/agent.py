"""
Agent for query information for ontology to schema conversion.

This module sets up an agent to determine if the queried slots are relevant to
the ontology definition. The step is used to filter out irrelevant slots from
the ontology definition, as some of the slots are high level abstract
attributes.
"""

import os

import yaml
from atomic_agents.agents.base_agent import (
    BaseAgent,
    BaseAgentConfig,
    BaseIOSchema,
)
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.components.system_prompt_generator import (
    SystemPromptGenerator,
)

from ontology_to_schema.util import setup_client


class Agent:
    """Agent for query information for ontology to schema conversion."""

    def __init__(
        self,
        config: str,
        name: str,
        input_schema: BaseIOSchema | None = None,
        output_schema: BaseIOSchema | None = None,
    ):
        """Initialize the agent.

        Args:
            config (str): The path to the configuration file.
            name (str): The name of the agent to load.
            input_schema: The input schema for the agent.
            output_schema: The output schema for the agent.
        """
        if os.path.isfile(config) and config.endswith(".yaml"):
            with open(config) as file:
                config_dict = yaml.safe_load(file)
        else:
            raise ValueError("Invalid configuration file")
        cfg = config_dict.get(name, None)
        if cfg is None:
            raise ValueError("Configuration for AgentRelevantSlot is missing")

        assert (
            "background" in cfg
        ), "background is missing in the configuration"
        assert "steps" in cfg, "steps is missing in the configuration"
        assert (
            "output_instructions" in cfg
        ), "output_instructions is missing in the configuration"
        system_prompt_generator = SystemPromptGenerator(
            background=[cfg["background"]],
            steps=[cfg["steps"]],
            output_instructions=[cfg["output_instructions"]],
        )

        self.agent = BaseAgent(
            config=BaseAgentConfig(
                client=setup_client(cfg.get("provider", "groq")),
                model=cfg.get("model", "llama-3.3-70b-versatile"),
                memory=AgentMemory(),
                model_api_parameters=cfg.get("model_api_parameters", {}),
                system_prompt_generator=system_prompt_generator,
                input_schema=input_schema,
                output_schema=output_schema,
                temperature=None,
            )
        )
