"""
This module sets up an agent to determine if the queried slots are relevant to
the ontology definition. The step is used to filter out irrelevant slots from
the ontology definition, as some of the slots are high level abstract attributes.
"""

import os

import yaml
from atomic_agents.agents.base_agent import (
    BaseAgent,
    BaseAgentConfig,
)
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.components.system_prompt_generator import (
    SystemPromptGenerator,
)

from ontology_to_schema.util import setup_client


class Agent:
    def __init__(self, config,name,input_schema=None, output_schema=None):
        if os.path.isfile(config) and config.endswith(".yaml"):
            with open(config, "r") as file:
                config = yaml.safe_load(file)
        else:
            raise ValueError("Invalid configuration file")
        config = config.get(name, None)
        if config is None:
            raise ValueError("Configuration for AgentRelevantSlot is missing")

        assert "background" in config, (
            "background is missing in the configuration"
        )
        assert "steps" in config, "steps is missing in the configuration"
        assert "output_instructions" in config, (
            "output_instructions is missing in the configuration"
        )
        system_prompt_generator = SystemPromptGenerator(
            background=[config["background"]],
            steps=[config["steps"]],
            output_instructions=[config["output_instructions"]],
        )

        self.agent = BaseAgent(
            config=BaseAgentConfig(
                client=setup_client(config.get("provider", "groq")),
                model=config.get("model", "llama-3.3-70b-versatile"),
                memory=AgentMemory(),
                model_api_parameters=config.get("model_api_parameters", {}),
                system_prompt_generator=system_prompt_generator,
                input_schema=input_schema,
                output_schema=output_schema,
                temperature=None, 
            )
        )
