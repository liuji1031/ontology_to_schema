"""
Agent for query wheather the slot is relevant to the ontology.

This module sets up an agent to determine if the queried slots are relevant to
the ontology definition. The step is used to filter out irrelevant slots from
the ontology definition, as some of the slots are high level abstract
attributes.
"""

import json
from typing import Dict

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field

from ontology_to_schema.agent import Agent


class CustomOutputSchema(BaseIOSchema):
    """The output schema for the CustomAgent.

    The output will be in json format, with each field correspond to a query
    from the user, and the value (boolean) indicating whether the query is
    true or false.
    """

    response: str = Field(
        ...,
        description=(
            "The response from the agent in json format, with each "
            "field corresponding to a query from the user, and the value "
            "(0 or 1) indicating whether the query is true or false."
        ),
    )

    def decode(self) -> Dict[str, bool]:
        """Decode the response from the agent."""
        response_dict = json.loads(self.response)
        for k, v in response_dict.items():
            response_dict[k] = bool(v)
        return response_dict


class CustomInputSchema(BaseIOSchema):
    """The input schema for the CustomAgent.

    The input will be a string representing a list of attribute names
    (delimited by ","). Each attribute name corresponds to a query from the
    user.
    """

    query: str = Field(
        ...,
        description=(
            "A string representing a list of attribute names "
            "(delimited by ','). Each attribute name corresponds to a query "
            "from the user."
        ),
    )


class AgentRelevantSlot(Agent):
    """Agent for query slot relevance to ontology."""

    def __init__(self, config: str):
        """Initialize the agent.

        Args:
            config (str): The path to the configuration file.
        """
        super().__init__(
            config,
            name="relevant_slot",
            input_schema=CustomInputSchema,
            output_schema=CustomOutputSchema,
        )

    def query(self, query: str | dict) -> Dict[str, bool]:
        """Send query to the agent to determine slot relevance.

        Args:
            query (str | dict): either a string representing a list of
            attribute names (delimited by ","), or a dictionary with keys
            containing the attribute names.
        """
        if isinstance(query, dict):
            query = ",".join(query.keys())
        response = self.agent.run(CustomInputSchema(query=query))
        # response: CustomOutputSchema
        return response.decode()
