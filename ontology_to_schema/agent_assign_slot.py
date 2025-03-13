"""
Agent for query slot assignment to classes.

This module sets up an agent to determine if the queried slots are relevant to
the ontology definition. The step is used to filter out irrelevant slots from
the ontology definition, as some of the slots are high level abstract
attributes.
"""

import json
from typing import Dict, List, Tuple

from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field

from ontology_to_schema.agent import Agent


class CustomOutputSchema(BaseIOSchema):
    """The output schema for the CustomAgent.

    The output will be a list of booleans, with each boolean indicating whether
    the corresponding query from the user is true (1) or false (0).
    """

    response: str = Field(
        ...,
        description=(
            "The response from the agent in a list, with each "
            "field corresponding to a query from the user, and the value "
            "(boolean) indicating whether the query is true (1) or false (0)."
        ),
    )

    def decode(self) -> Dict[Tuple[str, str], bool]:
        """Decode the response from the agent."""
        response_dict = json.loads(self.response)
        out_dict = {}
        for k, v in response_dict.items():
            cn, sn = k[1:-1].split(",")
            out_dict[(cn, sn)] = bool(v)
        return out_dict


class CustomInputSchema(BaseIOSchema):
    """The input schema for the CustomAgent.

    The input will be a string representing a list of tuples, where each tuple
    contains two elements: the first element is the class name, and the second
    element is the attribute name. Each tuple corresponds to a query from the
    user.
    """

    query: str = Field(
        ...,
        description=(
            "A string representing a list tuples, each consisting of a class "
            "name and an attribute name, "
            "Each tuple corresponds to a query from the user."
        ),
    )


class AgentAssignSlot(Agent):
    """Agent for query slot assignment to classes."""

    def __init__(self, config):
        """Initialize the agent.

        Args:
            config (str): The path to the configuration file.
        """
        super().__init__(
            config,
            name="assign_slot",
            input_schema=CustomInputSchema,
            output_schema=CustomOutputSchema,
        )

    def get_query(self, cls_name: List[str], slot_name: List[str]) -> str:
        """Generate a query string for class and attribute names.

        The string consists of a list of tuples, each containing a class name
        and an attribute name.

        Args:
            cls_name (List[str]): list of class names
            slot_name (List[str]): list of attribute names

        Returns:
            str: query for agent
        """
        query = ""
        for cn in cls_name:
            for an in slot_name:
                query += f"({cn}" + "," + f"{an})" + ","
        return query

    def query(
        self, cls_name: List[str], slot_name: List[str]
    ) -> Dict[str, bool]:
        """Send query to the agent to determine slot relevance.

        Args:
            cls_name (List[str]): list of class names
            slot_name (List[str]): list of attribute names
        """
        query = self.get_query(cls_name, slot_name)
        response = self.agent.run(CustomInputSchema(query=query))
        # response: CustomOutputSchema
        return response.decode()
