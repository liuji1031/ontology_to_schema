"""Test agents for ontology_to_schema package."""
import logging

from ontology_to_schema import AgentAssignSlot, AgentRelevantSlot

logger = logging.getLogger(__name__)


def test_relevant_slot_agent():
    """Test the agent that determines the relevant slots for a the ontology."""
    agent = AgentRelevantSlot("config/agent_doco.yaml")
    test_query = {
        "contains": True,
        "isContainedBy": True,
        "relation": False,
        "topObjectProperty": False,
        "isContainedByAsHeader": True,
    }
    result = agent.query(test_query)

    assert all([result[k] == test_query[k] for k in test_query.keys()])


def test_slot_assign_agent():
    """Test the agent that assigns slots to classes."""
    agent = AgentAssignSlot("config/agent_doco.yaml")
    cls_name = [
        "Acknowledgements",
        "Header",
        "BibliographicReference",
        "Caption",
        "DiscourseElement",
        "Reference",
        "Abstract",
        "Afterword",
        "Appendix",
        "BackMatter",
    ]
    slot_name = [
        "contains",
        "isContainedBy",
        "isContainedByAsHeader",
    ]
    result = agent.query(cls_name, slot_name)

    for k, v in result.items():
        logger.debug(f"{k}: {v}")
    assert len(result) == len(cls_name) * len(slot_name)
