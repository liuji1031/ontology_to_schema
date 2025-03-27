"""
Convert an ontology to a schema.

This package contains modules for converting an ontology to a schema, including
the agent that determines the relevant slots for a the ontology, the agent that
assigns slots to classes, and the main engine that imports the ontology and
generates the schema.
"""

from ontology_to_schema.gen_schema import (
    OwlImportEngine,
    generate_schema,
)

__all__ = [
    "OwlImportEngine",
    "generate_schema",
]
