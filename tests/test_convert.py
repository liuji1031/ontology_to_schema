"""Test the entire converting pipeline."""
from schema_automator.utils.schemautils import write_schema

from ontology_to_schema import OwlImportEngine, fix_class_definition_sequence


def test_import_engine():
    """Test the entire converting pipeline."""
    out_path = "ttl/doco.yaml"
    oie = OwlImportEngine(agent_cfg="config/agent_doco.yaml")
    schema = oie.convert("ttl/doco.ttl")
    write_schema(schema, out_path)
    fix_class_definition_sequence(out_path, overwrite=True)
