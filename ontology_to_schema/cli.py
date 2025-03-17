"""Command line interface for the ontology_to_schema package."""
import logging
import pathlib

import typer
from schema_automator.utils.schemautils import write_schema

from ontology_to_schema.gen_schema import (
    OwlImportEngine,
    fix_class_definition_sequence,
)

app = typer.Typer()

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("ontology_to_schema")


@app.command()
def main(
    inp_dir: str = typer.Option(
        ...,
        "--inp_dir",
        "-i",
        help="Directory containing the ontology file",
        file_okay=False,
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    file_name: str = typer.Option(
        ...,
        "--file_name",
        "-f",
        help="Name of the ontology file",
    ),
    out_dir: str = typer.Option(
        ...,
        "--out_dir",
        "-o",
        help="Path to save the output schema file",
        file_okay=False,
        writable=True,
        resolve_path=True,
    ),
    agent_cfg: str = typer.Option(
        ...,
        "--agent_cfg",
        "-a",
        help="Agent configuration file",
        dir_okay=False,
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    log_level: str = typer.Option(
        "INFO", "--log_level", "-l", help="Log level"
    ),
):
    """Command line interface for the ontology_to_schema package."""
    logger.setLevel(log_level)
    inp_path = pathlib.Path(inp_dir) / file_name
    assert inp_path.exists(), f"File {inp_path} does not exist"
    logger.info(f"Ontology file: {inp_path}")
    logger.info(f"Output schema file: {out_dir}")
    logger.info(f"Agent configuration file: {agent_cfg}")
    logger.info(f"Log level: {log_level}")

    # create output folder if it does not exist
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    converter = OwlImportEngine(agent_cfg=agent_cfg, log_level=log_level)
    schema = converter.convert(str(inp_path))
    logger.info(f"Writing schema to {out_dir}...")
    stem = file_name.split(".")[0]
    out_path = pathlib.Path(out_dir) / f"{stem}.yaml"
    write_schema(schema, str(out_path))
    fix_class_definition_sequence(str(out_path), overwrite=True)
    logger.info("✅ All done!")


if __name__ == "__main__":
    app()
