import logging
import pathlib

import typer
from schema_automator.utils.schemautils import write_schema

from ontology_to_schema.gen_schema import OwlImportEngine, fix_class_definition_sequence

app = typer.Typer()

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
logger = logging.getLogger("ontology_to_schema")


@app.command()
def main(
    inp_path: str = typer.Option(
        ...,
        "--inp_path",
        "-i",
        help="Path to the ontology file",
        dir_okay=False,
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    out_path: str = typer.Option(
        ...,
        "--out_path",
        "-o",
        help="Path to the output schema file",
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
    log_level: str = typer.Option("INFO", help="Log level"),
):
    logger.setLevel(log_level)
    logger.info(f"Ontology file: {inp_path}")
    logger.info(f"Output schema file: {out_path}")
    logger.info(f"Agent configuration file: {agent_cfg}")
    logger.info(f"Log level: {log_level}")

    # create output folder if it does not exist
    out_dir = pathlib.Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    converter = OwlImportEngine(agent_cfg=agent_cfg, log_level=log_level)
    schema = converter.convert(inp_path)
    logger.info(f"Writing schema to {out_path}...")
    write_schema(schema, out_path)
    fix_class_definition_sequence(out_path, overwrite=True)
    logger.info("✅ All done!")

if __name__ == "__main__":
    app()
