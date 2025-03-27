"""Utility functions for the ontology_to_schema module."""
import logging
import os
import pathlib

import instructor


def setup_client(provider, api_key_env_varname):
    """Set up the client for the specified provider.

    Args:
        provider (str): The provider to use.
        api_key_env_varname (str): The environment variable name for the API
        key.
    """
    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv(api_key_env_varname)
        client = instructor.from_openai(OpenAI(api_key=api_key))
    elif provider == "groq":
        from groq import Groq

        api_key = os.getenv(api_key_env_varname)
        client = instructor.from_groq(
            Groq(api_key=api_key), mode=instructor.Mode.JSON
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client


def ttl_to_ofn(ttl: str | list[str]) -> None:
    """Convert Turtle file(s) to an OFN file.

    Args:
        ttl (str | list[str]): The Turtle file or list of Turtle files to
        convert.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if isinstance(ttl, str):
        file_list = [ttl]
    else:
        file_list = ttl

    out_dir = None
    err_files = []

    for file in file_list:
        if out_dir is None:
            out_dir = pathlib.Path(file).parent / ".ofn"

        filepath = pathlib.Path(file)
        ofn_filepath = out_dir / pathlib.Path(file).name.replace(
            ".ttl", ".ofn"
        )

        if ofn_filepath.exists():
            logger.info(f"{str(ofn_filepath)} already exists. Skipping...")
            continue

        logger.info(f"Converting {filepath} to .ofn using ROBOT")
        if os.system(
            f"robot convert -i {str(filepath)} "
            + f"-output {str(ofn_filepath)}"
        ):
            logger.error(f"Error converting {str(filepath)} to .ofn")
            err_files.append(filepath)

    logger.info(f"Conversion complete. OFN files are in {str(out_dir)}")

    if err_files:
        logger.error("The following files failed to convert:")
        for f in err_files:
            logger.error(str(f))
