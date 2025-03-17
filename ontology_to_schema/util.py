"""Utility functions for the ontology_to_schema module."""
import os

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
