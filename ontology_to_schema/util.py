import os
import instructor

def setup_client(provider):
    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(OpenAI(api_key=api_key))
    elif provider == "groq":
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        client = instructor.from_groq(
            Groq(api_key=api_key), mode=instructor.Mode.JSON
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client