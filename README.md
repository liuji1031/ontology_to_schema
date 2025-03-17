# Ontology to Schema

The purpose of this tool is to extract schemas from ontology definitions such as Document Component Ontology ([DoCO](http://www.sparontologies.net/ontologies/doco)). DoCO provides a vocabulary to describe different components of a document such as a scientific research paper. By establishing a schema, structured information can be potentially extracted using it from collections of literature.

Specifically, this tool is designed to extract schema from ontology defined in Turtle format and generate [LinkML](https://linkml.io/linkml/) schemas. The LinkML schemas (`yaml` format) can be used to generate [Pydantic](https://linkml.io/linkml/generators/pydantic.html) models, [SQL](https://linkml.io/linkml/generators/sqlalchemy.html) models, etc.

## Usage
The package requires [ROBOT](https://robot.obolibrary.org/) package, which converts Turtle format into functional OWL format (`.ofn`). Functional OWL format makes it easier to track down nested sub-class definitions.

Example usage of the package on DoCO definition (the Turtle file can be downloaded from [here](https://sparontologies.github.io/doco/current/doco.ttl)):

```
ontology-to-schema -i doco.ttl -a config/agent_doco.yaml -o schema/doco.yaml
```

| Argument | Description | Required |
|----------|-------------| -------- |
| `--inp_dir` `-i`  | Directory of the input ontology file in Turtle format. | Yes |
| `--file_name` `-f` | File name of the Turtle file. | Yes |
| `--out_dir` `-o`   | Output directory of the output schema file in YAML format. | Yes |
| `--agent_cfg` `-a`   | Configuration file for agents in YAML format. | Yes |
|`--log_level` `-l`  | Log level for logging (ERROR, WARNING, INFO, etc.) | No|

The package uses 2 Atomic AI agent for dynamically determine the following:
1. For all the slots (i.e., attributes in LinkML lingo), determine the relevance of each slot for the ontology in question. The purpose is to filter out high-level abstract slots.
2. For each class, determine if the slot is applicable to the class. This is useful when the slot domain is not specified, i.e., it can be associated with "Any" class.

The settings for these 2 agents are specified in the supplied agent setting file, e.g., `config/agent_doco.yaml` in the above example command. The file contains the following fields:

```
relevant_slot: # step 1 agent
  provider: groq
  model: llama-3.3-70b-versatile
  model_api_parameters:
    max_tokens: 6000
    temperature: 0.0
  background: >-
    <background stuff>
  steps: >-
    <steps stuff>
  output_instructions: >-
    <output_instructions stuff>

assign_slot: # step 2 agent
  provider: groq
  model: llama-3.3-70b-versatile
  model_api_parameters:
    max_tokens: 6000
    temperature: 0.0
  background: >-
    <background stuff>
  steps: >-
    <steps stuff>
  output_instructions: >-
    <output_instructions stuff>
```

The detailed content for agent settings specifically for DoCO can be found in `config/agent_doco.yaml`. To use these
