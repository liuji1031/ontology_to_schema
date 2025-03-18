#!/bin/bash

version=$(<VERSION)

# Update with your data
api_key=${GROQ_API_KEY}
input_filename=doco.ttl
host_input_dir=./ttl/
host_output_dir=./schema/
host_agent_cfg=./config/agent_doco.yaml

container_input_dir="/inp_dir"
container_output_dir="/out_dir"
container_agent_cfg="/agent_config.yaml"

docker run -e API_KEY=${api_key} \
           -v $host_input_dir:/${container_input_dir} \
           -v $host_output_dir:/${container_output_dir} \
           -v $host_agent_cfg:/${container_agent_cfg} \
            --user $(id -u):$(id -g) \
            ontology-to-schema:${version} \
            --inp_dir ${container_input_dir} \
            --file_name ${input_filename} \
            --out_dir ${container_output_dir} \
            --agent_cfg ${container_agent_cfg}
