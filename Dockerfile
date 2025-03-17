FROM polusai/bfio:2.4.7

# ENV API_KEY=""
ENV PYSTOW_HOME=/opt/executables/
RUN apt-get update &&\
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# environment variables defined in polusai/bfio
ENV EXEC_DIR="/opt/executables"

# Work directory defined in the base container
WORKDIR ${EXEC_DIR}

# install robot tool
RUN cd /usr/local/bin/
RUN curl https://github.com/ontodev/robot/releases/download/v1.9.7/robot.jar -L -o /usr/local/bin/robot.jar
RUN curl https://raw.githubusercontent.com/ontodev/robot/master/bin/robot > robot
RUN chmod +x robot

# TODO: Change the tool_dir to the tool directory
ENV TOOL_DIR="ontology-to-schema"

# Copy the repository into the container
RUN mkdir ${TOOL_DIR}
COPY . ${EXEC_DIR}/${TOOL_DIR}

# install python 3.10
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3-pip python3-distutils python3-setuptools
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1  # python 3.10 uses pip3
RUN python -m pip install --upgrade pip

# Install the tool
RUN pip3 install "${EXEC_DIR}/${TOOL_DIR}" --no-cache-dir

# Set the entrypoint
# TODO: Change the entrypoint to the tool entrypoint
ENTRYPOINT ["ontology-to-schema"]
CMD ["--help"]
