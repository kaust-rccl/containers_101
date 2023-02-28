FROM ubuntu:focal
RUN apt-get update && \
    apt-get install -y git wget vim
# Install Conda
WORKDIR /software
ARG CONDA_HOME=/software/miniconda3
ARG TMP_DIR=/tmp
WORKDIR ${TMP_DIR}
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -f -p ${CONDA_HOME} && \
rm /tmp/Miniconda3-latest-Linux-x86_64.sh
ENV PATH=$PATH:${CONDA_HOME}/condabin:${CONDA_HOME}/bin

# Create a conda environment
WORKDIR /software
COPY environment.yml environment.yml
RUN conda env create -f environment.yml

COPY entrypoint.sh entrypoint.sh
RUN chmod +x /software/entrypoint.sh
ENTRYPOINT ["/software/entrypoint.sh"]

