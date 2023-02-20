FROM ubuntu:focal
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y wget

# install miniconda in user's $HOME directory
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# install miniconda in user's miniconda3 directory
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
# define environment path
ENV PATH=$PATH:/miniconda/pcondabin:/miniconda/bin
COPY packages packages
# update conda
RUN conda update --name base --channel defaults --yes conda

# install mamba in the base environment
RUN conda install --name base --channel conda-forge mamba --yes

RUN mamba env update -f packages/environment.yml

RUN mamba env create -f packages/environment_one.yml
SHELL ["conda","run","-n","one","/bin/bash","-c"]
RUN python -m ipykernel install --name kernel_one --display-name "Display Name One"
RUN pip install -U -r packages/requirements_one.txt


RUN mamba env create -f packages/environment_two.yml
SHELL ["conda","run","-n","two","/bin/bash","-c"]
RUN python -m ipykernel install --name kernel_two --display-name "Display Name Two"
RUN pip install -U -r packages/requirements_two.txt
RUN rm -rf packages
RUN rm Miniconda3-latest-Linux-x86_64.sh

# RUN mkdir -pv /etc/ipython/
#COPY ipython/ipython_config.py /etc/ipython/ipython_config.py

ENV NB_USER aslan
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

WORKDIR /home/${NB_USER}
USER $NB_USER

SHELL ["/bin/bash","-c"]
RUN conda init
RUN echo 'conda activate one' >> ~/.bashrc

EXPOSE 8888
# COPY README.md README.md

ENTRYPOINT ["jupyter", "notebook", "--no-browser","--ip=0.0.0.0","--NotebookApp.token=''","--NotebookApp.password=''"]