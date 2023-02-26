# change this into latest
FROM ubuntu:focal

RUN apt-get update
# colapse these lines into one 
RUN apt-get install -y git
RUN apt-get install -y wget

# install miniconda in user's $HOME directory
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# install miniconda in user's miniconda3 directory
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /software/miniconda3

# define environment path
ENV PATH=$PATH:/miniconda3/pcondabin:/software/miniconda3/bin

# copy packages folder to container
COPY packages packages
# volume mount like -v 
VOLUME 
# update conda
RUN conda update --name base --channel defaults --yes conda

# install mamba in the base environment
#RUN conda install --name base --channel conda-forge mamba --yes

########## install packages in base environment ##########

# # install packages from environment.yml that contains jupyter lab
# RUN mamba env update -f packages/environment.yml

########## install packages in new environment ##########

# # install packages from environment_one.yml
# RUN mamba env create -f packages/environment_one.yml
# SHELL ["conda","run","-n","one","/bin/bash","-c"]
# RUN python -m ipykernel install --name kernel_one --display-name "Display Name One"
# # install packages from requirements_one.txt
# RUN pip install -U -r packages/requirements_one.txt

########## install packages in new environment ##########

# RUN mamba env create -f packages/environment_two.yml
# SHELL ["conda","run","-n","two","/bin/bash","-c"]
# RUN python -m ipykernel install --name kernel_two --display-name "Display Name Two"
# RUN pip install -U -r packages/requirements_two.txt

# remove packages folder and miniconda installation file
RUN rm -rf packages
RUN rm Miniconda3-latest-Linux-x86_64.sh

# create user and UID
ENV NB_USER asad
ENV NB_UID 1000

# add user to container
RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER

# set user aslan as the owner of the container
WORKDIR /home/${NB_USER}
USER $NB_USER

# activate conda base environment
SHELL ["/bin/bash","-c"]
RUN conda init
# activate conda environment one on startup
# RUN echo 'conda activate one' >> ~/.bashrc

# Expose port 8888 for jupyter lab
EXPOSE 8888

#### optional: add jupyter lab to startup ####
# this is the entrypoint for the container and will be executed on startup
# ENTRYPOINT ["jupyter", "lab", "--no-browser","--ip=0.0.0.0","--NotebookApp.token=''","--NotebookApp.password=''"]
