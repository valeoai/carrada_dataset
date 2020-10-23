FROM ubuntu:18.04

RUN apt-get update && apt-get install -y wget bzip2 python3-pip
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes
RUN conda install python=3.6

RUN conda install -c menpo opencv
RUN pip install filterpy==1.4.5 matplotlib==3.1.2 numba==0.46.0 numpy==1.17.4 pandas==0.25.3 Pillow==6.2.1 pytest==5.3.1 scikit-image==0.16.2 scikit-learn==0.22 scipy==1.3.3 torchvision==0.4.2 xmltodict==0.12.0 llvmlite==0.32.1 jupyter

COPY ./ ./carrada_dataset
RUN pip install -e ./carrada_dataset

WORKDIR ./carrada_dataset

# Download the CARRADA Dataset
RUN mkdir /datasets
RUN wget -P /datasets http://download.tsi.telecom-paristech.fr/Carrada/Carrada.tar.gz
RUN tar -xvzf /datasets/Carrada.tar.gz -C /datasets

# Define commands for Jupyter Notebook
CMD ["jupyter", "notebook", "--port=8889", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
