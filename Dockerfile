# or FROM ubuntu:20.04
FROM ubuntu:22.04 

# clean-up: later will be deleted
RUN \
  apt-get update -y && \
  apt-get install software-properties-common -y && \
  apt-get update -y && \
  apt-get install -y openjdk-8-jdk \
                git \
                build-essential \
                subversion \
                perl \
                curl \
                unzip \
                cpanminus \
                make \
                && \
  rm -rf /var/lib/apt/lists/*

RUN apt-get install vim 

# Java version
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

# Timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python
#RUN \
#    apt install software-properties-common -y && \
#    add-apt-repository ppa:deadsnakes/ppa -y && \
#    apt install python3.9 -y 
#    #apt install python3.12.3 -y
#    # or apt install python3.9 -y 

COPY . /usr/latent_mutant 
WORKDIR /usr/latent_mutant 

RUN apt-get install python3-pip

# prepare python-venv 
RUN \ 
  python3 -m pip install --user virtualenv && \
  python3 -m venv env 

# after activating 
RUN \
  . ./env/bin/activate  && \ 
  python3 -m pip3 install setuptools && \
  python3 -m pip3 install -r new_requirements.txt && \ 
  deactivate 

