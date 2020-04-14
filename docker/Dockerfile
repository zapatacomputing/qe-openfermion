FROM ubuntu
WORKDIR /app
USER root
RUN apt-get update --fix-missing

# Install python, pip, and other utilities
RUN yes | apt-get install python3
RUN yes | apt-get install python3-pip
RUN yes | apt-get install vim
RUN apt-get -y install htop
RUN yes | apt-get install git
RUN yes | apt-get install curl

RUN python3 -m pip install scipy==1.2.2 \
                           numpy==1.15.4 \
                           openfermion==0.10.0 \
                           pyyaml==5.1 \
                           python-rapidjson==0.9.1

WORKDIR /app

ENTRYPOINT bash
