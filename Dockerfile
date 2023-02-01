FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt update
RUN apt -y install python3 python3-pip
RUN yes | pip3 install -r requirements.txt

COPY . .

RUN git clone https://github.com/jnordberg/tortoise-tts.git
RUN cd tortoise-tts && python3 setup.py install


CMD [ "python3", "run.py"]
