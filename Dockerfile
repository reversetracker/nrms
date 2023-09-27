FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY csv /opt/nrms/csv

COPY requirements.txt /opt/nrms/requirements.txt
RUN pip install -r /opt/nrms/requirements.txt

COPY . /opt/nrms
WORKDIR /opt/nrms

CMD ["python", "train.py"]
