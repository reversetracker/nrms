FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

COPY requirements.txt /opt/nrms/requirements.txt
RUN pip install -r /opt/nrms/requirements.txt

WORKDIR /opt/nrms
COPY . /opt/nrms

CMD ["python", "train.py"]
