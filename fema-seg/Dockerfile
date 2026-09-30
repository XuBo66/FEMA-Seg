FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
WORKDIR /workspace/fema-seg
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install packaging ninja && pip install --no-build-isolation -r requirements-cuda.txt
COPY . .
CMD ["python", "train.py", "--help"]
