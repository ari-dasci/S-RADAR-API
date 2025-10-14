# RADAR-API

This is the backend API for [**RADAR-Frontend**](https://github.com/ari-dasci/RADAR-frontend), a system for interactive exploration of anomaly detection algorithms in different types of data.

The API exposes endpoints to run pipelines, manage data, and execute algorithms provided by the [**RADAR library**](https://github.com/ari-dasci/RADAR).

---

## 🧩 Features

- Modular pipeline execution
- Integration with the RADAR anomaly detection library
- Support for time series and tabular data
- RESTful API built with **FastAPI**

---

## ⚙️ Installation

### Prerequisites

- **Python: 3.10** (tested with 3.10.18)
- **Pytorch: 2.7.1** (tested with with CUDA 11.8 support)
- `conda` (optional but recommended)

### Clone the API (backend) repository

```bash
git clone https://github.com/ari-dasci/RADAR-API.git
```

### Clone the RADAR library

```bash
git clone https://github.com/ari-dasci/RADAR.git
```

### Install RADAR library + API (conda environment)


```bash
conda create --prefix ./envs/radar-env python=3.10.18

conda activate ./envs/radar-env 

conda env update --prefix /path/to/env --file environment.yml --prune   # Replace /path/to/env with the location of your Conda environment.

#Make sure Pytorch is installed now, if you have CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install pytorch-lightning
```

Now install the API with

```bash
pip install "fastapi[standard]"
pip install "uvicorn[standard]"
```

Export path to RADAR library

```bash
export PYTHONPATH=«route_to_RADAR»
```

### Run the FastAPI server

```bash
uvicorn main:app --reload --host 0.0.0.0 
```

API will be available at: [http://localhost:8000](http://localhost:8000)

Swagger documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

---
## 📘 Full Documentation & Frontend Access

Please visit the frontend repository for:

- Full documentation

- Usage examples

- UI for interacting with the API

👉 [**RADAR-Frontend Repository**](https://github.com/ari-dasci/RADAR-frontend)
