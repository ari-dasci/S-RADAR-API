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

## 🐳 Docker — Full System (Frontend + Backend)

The easiest way to get the complete RADAR system running is via Docker. A single image bundles the **Angular frontend**, the **FastAPI backend**, and the **RADAR library**.

### Option 1: Pull from Docker Hub

```bash
docker pull beatrizbellogarcia/radar:latest
docker run -d -p 80:80 --name radar beatrizbellogarcia/radar:latest
```

Then open [http://localhost](http://localhost) in your browser.

### Option 2: Build from source

```bash
git clone https://github.com/ari-dasci/RADAR-API.git
cd RADAR-API
docker build -f docker/Dockerfile -t radar-unified .
docker run -d -p 80:80 --name radar radar-unified
```

This will automatically:
1. Clone and compile the [RADAR-Frontend](https://github.com/ari-dasci/RADAR-frontend) Angular app.
2. Clone the [RADAR library](https://github.com/ari-dasci/RADAR) (if not already present locally).
3. Install all Python dependencies and bundle everything into a single lightweight image.

### Useful commands

```bash
# Stop the container
docker stop radar

# Start it again
docker start radar

# View logs
docker logs -f radar

# Remove the container
docker rm -f radar
```

---

## 📘 Full Documentation & Frontend Access

Please visit the frontend repository for:

- Full documentation

- Usage examples

- UI for interacting with the API

👉 [**RADAR-Frontend Repository**](https://github.com/ari-dasci/RADAR-frontend)
