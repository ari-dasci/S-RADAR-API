from fastapi import FastAPI
from routers import utils
from routers import federated_data
from routers import static_data
from routers import time_series
from routers import pipelines   
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.include_router(utils.router)
app.include_router(federated_data.router)
app.include_router(static_data.router)
app.include_router(time_series.router)
app.include_router(pipelines.router)

origins = [
    "http://localhost:36799",
    "*"
]

# Add CORS middleware with allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow frontend's domain
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!!"}