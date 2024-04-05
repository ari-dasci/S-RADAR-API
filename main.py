from fastapi import FastAPI
from routers import utils
from routers import static_data
from routers import time_series
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.include_router(utils.router)
app.include_router(static_data.router)
app.include_router(time_series.router)

origins = [
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!!"}