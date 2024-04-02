from fastapi import FastAPI
from routers import utils
from routers import static_data
from routers import time_series
app = FastAPI()

app.include_router(utils.router)
app.include_router(static_data.router)
app.include_router(time_series.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!!"}