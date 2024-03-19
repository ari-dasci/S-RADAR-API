from fastapi import FastAPI
from routers import utils
app = FastAPI()

app.include_router(utils.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}