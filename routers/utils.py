from fastapi import APIRouter

router = APIRouter()

@router.get("/utils/blocks", tags=["utils"])
async def get_all_blocks():
    return [{"username": "Rick"}, {"username": "Morty"}]
