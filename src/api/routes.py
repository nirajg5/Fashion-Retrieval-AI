from pathlib import Path
import shutil
import uuid

from fastapi import (
    APIRouter,
    File,
    Form,
    UploadFile
)

from .schemas import TextSearchRequest
from src.retrieval.retriever import retriever

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.get("/")
def health():

    return {
        "status": "running",
        "service": "Fashion Retrieval API"
    }


@router.post("/search/text")
def search_text(request: TextSearchRequest):

    return retriever.retrieve(
        request.query
    )


@router.post("/search/image")
async def search_image(
    file: UploadFile = File(...)
):

    suffix = Path(file.filename).suffix

    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = retriever.retrieve_by_image(
        str(temp_path)
    )

    temp_path.unlink(missing_ok=True)

    return {
        "results": results
    }

@router.post("/search/image")
async def search_image(file: UploadFile = File(...)):

    suffix = Path(file.filename).suffix

    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    response = retriever.retrieve_by_image(
        str(temp_path)
    )

    temp_path.unlink(missing_ok=True)

    return response


