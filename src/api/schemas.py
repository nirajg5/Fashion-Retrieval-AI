from pydantic import BaseModel


class TextSearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    image_id: int
    image_path: str
    caption: str

    category: str | None = None
    supercategory: str | None = None

    image_score: float
    crop_score: float
    caption_score: float

    fusion_score: float
    rerank_score: float
    final_score: float