from findit_client.models.model_common import ResultCommonStatusMeta
from pydantic import BaseModel


class ImageSearchResultSingleModel(BaseModel):
    id: int
    source: str
    preview: str
    img: str
    score: float
    pool: str
    query: str
    size: list
    color: str


class ImageSearchResultModel(BaseModel):
    content: list[ImageSearchResultSingleModel]
    vector: list[float]


class ImageSearchResultListModel(BaseModel):
    count: int
    data: list[ImageSearchResultModel]


class ImageSearchResultQdrantConfigMeta(BaseModel):
    limit: int
    pools: list[str]
    vector: list[float]


class ImageSearchResultQdrantMeta(BaseModel):
    time: float
    qdrant_version: str
    status: ResultCommonStatusMeta
    config: ImageSearchResultQdrantConfigMeta


class ImageSearchResultRaw(BaseModel):
    search_version: str
    latency_search: float
    time_groping: float
    time: float
    qdrant_meta: ImageSearchResultQdrantMeta
    status: ResultCommonStatusMeta


class ImageSearchResponseModel(BaseModel):
    api_version: str
    scroll_token: str
    mode: str
    load_image_time: float
    embedding_time: float
    latency_search: float
    post_process_time: float
    results: ImageSearchResultListModel
    search_meta: ImageSearchResultRaw
    status: ResultCommonStatusMeta
