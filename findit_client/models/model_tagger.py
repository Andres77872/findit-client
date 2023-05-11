from pydantic import BaseModel

from findit_client.models.model_common import ResultCommonStatusMeta


class TagModel(BaseModel):
    tag: str
    score: float


class TagsModel(BaseModel):
    rating: list[TagModel]
    general: list[TagModel]
    character: list[TagModel]


class TaggerResultsModel(BaseModel):
    count: int
    data: TagsModel


class TaggerResponseModel(BaseModel):
    api_version: str
    mode: str
    load_image_time: float
    embedding_time: float
    post_process_time: float
    results: TaggerResultsModel
    status: ResultCommonStatusMeta
