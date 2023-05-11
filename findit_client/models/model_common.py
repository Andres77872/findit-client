from pydantic import BaseModel


class ResultCommonStatusMeta(BaseModel):
    code: str
    msg: list[str, ...]
