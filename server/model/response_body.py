from pydantic import BaseModel


class Response(BaseModel):
    status_code: int
    payload: dict | None = None
    message: str
