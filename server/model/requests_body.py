from pydantic import BaseModel


class NerRequsts(BaseModel):
    task: str
    description: str

