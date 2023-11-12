# define model server
import uvicorn

from fastapi import FastAPI
from model.enums.status_enum import ResponseCode
from routers.ner_routers import ner_router

app = FastAPI()


@app.get('/health')
async def health() -> dict:
    resp = {'status_code':ResponseCode.Success.value,'message':'Service is health'}
    return resp

app.include_router(ner_router,prefix='/api/v1')

if __name__ == '__main__':
    uvicorn.run('main:app')
