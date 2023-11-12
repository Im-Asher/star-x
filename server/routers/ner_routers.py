from fastapi import APIRouter
from model.requests_body import NerRequsts
from model.enums.status_enum import ResponseCode

ner_router = APIRouter(
    prefix='/ner',
    tags=['ner']
)


@ner_router.post('/extract')
async def extract(data: NerRequsts):
    resp = {'status_code':ResponseCode.Success.value,'message':'is NER'}

@ner_router.get('/health')
async def extract():
    resp = {'status_code':ResponseCode.Success.value,'message':'is NER'}
    return resp
