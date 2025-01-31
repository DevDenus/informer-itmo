import time
import asyncio
import logging
from typing import List

from fastapi import FastAPI, HTTPException, Request, Response
from schemas.request import PredictionRequest, PredictionResponse
from sentence_transformers import SentenceTransformer

from src.crawler import ITMONewsCrawler
from src.database import RetrievalDatabase
from src.informer import LLMResponseGenerator
from src.utils.logger import setup_logger
from schemas.request import PredictionResponse

BASE_URL = "https://news.itmo.ru/ru/"

app = FastAPI()
logger = logging.getLogger(__name__)

db = RetrievalDatabase(dimension=1024)
encoder = SentenceTransformer("intfloat/multilingual-e5-large", device='cuda')
crawler = ITMONewsCrawler(BASE_URL, db, encoder, 0.5)
generator = LLMResponseGenerator(db, encoder)

async def run_crawler_periodically(interval_minutes: int = 60):
    while True:
        logger.info("Запускаем краулер для обновления базы...")
        start_time = time.time()

        try:
            crawler.crawl()
            logger.info(f"Краулер успешно обновил данные за {time.time() - start_time:.2f} секунд.")
        except Exception as e:
            logger.error(f"Ошибка при работе краулера: {e}")

        logger.info(f"Ожидаем {interval_minutes} минут до следующего запуска...")
        await asyncio.sleep(interval_minutes * 60)


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()
    asyncio.create_task(run_crawler_periodically(30))

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        #logger.info(f"Processing prediction request with id: {body.id}")
        result = generator.generate_response(body.query, 7)

        response = PredictionResponse(
            id=body.id,
            answer=str(result['answer']),
            reasoning=result['reasoning'],
            sources=result['sources'],
        )
        logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
