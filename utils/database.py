import json
from typing import List, Dict, Any

import numpy as np
import redis
import faiss
import time

class RetrievalDatabase:
    _instance = None  # Singleton Instance

    def __new__(cls, dimension: int, redis_host: str = "localhost", redis_port: int = 6379, faiss_index_file : str = "faiss_index.bin"):
        """
        Singleton: гарантирует, что класс создаст только один инстанс.
        """
        if cls._instance is None:
            cls._instance = super(RetrievalDatabase, cls).__new__(cls)
            cls._instance._init_instance(dimension, redis_host, redis_port, faiss_index_file)
        return cls._instance

    def _init_instance(self, dimension: int, redis_host: str, redis_port: int, faiss_index_file : str) -> None:
        self.redis_db = redis.Redis(host=redis_host, port=redis_port, db=1, decode_responses=True)
        self.dimension = dimension
        self.faiss_index_file = faiss_index_file

        try:
            self.index = faiss.read_index(self.faiss_index_file)
            print("FAISS индекс загружен из файла!")
        except RuntimeError:
            self.index = faiss.IndexFlatL2(self.dimension)
            print("FAISS индекс не найден, создан новый!")

    def insert_news(self, news_list: List[Dict[str, Any]]) -> None:
        """
        Добавляет новости в Redis + их векторное представление в FAISS.
        """
        for i, news in enumerate(news_list, start=self.index.ntotal):
            news_id = f"news:{i}"
            cleared_news = {
                'url': news['url'],
                'title': news['title'],
                'date_time': news['date_time'],
                'content': news['content']
            }
            self.redis_db.set(news_id, json.dumps(cleared_news))

            embedding = news['embedding'].reshape(1, -1)
            self.index.add(embedding)

        print(f"Добавлено {len(news_list)} новостей")

    def save_index(self) -> None:
        """
        Сохраняет FAISS-индекс на диск.
        """
        faiss.write_index(self.index, self.faiss_index_file)
        print("FAISS индекс сохранён.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, str]]:
        distances, indices = self.index.search(query_embedding, top_k)

        news_ids = [f"news:{idx}" for idx in indices[0]]
        news_data = self.redis_db.mget(news_ids)
        return [json.loads(news) for news in news_data if news]



if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    db = RetrievalDatabase(dimension=384)

    query = "какие правила приёма абитуриентов в 2025 году?"
    querry_embed = encoder.encode(query).reshape(1, -1)

    time_start = time.time()

    results = db.search(querry_embed, top_k=3)

    print("\nНайденные новости:")
    for res in results:
        print(f"{res['title']} ({res['url']})")
        print(f"{res['content']}\n")
    print(f'Заняло {time.time() - time_start}')
