import json
from typing import List, Dict

import numpy as np
import redis
import faiss
import time

class RetrievalDatabase:
    _instance = None  # Singleton Instance

    def __new__(cls, dimension: int, encoder, redis_host: str = "localhost", redis_port: int = 6379, faiss_index_file : str = "faiss_index.bin"):
        """
        Singleton: гарантирует, что класс создаст только один инстанс.
        """
        if cls._instance is None:
            cls._instance = super(RetrievalDatabase, cls).__new__(cls)
            cls._instance._init_instance(dimension, encoder, redis_host, redis_port, faiss_index_file)
        return cls._instance

    def _init_instance(self, dimension: int, encoder, redis_host: str, redis_port: int, faiss_index_file : str) -> None:
        self.redis_db = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
        self.encoder = encoder
        self.dimension = dimension
        self.faiss_index_file = faiss_index_file

        try:
            self.index = faiss.read_index(self.faiss_index_file)
            print("FAISS индекс загружен из файла!")
        except RuntimeError:
            self.index = faiss.IndexFlatL2(self.dimension)
            print("FAISS индекс не найден, создан новый!")

    def insert_news(self, news_list: List[Dict[str, str]]) -> None:
        """
        Добавляет новости в Redis + их векторное представление в FAISS.
        """
        embeddings = []
        for i, news in enumerate(news_list):
            news_id = f"news:{i}"
            self.redis_db.set(news_id, json.dumps(news))

            embedding = self.encoder.encode(news["content"]).reshape(1, -1)
            embeddings.append(embedding)

        if embeddings:
            embeddings = np.vstack(embeddings)
            self.index.add(embeddings)
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

    db = RetrievalDatabase(dimension=384, encoder=encoder)

    sample_news = [
        {
            "url": "https://news.itmo.ru/ru/education/official/news/13092/",
            "title": "ИТМО вошёл в топ-10 вузов",
            "date": "22 Января 2025",
            "content": "ИТМО стал первым среди петербургских вузов по качеству подготовки IT-специалистов."
        },
        {
            "url": "https://news.itmo.ru/ru/university_live/achievements/news/14133/",
            "title": "Цифровая кафедра ИТМО",
            "date": "10 Января 2025",
            "content": "ИТМО вошёл в число лидеров цифровой экономики, создав цифровую кафедру."
        },
        {
            "url": "https://news.itmo.ru/ru/science/ai/news/15000/",
            "title": "Нейросети в ИТМО",
            "date": "5 Февраля 2025",
            "content": "Исследователи ИТМО разработали новую архитектуру нейросетей для анализа данных."
        }
    ]

    #db.insert_news(sample_news)

    #db.save_index()


    query = "где лучшие IT-специалисты?"
    querry_embed = encoder.encode(query).reshape(1, -1)

    time_start = time.time()

    results = db.search(querry_embed, top_k=2)

    print("\n🔍 Найденные новости:")
    for res in results:
        print(f"{res['title']} ({res['url']})")
        print(f"{res['content']}\n")
    print(f'Заняло {time.time() - time_start}')
