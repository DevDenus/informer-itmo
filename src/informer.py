import re
import time
import json

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

from src.database import RetrievalDatabase

class LLMResponseGenerator:
    def __init__(self, db : RetrievalDatabase, encoder : SentenceTransformer, llm_model: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"):
        """
        Инициализация энкодера для поиска и LLM для генерации ответа.
        """
        self.db = db
        self.encoder = encoder
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model, quantization_config=bnb_config, device_map='cuda'
        )
        self.model = torch.compile(self.model)

    def format_llm_response(self, response_text: str) -> dict:
        """
        Форматирует ответ, полученный от LLM, в корректный JSON.
        :param response_text: Ответ от LLM в виде строки.
        :return: Словарь с полями 'answer', 'reasoning' и 'sources'.
        """
        try:
            response_data = json.loads(response_text)
        except json.JSONDecodeError:
            response_data = {
                "answer": -1,
                "reasoning": "",
                "sources": []
            }
            return response_data

        answer = response_data.get("answer", "")

        if isinstance(answer, str):
            match = re.match(r"\{(\d+)\}", answer)
            if match:
                answer = match.group(1)

        reasoning = response_data.get("reasoning", "").strip() + "\nModel: mistral-7b-instruct-v0.3-bnb-4bit"

        sources = response_data.get("sources", [])

        formatted_response = {
            "answer": answer,
            "reasoning": reasoning,
            "sources": sources
        }

        return formatted_response


    def generate_response(self, query: str, top_k: int = 5) -> str:
        """
        Выполняет поиск в FAISS + генерирует ответ с LLM.
        """
        start_time = time.time()

        query_embedding = self.encoder.encode(query, convert_to_numpy=True).reshape(1, -1)

        print(f"Енкодинг занял: {time.time() - start_time:.4f} сек")
        retrieved_docs = self.db.search(query_embedding, top_k=top_k)

        context = "\n\n".join([f"{doc['title']} ({doc['url']}): {doc['content']}" for doc in retrieved_docs][::-1])
        if len(context) > 4097:
            context = context[-4096:] # двигаем более близкий контекст к началу генерации

        response_format = """{
            "answer": {Номер ответа, если нет правильного - -1},
            "reasoning": {Кратко обоснуй ответ, тут нельзя использовать ссылки!},
            "sources": {список ссылок на используемые источники, не более 3х, если они вообще требуются}
        }"""

        prompt = f"Используя следующую информацию: \n\nИнформация:\n{context}\n\nОтветьте на вопрос, при этом твой ответ должен быть строго в формате JSON, а именно:\n{response_format}\nВопрос: {query}\nОтвет:\n"
        with torch.inference_mode():
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')

            output = self.model.generate(input_ids, max_new_tokens=256)
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        formatted_response = self.format_llm_response(response.split('Ответ:\n')[1])


        print(f"🔍 Поиск и генерация заняли {time.time() - start_time:.4f} сек")
        return formatted_response


if __name__ == "__main__":
    db = RetrievalDatabase(1024)

    generator = LLMResponseGenerator(db)

    # 🔹 Запрос от пользователя
    query = "В каком году Университет ИТМО был включён в число Национальных исследовательских университетов России?\n1. 2007\n2. 2009\n3. 2011\n4. 2015"
    response = generator.generate_response(query)
    print(response)

    # Стресс тест
    start_time = time.time()
    for i in range(100):
        query = "В каком году Университет ИТМО был включён в число Национальных исследовательских университетов России?\n1. 2007\n2. 2009\n3. 2011\n4. 2015"
        response = generator.generate_response(query)
    print(f"🔍 Поиск и генерация заняли {time.time() - start_time:.4f} сек")
    #print("\n🤖 Ответ от LLM:")
    #print(response)
