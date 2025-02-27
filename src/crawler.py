import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time

from src.database import RetrievalDatabase

class ITMONewsCrawler:
    def __init__(self, base_url : str, db : RetrievalDatabase, encoder, delay : float = 1, storage_file : str = "processed_urls.json"):
        self.base_url = base_url
        self.db = db
        self.encoder = encoder
        self.delay = delay
        self.storage_file = storage_file
        self.visited_urls = self.load_processed_urls(storage_file)

    def load_processed_urls(self, filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except FileNotFoundError:
            return set()

    def save_processed_urls(self):
        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.visited_urls), f, ensure_ascii=False, indent=4)
        print(f"Saved {self.storage_file}")

    def get_sections(self, base_url : str) -> list[str]:
        try:
            response = requests.get(base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            news_links = []
            section_divs = soup.find_all('div', class_='contentbox')
            for section in section_divs:
                link = section.find('a').get('href')
                if link:
                    news_links.append(urljoin(base_url, link))

            return news_links
        except Exception as e:
            print(f"Ошибка при сборе разделов: {e}")
            return []

    def get_news_links(self, section_url : str) -> list[str]:
        try:
            response = requests.get(section_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            news_links = []
            section_elements = soup.find('div', class_='contentbox').find('ul').find_all('li')
            for element in section_elements:
                link = element.find('a').get('href')
                if link:
                    full_link = urljoin(section_url, link)
                    if full_link not in self.visited_urls:
                        news_links.append(full_link)
                    else:
                        break

            prev_next_action_urls = soup.find('div', class_='pagination').find_all('a')
            if len(prev_next_action_urls) > 1 and len(news_links) < 500:
                next_section_url = urljoin(section_url, prev_next_action_urls[1].get('href'))
                if next_section_url == section_url:
                    return news_links
                print(f'Visiting next section page: {next_section_url}')
                time.sleep(self.delay)
                try:
                    news_links += self.get_news_links(next_section_url)
                except Exception as e:
                    print(f'Limit exceed: {e}')
                    
            return news_links

        except Exception as e:
            print(f"Ошибка при сборе ссылок с раздела {section_url}: {e}")
            return []

    def clean_news_content(self, raw_content : str) -> list[str]:
        """
        Очищает текст контента от ненужных строк, пробелов и лишних символов.
        """
        lines = raw_content.split("\n")
        cleaned_lines = [
            line.strip().lower() for line in lines if line.strip() and "К началу" not in line and "Фото: " not in line
        ]
        return cleaned_lines

    def parse_news(self, url : str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.find('title').get_text(strip=True).lower()
            date_time = soup.find('div', class_='news-info-wrapper').find('time').get('datetime')
            content = soup.find('div', class_='content').get_text()
            clean_content = self.clean_news_content(content)

            news_data = []
            for paragraph in clean_content:
                embedding = self.encoder.encode(title + " " + date_time + " " + paragraph)
                news_data.append({
                    'url': url,
                    'title': title,
                    'date_time': date_time,
                    'content': paragraph,
                    'embedding': embedding
                })

            return news_data

        except ZeroDivisionError as e:
            return {'error': str(e), 'url': url}

    def crawl(self):
        sections = self.get_sections(self.base_url)
        max_retries_exceed = False

        for section_url in sections:
            section_news = []
            print(f"Сканирую раздел: {section_url}")
            try:
                news_links = self.get_news_links(section_url)
            except AttributeError as e:
                print(f"Ошибка при сборе новостных ссылок с раздела {section_url}: {e}")
                continue

            for news_url in news_links:
                print(f"Обрабатываю новость: {news_url}")
                try:
                    self.visited_urls.add(news_url)
                    news_data = self.parse_news(news_url)
                    section_news += news_data
                    time.sleep(self.delay)
                except Exception as e:
                    max_retries_exceed = True
                    print(f"max_retries exceed {e}")
                    break

            if section_news:
                self.db.insert_news(section_news)
                self.db.save_index()

            if max_retries_exceed:
                break

        self.save_processed_urls()
        print(f"Всего новостей в базе: {len(self.visited_urls)}")


if __name__ == '__main__':
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("intfloat/multilingual-e5-large", device='cuda')

    db = RetrievalDatabase(dimension=1024)
    BASE_URL = "https://news.itmo.ru/ru/"
    crawler = ITMONewsCrawler(BASE_URL, db, encoder)
    crawler.crawl()

    query = "какие правила приёма абитуриентов в 2025 году?"
    querry_embed = encoder.encode(query).reshape(1, -1)

    time_start = time.time()

    results = db.search(querry_embed, top_k=3)

    print("\nНайденные новости:")
    for res in results:
        print(f"{res['title']} ({res['url']})")
        print(f"{res['content']}\n")
    print(f'Заняло {time.time() - time_start}')
