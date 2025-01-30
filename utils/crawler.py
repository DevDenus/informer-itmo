import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time

class ITMONewsCrawler:
    def __init__(self, base_url : str, delay : float = 1, storage_file : str = "processed_urls.json"):
        self.base_url = base_url
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
            json.dump(self.visited_urls, f, ensure_ascii=False, indent=4)
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
            seen_visited_pages = False

            news_links = []
            section_elements = soup.find('div', class_='contentbox').find('ul').find_all('li')
            for element in section_elements:
                link = element.find('a').get('href')
                if link:
                    full_link = urljoin(section_url, link)
                    if full_link not in self.visited_urls:
                        news_links.append(full_link)
                    else:
                        seen_visited_pages = True
            if not seen_visited_pages:
                prev_next_action_urls = soup.find('div', class_='pagination').find_all('a')
                if len(prev_next_action_urls) > 1:
                    next_section_url = urljoin(section_url, prev_next_action_urls[1].get('href'))
                    print(f'Visiting next section page: {next_section_url}')
                    time.sleep(self.delay)
                    news_links += self.get_news_links(next_section_url)
            return news_links

        except Exception as e:
            print(f"Ошибка при сборе ссылок с раздела {section_url}: {e}")
            return []

    def clean_news_content(self, raw_content : str) -> str:
        """
        Очищает текст контента от ненужных строк, пробелов и лишних символов.
        """
        lines = raw_content.split("\n")
        cleaned_lines = [
            line.strip() for line in lines if line.strip() and "К началу" not in line and "Фото: " not in line
        ]

        cleaned_content = "\n".join(cleaned_lines)
        return cleaned_content

    def parse_news(self, url : str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.find('title').get_text(strip=True)
            date = soup.find('div', class_='news-info-wrapper').find('time').get_attribute_list('datetime')[0]
            print(date)
            content = soup.find('div', class_='content').get_text()

            news_data = {
                'url': url,
                'title': title,
                'date': date,
                'content': self.clean_news_content(content)
            }
            return news_data

        except Exception as e:
            return {'error': str(e), 'url': url}

    def crawl(self):
        all_news = []
        sections = [self.get_sections(self.base_url)[0]]

        for section_url in sections:
            print(f"Сканирую раздел: {section_url}")
            try:
                news_links = self.get_news_links(section_url)
            except AttributeError as e:
                print(f"Ошибка при сборе новостных ссылок с раздела {section_url}: {e}")
                continue

            for news_url in news_links:
                print(f"Обрабатываю новость: {news_url}")
                self.visited_urls.add(news_url)
                news_data = self.parse_news(news_url)
                all_news.append(news_data)
                time.sleep(self.delay)  # Чтобы не перегружать сервер

        self.save_processed_urls()


if __name__ == '__main__':
    BASE_URL = "https://news.itmo.ru/ru/"
    crawler = ITMONewsCrawler(BASE_URL)
    crawler.crawl()
