import requests
from bs4 import BeautifulSoup
import json


def clean_news_content(raw_content : str) -> str:
    """
    Очищает текст контента от ненужных строк, пробелов и лишних символов.
    """
    lines = raw_content.split("\n")
    cleaned_lines = [
        line.strip() for line in lines if line.strip() and "К началу" not in line and "Фото: " not in line
    ]

    cleaned_content = "\n".join(cleaned_lines)
    return cleaned_content

def parse_news(url : str) -> str:
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
            'content': clean_news_content(content)
        }
        return news_data

    except Exception as e:
        return {'error': str(e), 'url': url}


if __name__ == '__main__':
    urls = [
        "https://news.itmo.ru/ru/education/official/news/13092/",
        "https://news.itmo.ru/ru/education/official/news/14134/",
        "https://news.itmo.ru/ru/university_live/achievements/news/14133/"
    ]

    news_list = []
    for url in urls:
        news = parse_news(url)
        news_list.append(news)

    with open('news.json', 'w', encoding='utf-8') as f:
        json.dump(news_list, f, ensure_ascii=False, indent=4)

    print("Новости успешно собраны и сохранены в news.json")
