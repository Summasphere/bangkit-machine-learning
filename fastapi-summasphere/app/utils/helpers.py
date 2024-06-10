import ast

import requests
from bs4 import BeautifulSoup


def sanitize_text(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")


def string_to_object(string: str):
    return ast.literal_eval(string.strip(";"))


def fetch_html(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        return None


def parse_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup


def extract_info(soup):
    # Example: Extract text from all paragraphs
    txt_Arr = []
    paragraphs = soup.find_all("p")
    for para in paragraphs:
        txt_Arr.append(para.get_text())
    return txt_Arr


def process_url(url):
    html_content = fetch_html(url)
    if html_content:
        soup = parse_html(html_content)
        text = extract_info(soup)
        text = " ".join(text)
        return text
