import requests
from bs4 import BeautifulSoup


def scrape_page(url):

    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove obvious junk
        for tag in soup(["nav", "footer", "header", "aside", "script", "style"]):
            tag.decompose()

        content = None

        # Try common documentation containers
        selectors = [
            "main",
            "article",
            ".s-prose",              # StackOverflow answers/questions
            ".postcell",             # StackOverflow question body
            ".answercell",           # StackOverflow answer body
            ".md-content",           # many docs frameworks
            ".markdown-body",        # GitHub docs
        ]

        for sel in selectors:
            content = soup.select_one(sel)
            if content:
                break

        if not content:
            content = soup

        paragraphs = content.find_all("p")

        text = "\n".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 40
        )

        return {
            "url": url,
            "text": text
        }

    except Exception as e:
        print("Scrape failed:", url, e)
        return None