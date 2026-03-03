import re


def clean_extracted_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    return text.strip()
