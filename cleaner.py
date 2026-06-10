import re

def clean_text(text: str) -> str:
    # Normalize excessive newlines → max double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces/tabs into single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace form feed characters
    text = re.sub(r'\f', '\n', text)
    # Normalize repeated punctuation
    text = re.sub(r'\.{3,}', '...', text)
    text = re.sub(r'-{4,}', '---', text)
    # Remove decorative separator lines (===, ---, ___)
    text = re.sub(r'^[=\-_]{3,}\s*$', '', text,
                  flags=re.MULTILINE)
    # Remove page number artifacts
    text = re.sub(r'\b(Page|PAGE)\s+\d+\b', '', text)
    # Replace URLs and emails with tokens
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
    # Drop lines that are too short to be content (<10 chars)
    lines = [l.strip() for l in text.split('\n')]
    lines = [l for l in lines if len(l) >= 40]
    return '\n'.join(lines).strip()

def clean_documents(docs: list) -> list:
    cleaned = []
    total_removed = 0
    for doc in docs:
        original_len = len(doc.page_content)
        doc.page_content = clean_text(doc.page_content)
        cleaned_len = len(doc.page_content)
        removed = original_len - cleaned_len
        total_removed += removed
        cleaned.append(doc)
    print(f"  Cleaned {len(docs)} docs — "
          f"removed {total_removed} noise chars total")
    return cleaned