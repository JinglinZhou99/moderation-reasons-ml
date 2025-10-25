import re
def clean_text(s: str) -> str:
    s = re.sub(r'https?://\S+', ' URL ', s)
    s = re.sub(r'@[A-Za-z0-9_]+', ' USER ', s)
    return s.strip()
