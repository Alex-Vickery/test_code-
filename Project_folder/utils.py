# utils.py
import hashlib, datetime, yaml
import pandas as pd

def textwrap_fixed(s: str, width: int) -> str:
    import textwrap
    lines = []
    for line in s.splitlines():
        wrapped = textwrap.wrap(line, width=width) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)

def file_sha256(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()

def normalize_value(val):
    if pd.isna(val):
        return ""
    if isinstance(val, float):
        return f"{val:.12g}"
    return str(val).strip()

def rule_hash(rule_obj) -> str:
    dumped = yaml.safe_dump(rule_obj, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

def row_hash(row: pd.Series, cols: list, rule_hash_str: str) -> str:
    parts = []
    for c in cols:
        if c == "individual_id":
            continue
        parts.append(normalize_value(row[c]))
    parts.append(rule_hash_str)
    s = "|".join(parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def now_str() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
