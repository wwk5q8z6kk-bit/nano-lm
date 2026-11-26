"""Rebuild the fixed evaluation texts from the original US government publications."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata
from urllib.request import Request, urlopen

from tokenizers import Tokenizer


HERE = Path(__file__).resolve().parent
DATES = ("0129", "0319", "0507", "0618", "0730", "0917", "1029", "1210")


def sources():
    for date in DATES:
        yield "fed", date, f"https://www.federalreserve.gov/monetarypolicy/fomcminutes2025{date}.htm"
    for number in range(1, 9):
        yield "law", str(number), f"https://www.govinfo.gov/content/pkg/PLAW-119publ{number}/html/PLAW-119publ{number}.htm"
    for number in range(1, 9):
        storm = f"al{number:02d}"
        yield "weather", storm, f"https://www.nhc.noaa.gov/archive/2025/{storm}/{storm}2025.discus.003.shtml"


def extract(source, raw):
    if source == "fed":
        start = re.search(r"<strong>\s*Staff Review of the Economic Situation\s*</strong>", raw, re.I)
        if start is None:
            raise ValueError("Missing economic situation heading")
        section = re.split(r"<(?:strong|h[1-6])\b", raw[start.end():], maxsplit=1, flags=re.I)[0]
        content = html.unescape(re.sub(r"<[^>]+>", " ", section))
    else:
        pre = re.search(r"<pre\b[^>]*>(.*?)</pre>", raw, flags=re.I | re.S)
        if pre is None:
            raise ValueError(f"Missing {source} document")
        content = html.unescape(re.sub(r"<[^>]+>", "", pre.group(1)))
        if source == "law":
            start = re.search(r"(?:Be it enacted|Resolved by the Senate)", content, re.I)
            if start is None:
                raise ValueError("Missing law enactment")
            content = re.split(r"\bApproved,?\s|LEGISLATIVE HISTORY", content[start.start():], maxsplit=1)[0]
            content = re.sub(r"<<NOTE:.*?>>|\[\[Page .*?\]\]", "", content, flags=re.S)
        else:
            start = re.search(r"\b(?:AM|PM)\s+[A-Z]+\s+[^\n]*2025\s*\n", content)
            if start is None or "FORECAST POSITIONS AND MAX WINDS" not in content:
                raise ValueError("Missing NHC discussion boundaries")
            content = content[start.end():].split("FORECAST POSITIONS AND MAX WINDS")[0]
    return " ".join(unicodedata.normalize("NFC", content).split())


def clipped(tokenizer, text):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids[:1024]
    while ids:
        prefix = tokenizer.decode(ids, skip_special_tokens=False)
        if text.startswith(prefix) and tokenizer.encode(prefix, add_special_tokens=False).ids == ids:
            return prefix, ids
        ids.pop()
    raise ValueError("No round-tripping passage")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    path = args.data / "tokenizer_32k.json"
    expected = json.loads((HERE / "inputs.json").read_text())
    if sha(path.read_bytes()) != expected["tokenizer_sha256"]:
        raise ValueError("Wrong historical tokenizer")
    tok = Tokenizer.from_file(str(path))
    if tok.get_vocab_size() != 32768 or tok.token_to_id("<|eos|>") != 1:
        raise ValueError("Tokenizer vocabulary or boundary mismatch")
    rows = []
    for group, name, url in sources():
        doc = f"{group}-{name}"
        cached = args.cache / f"{doc}.html"
        if not cached.exists():
            with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as response:
                if response.geturl() != url:
                    raise ValueError(f"Unexpected redirect for {doc}")
                raw = response.read()
            cached.write_bytes(raw)
        try:
            normal = extract(group, cached.read_bytes().decode("utf-8"))
            passage, ids = clipped(tok, normal)
        except Exception as err:
            raise ValueError(f"Failed to prepare {doc}") from err
        if len(ids) < 16 or set(ids) & {0, 1}:
            raise ValueError(f"Invalid or empty passage {doc}")
        rows.append(dict(doc=doc, source=group, url=url, tokens=len(ids),
                         bytes=len(passage.encode()), sha256=sha(passage.encode()), text=passage))
    locks = [{k: v for k, v in row.items() if k != "text"} for row in rows]
    expected_locks = HERE / "sources.json"
    if expected_locks.exists() and locks != json.loads(expected_locks.read_text()):
        raise ValueError("Prepared source digest differs from locked manifest")
    write_json(args.output, rows)


if __name__ == "__main__":
    main()
