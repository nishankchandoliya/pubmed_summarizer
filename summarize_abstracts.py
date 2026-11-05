# summarize_abstracts.py
# cleans the messy pubmed json and asks our local llama 3.3 70b (vllm on the dgx spark)
# to summarize the clinical outcome of each study in one sentence
#
# how to run:
#   python summarize_abstracts.py
#   python summarize_abstracts.py abstracts.json --output results.json

import argparse
import json
import logging
import os
import re
import sys
import time

import requests

API_URL = os.getenv("LLM_API_URL", "http://192.168.1.100:8000/v1/chat/completions")
MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.3-70b")  # same as --served-model-name in the docker command

# 70b on the spark only does around 5 tokens per sec, so we give it time
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 120
MAX_TRIES = 3
RETRY_WAIT = 5  # wait 5s after the 1st fail and 10s after the 2nd
RETRY_CODES = {429, 500, 502, 503, 504}

MAX_ABSTRACT_CHARS = 6000  # around 1500 tokens, way under the 8192 max model len
MAX_SUMMARY_TOKENS = 100

PROMPT = "Summarize the clinical outcome of this study in one sentence."
SYSTEM_PROMPT = "You summarize medical study abstracts: only use facts that are written in the abstract"

# the old db uses different names for the same field, we check all of them in lower case
TITLE_KEYS = ["articletitle", "title"]
ABSTRACT_KEYS = ["abstract_text", "abstracttext", "abstract"]
AUTHOR_KEYS = ["authorlist", "authors"]

# only a full match counts as empty, "Null hypothesis rejected" is a real abstract
EMPTY_VALUES = {"", "null", "none", "n/a", "na", "unknown"}

log = logging.getLogger("summarizer")


class LLMError(Exception):
    pass


def clean_text(value):
    # returns None if the value is missing, not a string or junk like "null"
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if value.lower() in EMPTY_VALUES:
        return None
    return value


def first_text(fields, keys):
    for key in keys:
        text = clean_text(fields.get(key))
        if text:
            return text
    return None


def clean_id(raw_id):
    # ids like "PMID:349281" and 8829102 become plain digits, "ERR_NOT_FOUND" gives None
    match = re.fullmatch(r"(?:pmid:?\s*)?(\d+)", str(raw_id).strip(), re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def clean_authors(raw_authors):
    # can be a list, one string or just "Unknown"
    if isinstance(raw_authors, list):
        names = [clean_text(name) for name in raw_authors]
        return [name for name in names if name]
    text = clean_text(raw_authors)
    if not text:
        return []
    return [name.strip() for name in text.split(";") if name.strip()]


def clean_record(raw):
    # returns (record, None) if we can use it, or (None, reason) if we skip it
    if not isinstance(raw, dict):
        return None, "entry is not a json object"

    fields = {str(key).lower(): value for key, value in raw.items()}

    raw_id = fields.get("id")
    if raw_id is None:
        raw_id = fields.get("pmid")
    pmid = clean_id(raw_id)
    abstract = first_text(fields, ABSTRACT_KEYS)

    problems = []
    if not pmid:
        problems.append(f"bad id {raw_id}")
    if not abstract:
        problems.append("no abstract text")
    if problems:
        return None, ", ".join(problems)

    if len(abstract) > MAX_ABSTRACT_CHARS:
        log.warning("pmid %s: abstract too long, cut to %d chars", pmid, MAX_ABSTRACT_CHARS)
        abstract = abstract[:MAX_ABSTRACT_CHARS]

    authors = []
    for key in AUTHOR_KEYS:
        authors = clean_authors(fields.get(key))
        if authors:
            break

    record = {
        "pmid": pmid,
        "title": first_text(fields, TITLE_KEYS),
        "authors": authors,
        "abstract": abstract,
    }
    return record, None


def load_records(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("top level json should be a list of records")
    return data


def clean_all(raw_records):
    good = []
    skipped = []
    seen = set()
    for position, raw in enumerate(raw_records, start=1):
        record, reason = clean_record(raw)
        if record and record["pmid"] in seen:
            record, reason = None, f"duplicate pmid {record['pmid']}"
        if record is None:
            log.warning("skipped entry %d: %s", position, reason)
            raw_id = raw.get("id") if isinstance(raw, dict) else None
            skipped.append({"position": position, "id": raw_id, "reason": reason})
            continue
        seen.add(record["pmid"])
        good.append(record)
    return good, skipped


def build_payload(record):
    lines = [PROMPT, ""]
    if record["title"]:
        lines.append(f"Title: {record['title']}")
    lines.append(f"Abstract: {record['abstract']}")
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "temperature": 0,
        "max_tokens": MAX_SUMMARY_TOKENS,
    }


def read_summary(response):
    try:
        choice = response.json()["choices"][0]
        summary = (choice["message"]["content"] or "").strip()
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError("server sent a reply we could not read")
    if not summary:
        raise LLMError("server sent an empty summary")
    if choice.get("finish_reason") == "length":
        log.warning("summary hit max_tokens, it may be cut off")
    return summary


def ask_llm(session, payload):
    error = None
    for attempt in range(1, MAX_TRIES + 1):
        try:
            response = session.post(API_URL, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
        # connect timeout has to be caught first because it is also a ConnectionError
        except requests.exceptions.ConnectTimeout:
            error = f"could not connect within {CONNECT_TIMEOUT}s"
        except requests.exceptions.ReadTimeout:
            error = f"no answer within {READ_TIMEOUT}s"
        except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError) as err:
            error = f"connection failed or dropped ({type(err).__name__})"
        else:
            if response.status_code == 200:
                return read_summary(response)
            if response.status_code not in RETRY_CODES:
                # a 400 or 404 will fail the same way every time, so no point retrying
                raise LLMError(f"server said {response.status_code}: {response.text[:200]}")
            error = f"server busy ({response.status_code})"

        if attempt < MAX_TRIES:
            wait = RETRY_WAIT * attempt
            log.warning("try %d of %d failed: %s, trying again in %ds", attempt, MAX_TRIES, error, wait)
            time.sleep(wait)

    raise LLMError(f"gave up after {MAX_TRIES} tries: {error}")


def summarize_all(records):
    # one request at a time on purpose, we don't want to flood the server
    results = []
    with requests.Session() as session:
        session.trust_env = False  # ignore any proxy in env vars, the server is on our lan
        for number, record in enumerate(records, start=1):
            log.info("record %d of %d: pmid %s", number, len(records), record["pmid"])
            result = {"pmid": record["pmid"], "title": record["title"], "authors": record["authors"]}
            try:
                result["summary"] = ask_llm(session, build_payload(record))
                result["status"] = "ok"
            except LLMError as err:
                log.error("pmid %s failed: %s", record["pmid"], err)
                result["summary"] = None
                result["status"] = "failed"
                result["error"] = str(err)
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="summarize pubmed abstracts with our local llama server")
    parser.add_argument("input", nargs="?", default="abstracts.json", help="raw json file")
    parser.add_argument("--output", default="results.json", help="where to save the summaries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        raw_records = load_records(args.input)
    except (OSError, ValueError) as err:
        log.error("could not read %s: %s", args.input, err)
        return 1

    records, skipped = clean_all(raw_records)
    log.info("records to summarize: %d, skipped: %d", len(records), len(skipped))

    results = summarize_all(records)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump({"results": results, "skipped": skipped}, file, indent=2, ensure_ascii=False)

    failed = sum(1 for result in results if result["status"] == "failed")
    log.info("saved %s (ok: %d, failed: %d, skipped: %d)", args.output, len(results) - failed, failed, len(skipped))

    # exit code 1 if anything failed, so cron or ci can catch it
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
