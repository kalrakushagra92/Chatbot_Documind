"""
Shared building blocks: Ollama calls, embeddings, reranker, BM25,
ChromaDB incremental sync. Used by the PDF and the spreadsheet engines.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

# =====================================================
# CONFIGURATION
# =====================================================

LLM_MODEL = "qwen3:1.7b"

# bge-small is a clear step up from all-MiniLM-L6-v2 for retrieval
# while staying small and fast on CPU.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Cross-encoder that re-scores the best candidates: the biggest single
# improvement to answer relevance.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DB_PATH = str(Path(__file__).resolve().parent / "rag_db")
NUM_CTX = 8192
SYNC_BATCH = 256


class OllamaUnavailable(RuntimeError):
    """Ollama is not running or the model is not pulled."""


@dataclass
class Answer:
    text: str                       # what the user sees
    route: str                      # "pdf", "pandas", "rows" or "error"
    ok: bool = True                 # False = not found / failed
    table: pd.DataFrame | None = None
    plan: dict | None = None
    sources: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    exact: str | None = None        # exact computed result (spreadsheets)


# =====================================================
# MODELS (loaded once per process)
# =====================================================

@lru_cache(maxsize=4)
def get_embedding_function(model_name: str):
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
    return SentenceTransformerEmbeddingFunction(
        model_name=model_name, normalize_embeddings=True)


def query_prefix(model_name: str) -> str:
    """bge v1.5 models retrieve better with this instruction on queries."""
    name = model_name.lower()
    if "bge" in name and "m3" not in name:
        return "Represent this sentence for searching relevant passages: "
    return ""


@lru_cache(maxsize=4)
def get_chroma_client(path: str):
    import chromadb
    return chromadb.PersistentClient(path=path)


@lru_cache(maxsize=2)
def get_reranker(model_name: str):
    try:
        from sentence_transformers import CrossEncoder
        return CrossEncoder(model_name)
    except Exception:
        return None                 # app keeps working without it


def rerank(query, texts, model_name=RERANK_MODEL):
    """Relevance score per text (higher = better), or None if unavailable."""
    model = get_reranker(model_name)
    if model is None or not texts:
        return None
    scores = model.predict([(query, t) for t in texts], batch_size=16)
    return [float(s) for s in scores]


def relevance_percent(score):
    """Turn a cross-encoder logit into a friendly 0-100 bar."""
    return int(round(100 / (1 + math.exp(-score / 2))))


# =====================================================
# OLLAMA
# =====================================================

def _ollama():
    try:
        import ollama
        return ollama
    except ImportError as exc:
        raise OllamaUnavailable("Python package missing: pip install ollama") \
            from exc


def _friendly(exc, model):
    text = str(exc).lower()
    if getattr(exc, "status_code", None) == 404 or "not found" in text:
        return OllamaUnavailable(
            f"Model '{model}' is not installed. Run in a terminal: "
            f"ollama pull {model}")
    if "connect" in text or isinstance(exc, OSError):
        return OllamaUnavailable(
            "Cannot reach Ollama. Open the Ollama app (or run `ollama serve`) "
            "and try again.")
    return OllamaUnavailable(f"Ollama error: {exc}")


def _kwargs(model, messages, json_mode, num_predict, stream):
    kwargs = {"model": model, "messages": messages, "stream": stream,
              "options": {"temperature": 0, "top_p": 0.9,
                          "num_ctx": NUM_CTX, "num_predict": num_predict}}
    if json_mode:
        kwargs["format"] = "json"
    return kwargs


def _strip_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()


def llm_chat(model, messages, json_mode=False, num_predict=700):
    ollama = _ollama()
    kwargs = _kwargs(model, messages, json_mode, num_predict, stream=False)
    try:
        try:
            response = ollama.chat(**kwargs, think=False)
        except TypeError:                         # old ollama package
            response = ollama.chat(**kwargs)
        except ollama.ResponseError as exc:       # model without "think"
            if "think" not in str(exc).lower():
                raise
            response = ollama.chat(**kwargs)
    except Exception as exc:
        raise _friendly(exc, model) from exc
    return _strip_think(response["message"]["content"] or "")


def llm_stream(model, messages, num_predict=900):
    """Yield the answer piece by piece (for st.write_stream)."""
    ollama = _ollama()
    kwargs = _kwargs(model, messages, False, num_predict, stream=True)

    def pieces(use_think):
        try:
            stream = ollama.chat(**kwargs, think=False) if use_think \
                else ollama.chat(**kwargs)
        except TypeError:
            stream = ollama.chat(**kwargs)
        for part in stream:
            yield part["message"]["content"] or ""

    def generator():
        buffer, in_think, started = "", False, False
        attempts = [True, False]
        for use_think in attempts:
            try:
                for piece in pieces(use_think):
                    buffer += piece
                    # Hide any <think>...</think> block.
                    if not started and buffer.lstrip().startswith("<think>"):
                        in_think = True
                    if in_think:
                        if "</think>" in buffer:
                            buffer = buffer.split("</think>", 1)[1].lstrip()
                            in_think = False
                        else:
                            continue
                    if buffer.strip() or started:
                        started = True
                        yield buffer
                        buffer = ""
                return
            except Exception as exc:
                if use_think and not started and "think" in str(exc).lower():
                    continue                     # retry without "think"
                raise _friendly(exc, model) from exc

    return generator()


def check_ollama(model):
    """(ok, message) for the status pill."""
    try:
        listing = _ollama().list()
    except Exception:
        return False, "Ollama is not running"

    models = listing.get("models", []) if isinstance(listing, dict) \
        else getattr(listing, "models", [])
    names = set()
    for m in models:
        name = (m.get("model") or m.get("name")) if isinstance(m, dict) \
            else getattr(m, "model", None)
        if name:
            names.update({name, name.removesuffix(":latest")})
    if model in names or model.removesuffix(":latest") in names:
        return True, f"{model} ready"
    return False, f"Run: ollama pull {model}"


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError("The model did not return valid JSON.")


# =====================================================
# KEYWORD SEARCH (BM25) - catches exact terms vectors miss
# =====================================================

STOPWORDS = set("""a an the and or of to in on for with by from at as is are
was were be been being it its this that these those what which who whom
whose when where why how do does did can could should would will shall may
might must i me my we our you your he she they them their his her not no
yes if then than so such about into over under also there here any all
please tell give show find""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*",
                                  text.casefold())
            if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.docs else 0
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5))
                    for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query, k=25):
        terms = tokenize(query)
        if not terms or not self.docs:
            return []
        scores = []
        for i, (d, length) in enumerate(zip(self.docs, self.lengths)):
            s = 0.0
            for t in terms:
                f = d.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * length / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


def rrf_merge(*rankings, k=60):
    """Reciprocal Rank Fusion of several ranked id lists."""
    scores = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


# =====================================================
# CHROMADB: incremental sync (never wipe & rebuild)
# =====================================================

def file_signature(path):
    stat = Path(path).stat()
    return stat.st_mtime_ns, stat.st_size


def collection_name(kind, path, extra, embedding_model):
    key = f"{Path(path).resolve()}|{extra}|{embedding_model}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9]+", "_", Path(path).stem).strip("_")[:36]
    return f"{kind}_{stem or 'file'}_{digest}"


def get_collection(name, path, embedding_model, db_path=DB_PATH,
                   embedding_function=None):
    client = get_chroma_client(db_path)
    ef = embedding_function or get_embedding_function(embedding_model)
    return client.get_or_create_collection(
        name=name, embedding_function=ef,
        metadata={"hnsw:space": "cosine",
                  "source_file": str(Path(path).resolve()),
                  "embedding_model": embedding_model})


def sync_collection(coll, ids, documents, metadatas):
    """
    Bring a collection in line with the current file:
      new ids -> embedded, changed text -> re-embedded,
      same text but new metadata (e.g. row moved) -> metadata only,
      ids no longer present -> removed.
    """
    stored = {}
    offset, page = 0, 5000
    while True:
        got = coll.get(include=["metadatas"], limit=page, offset=offset)
        for doc_id, meta in zip(got["ids"], got["metadatas"]):
            meta = meta or {}
            stored[doc_id] = (meta.get("hash"), meta.get("sig"))
        if len(got["ids"]) < page:
            break
        offset += page

    to_embed, to_update = [], []
    for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
        meta["hash"] = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        meta["sig"] = hashlib.md5(json.dumps(
            {k: v for k, v in meta.items() if k != "sig"},
            sort_keys=True, default=str).encode()).hexdigest()
        old_hash, old_sig = stored.get(doc_id, (None, None))
        if old_hash != meta["hash"]:
            to_embed.append(i)
        elif old_sig != meta["sig"]:
            to_update.append(i)

    for start in range(0, len(to_embed), SYNC_BATCH):
        batch = to_embed[start:start + SYNC_BATCH]
        coll.upsert(ids=[ids[i] for i in batch],
                    documents=[documents[i] for i in batch],
                    metadatas=[metadatas[i] for i in batch])
    for start in range(0, len(to_update), SYNC_BATCH):
        batch = to_update[start:start + SYNC_BATCH]
        coll.update(ids=[ids[i] for i in batch],
                    metadatas=[metadatas[i] for i in batch])

    current = set(ids)
    removed = [doc_id for doc_id in stored if doc_id not in current]
    for start in range(0, len(removed), SYNC_BATCH):
        coll.delete(ids=removed[start:start + SYNC_BATCH])

    added = sum(1 for i in to_embed if ids[i] not in stored)
    return {"added": added, "updated": len(to_embed) - added,
            "moved": len(to_update), "deleted": len(removed),
            "unchanged": len(ids) - len(to_embed) - len(to_update),
            "total": coll.count()}


def vector_search(coll, query, embedding_model, k, embedding_function=None):
    """Return ids ranked by vector similarity."""
    count = coll.count()
    if count == 0:
        return []
    ef = embedding_function or get_embedding_function(embedding_model)
    embedding = ef([query_prefix(embedding_model) + query])
    res = coll.query(query_embeddings=embedding, n_results=min(k, count),
                     include=[])
    return res["ids"][0]


def forget_file(path, db_path=DB_PATH):
    """Delete the collections of a file the user removed from the app."""
    client = get_chroma_client(db_path)
    target = str(Path(path).resolve())
    for coll in client.list_collections():
        if isinstance(coll, str):
            coll = client.get_collection(coll)
        if (coll.metadata or {}).get("source_file") == target:
            client.delete_collection(coll.name)
