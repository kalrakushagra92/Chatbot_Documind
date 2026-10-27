"""
PDF question answering with grounded, cited answers.

Pipeline
  1. Extract text block by block (+ tables as Markdown) with PyMuPDF.
     Repeated headers/footers and page numbers are removed.
  2. Split each page into overlapping ~1,000-character chunks.
  3. Hybrid search: BM25 keywords + vector similarity, fused with RRF.
  4. A cross-encoder reranks the candidates and drops weak ones.
  5. The LLM answers ONLY from the numbered passages and must cite them.
     If no passage is relevant enough, it says so instead of guessing.
  6. Citations are checked and shown as page numbers.

ChromaDB is synced incrementally: chunk IDs are content hashes, so
re-uploading an edited PDF only embeds the chunks that changed.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter
from pathlib import Path

import rag_core as rc

CHUNK_CHARS = 1000
CHUNK_OVERLAP = 200
CANDIDATES = 25           # per search method, per file
RERANK_POOL = 30          # candidates sent to the cross-encoder
TOP_SOURCES = 6           # passages given to the LLM
MIN_RERANK_SCORE = -4.0   # below this a passage is treated as irrelevant
SUMMARY_GROUP_CHARS = 9000
MAX_SUMMARY_GROUPS = 20

NOT_FOUND = "I could not find this information in the uploaded document."


# =====================================================
# TEXT EXTRACTION
# =====================================================

def _normalise_line(text):
    return re.sub(r"\d+", "#", text.strip().casefold())


def extract_pages(path):
    """Return [{"page": n, "text": str}] with clean paragraphs."""
    try:
        import pymupdf as fitz
    except ImportError:              # older PyMuPDF versions
        import fitz

    doc = fitz.open(path)
    raw_pages = []
    for page in doc:
        blocks = []
        for block in page.get_text("blocks", sort=True):
            if block[6] != 0:                 # image block
                continue
            text = block[4]
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # de-hyphenate
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if text:
                blocks.append(text)

        tables = []
        try:
            for table in page.find_tables().tables:
                md = table.to_markdown().strip()
                if md and md.count("|") > 4:
                    tables.append(md)
        except Exception:
            pass                               # older PyMuPDF / odd page

        raw_pages.append((page.number + 1, blocks, tables))
    doc.close()

    # Drop headers/footers: short blocks repeated on many pages.
    n_pages = len(raw_pages)
    repeated = set()
    if n_pages >= 3:
        counts = Counter()
        for _, blocks, _ in raw_pages:
            counts.update({_normalise_line(b) for b in blocks if len(b) < 120})
        repeated = {line for line, c in counts.items()
                    if c >= max(3, 0.5 * n_pages)}

    pages = []
    for number, blocks, tables in raw_pages:
        kept = [b for b in blocks
                if _normalise_line(b) not in repeated
                and not re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", b,
                                     flags=re.IGNORECASE)]
        text = "\n\n".join(kept)
        if tables:
            text += "\n\n" + "\n\n".join(f"Table:\n{t}" for t in tables)
        pages.append({"page": number, "text": text.strip()})
    return pages


def _split_long(paragraph, size):
    """Split a very long paragraph on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    parts, current = [], ""
    for s in sentences:
        while len(s) > size:                  # no punctuation at all
            parts.append(s[:size])
            s = s[size:]
        if current and len(current) + len(s) + 1 > size:
            parts.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_pages(pages, size=CHUNK_CHARS, overlap=CHUNK_OVERLAP):
    """Chunks never cross pages, so every citation maps to one page."""
    chunks = []
    for p in pages:
        paragraphs = []
        for para in p["text"].split("\n\n"):
            para = para.strip()
            if not para:
                continue
            if para.startswith("Table:"):     # keep tables whole if possible
                paragraphs.append(para)
            else:
                paragraphs.extend(_split_long(para, size))

        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > size:
                chunks.append({"page": p["page"], "text": current})
                tail = current[-overlap:]
                cut = tail.find(". ")
                tail = tail[cut + 2:] if cut != -1 else ""
                current = f"{tail}\n\n{para}".strip()
            else:
                current = f"{current}\n\n{para}".strip()
        if current:
            chunks.append({"page": p["page"], "text": current})

    seen = Counter()
    for c in chunks:
        base = hashlib.sha1(f"{c['page']}|{c['text']}".encode()).hexdigest()[:20]
        seen[base] += 1
        c["id"] = base if seen[base] == 1 else f"{base}_{seen[base]}"
    return chunks


# =====================================================
# ENGINE (one per PDF)
# =====================================================

class PDFRAG:

    def __init__(self, file_path, embedding_model=rc.EMBEDDING_MODEL,
                 db_path=rc.DB_PATH, embedding_function=None):
        self.file_path = str(Path(file_path).resolve())
        self.name = Path(file_path).name
        self.embedding_model = embedding_model
        self.db_path = db_path
        self._ef = embedding_function
        self._lock = threading.RLock()
        self._signature = None
        self.collection = None
        self.pages, self.chunks, self.warnings = [], [], []
        self.last_sync = {}

    def refresh(self, force=False):
        with self._lock:
            signature = rc.file_signature(self.file_path)
            if not force and signature == self._signature:
                return None

            pages = extract_pages(self.file_path)
            chunks = chunk_pages(pages)

            self.warnings = []
            text_chars = sum(len(p["text"]) for p in pages)
            if not chunks:
                self.warnings.append(
                    "No text found. This looks like a scanned PDF; run OCR "
                    "on it first (e.g. ocrmypdf) and upload again.")
            elif text_chars < 80 * len(pages):
                self.warnings.append(
                    "Very little text was found; some pages may be scanned "
                    "images, which can't be searched.")

            if self.collection is None:
                self.collection = rc.get_collection(
                    rc.collection_name("pdf", self.file_path, "",
                                       self.embedding_model),
                    self.file_path, self.embedding_model, self.db_path, self._ef)

            stats = rc.sync_collection(
                self.collection,
                [c["id"] for c in chunks],
                [c["text"] for c in chunks],
                [{"page": c["page"], "file": self.name} for c in chunks])

            self.pages, self.chunks = pages, chunks
            self.by_id = {c["id"]: c for c in chunks}
            self.bm25 = rc.BM25([c["text"] for c in chunks])
            self._signature = signature
            self.last_sync = stats
            return stats

    @property
    def word_count(self):
        return sum(len(p["text"].split()) for p in self.pages)

    def search(self, query, k=CANDIDATES):
        """Hybrid keyword + vector search, fused with RRF."""
        if not self.chunks:
            return []
        keyword = [self.chunks[i]["id"] for i in self.bm25.search(query, k)]
        vector = rc.vector_search(self.collection, query,
                                  self.embedding_model, k, self._ef)
        fused = rc.rrf_merge(vector, keyword)
        return [dict(self.by_id[i], file=self.name, rank=r)
                for r, i in enumerate(fused) if i in self.by_id]


# =====================================================
# RETRIEVE + ANSWER (works across one or many PDFs)
# =====================================================

ANSWER_PROMPT = """You are a careful document analyst. Answer the question
using ONLY the numbered sources below, which come from the user's PDF.

Rules:
- Every factual sentence or bullet must end with its citation, like [S2]
  or [S1][S4]. Cite only sources that actually say it.
- Copy numbers, names, dates and units exactly as written.
- If the sources only partly answer, give what they say and state what is
  missing.
- If the sources do not contain the answer, reply exactly:
  {not_found}
- Never use outside knowledge and never guess.
- Be clear and well organised: a direct answer first, then short bullet
  points if helpful. No preamble like "Based on the sources".
"""

REWRITE_PROMPT = """Rewrite the user's latest question as a standalone search
question, using the previous question only to resolve words like "it",
"they", "that" or "the second one". If it is already standalone, return it
unchanged. Return only the question."""


def _rewrite(question, previous, llm_model):
    if not previous:
        return question
    try:
        rewritten = rc.llm_chat(llm_model, [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Previous question: {previous}\n"
                                        f"Latest question: {question}"},
        ], num_predict=120).strip().strip('"')
        return rewritten if 3 < len(rewritten) < 400 else question
    except rc.OllamaUnavailable:
        raise
    except Exception:
        return question


def retrieve(engines, question, top=TOP_SOURCES):
    """Return (sources, reranked) - best passages across the given PDFs."""
    pool = []
    for engine in engines:
        pool.extend(engine.search(question))
    pool.sort(key=lambda c: c["rank"])
    pool = pool[:RERANK_POOL]

    scores = rc.rerank(question, [c["text"] for c in pool])
    if scores is None:                        # reranker not available
        for c in pool:
            c["score"] = None
        return pool[:top], False

    for c, s in zip(pool, scores):
        c["score"] = s
    pool.sort(key=lambda c: c["score"], reverse=True)
    return pool[:top], True


def prepare_answer(engines, question, llm_model, previous_question=None):
    """
    Returns (sources, messages, search_query).
    messages is None when nothing relevant was found (no LLM call made,
    so nothing can be invented).
    """
    search_query = _rewrite(question, previous_question, llm_model)
    sources, reranked = retrieve(engines, search_query)

    if reranked:
        strong = [s for s in sources if s["score"] >= MIN_RERANK_SCORE]
        if not strong:
            return sources[:3], None, search_query
        sources = strong

    # Keep reading order inside each page for a more coherent context.
    numbered = "\n\n".join(
        f"[S{i}] ({s['file']}, page {s['page']})\n{s['text']}"
        for i, s in enumerate(sources, 1))

    messages = [
        {"role": "system",
         "content": ANSWER_PROMPT.format(not_found=NOT_FOUND)},
        {"role": "user",
         "content": f"SOURCES:\n\n{numbered}\n\nQUESTION: {question}"},
    ]
    return sources, messages, search_query


def finalize(text, sources, multi_file=False):
    """
    Check citations and turn [S2] into page references.
    Returns (final_text, notes, ok).
    """
    text = (text or "").strip()
    if not text or text.startswith(NOT_FOUND[:30]):
        return NOT_FOUND, [], False

    notes = []
    cited = {int(n) for n in re.findall(r"\[S(\d+)\]", text)}
    invalid = {n for n in cited if not 1 <= n <= len(sources)}
    if invalid:
        notes.append("Some citations didn't match a passage and were removed.")
    if not cited - invalid:
        notes.append("The model didn't cite its sources - check the "
                     "passages below before relying on this answer.")

    def label(match):
        n = int(match.group(1))
        if not 1 <= n <= len(sources):
            return ""
        s = sources[n - 1]
        ref = f"{s['file']} p.{s['page']}" if multi_file else f"p.{s['page']}"
        return f" `{ref}`"

    final = re.sub(r"\s*\[S(\d+)\]", label, text)
    # Merge repeated identical references like `p.3` `p.3`
    final = re.sub(r"(`[^`]+`)(\s*\1)+", r"\1", final)

    for i, s in enumerate(sources, 1):
        s["cited"] = i in cited
    return final, notes, True


# =====================================================
# SUMMARY & INSIGHTS (map-reduce over the whole PDF)
# =====================================================

MAP_PROMPTS = {
    "summary": "Summarise the key points of this excerpt in 3-6 bullets. "
               "Use only what the text says; keep numbers and names exact.",
    "insights": "List the most important facts, figures, dates, findings, "
                "risks and recommendations in this excerpt as short bullets. "
                "Use only what the text says; keep numbers exact. If nothing "
                "important, reply 'Nothing notable'.",
}

REDUCE_PROMPTS = {
    "summary": """Write a clear summary of the whole document from these
section notes. Use this structure with Markdown headings:
### Overview (2-3 sentences)
### Key points (bullets)
### Important figures & dates (bullets, only if present)
### Conclusion
Use only the notes. Add page references like (p. 4) from the notes'
page ranges. Do not invent anything.""",
    "insights": """From these section notes, write the most valuable
insights in the document with Markdown headings:
### Key findings
### Numbers that matter
### Risks or issues
### Recommendations / next steps
Only include sections the notes support. Use bullets, keep numbers exact,
add page references like (p. 4). Do not invent anything.""",
}


def summarize(engine, llm_model, mode="summary", progress=None):
    """Return a generator that streams the final summary/insights."""
    groups, current, start = [], "", None
    for p in engine.pages:
        if not p["text"]:
            continue
        if start is None:
            start = p["page"]
        if current and len(current) + len(p["text"]) > SUMMARY_GROUP_CHARS:
            groups.append((start, prev, current))
            current, start = "", p["page"]
        current += f"\n\n[Page {p['page']}]\n{p['text']}"
        prev = p["page"]
    if current:
        groups.append((start, prev, current))

    note = ""
    if len(groups) > MAX_SUMMARY_GROUPS:
        note = (f"\n\n_Note: covers the first {MAX_SUMMARY_GROUPS} sections "
                f"(pages 1-{groups[MAX_SUMMARY_GROUPS - 1][1]}) of a long "
                "document._")
        groups = groups[:MAX_SUMMARY_GROUPS]

    if len(groups) == 1:                       # short document: one pass
        notes_text = groups[0][2]
    else:
        partials = []
        for i, (a, b, text) in enumerate(groups):
            if progress:
                progress(i / len(groups), f"Reading pages {a}-{b}...")
            bullet_notes = rc.llm_chat(llm_model, [
                {"role": "system", "content": MAP_PROMPTS[mode]},
                {"role": "user", "content": text},
            ], num_predict=400)
            partials.append(f"Notes for pages {a}-{b}:\n{bullet_notes}")
        notes_text = "\n\n".join(partials)
    if progress:
        progress(1.0, "Writing...")

    stream = rc.llm_stream(llm_model, [
        {"role": "system", "content": REDUCE_PROMPTS[mode]},
        {"role": "user", "content": notes_text},
    ], num_predict=1200)

    def with_note():
        yield from stream
        if note:
            yield note
    return with_note()
