"""
DocuMind - chat with your PDFs, Excel sheets and CSV files.

Start with:   python app.py      (or: streamlit run app.py)
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent


def _inside_streamlit():
    try:
        from streamlit.runtime import exists
        return exists()
    except Exception:
        return False


# "python app.py" -> relaunch through Streamlit (opens the browser).
if __name__ == "__main__" and not _inside_streamlit():
    import subprocess
    sys.exit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(APP_DIR / "app.py"),
         *sys.argv[1:]], cwd=APP_DIR))


import hashlib  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import streamlit as st  # noqa: E402

sys.path.insert(0, str(APP_DIR))
import pdf_engine as pe  # noqa: E402
import rag_core as rc  # noqa: E402
import table_engine as te  # noqa: E402
from rag_core import Answer  # noqa: E402

KB_DIR = APP_DIR / "knowledge_base"
LINKS_FILE = KB_DIR / "_linked.json"
KB_DIR.mkdir(exist_ok=True)

FILE_TYPES = ["pdf", "xlsx", "xlsm", "xls", "csv"]
ICONS = {"pdf": "📕", "xlsx": "📗", "xlsm": "📗", "xls": "📗", "csv": "📄"}
ALL_PDFS = "__all_pdfs__"
SUMMARY_RE = re.compile(
    r"\b(summari[sz]e|summary|overview|tl;?dr|key (points|takeaways|insights|"
    r"findings)|main (points|insights|findings)|what is (this|the) "
    r"(document|pdf|file|report) about)\b", re.IGNORECASE)

st.set_page_config(page_title="DocuMind · Ask your documents",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")


# =====================================================
# STYLE
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea, select {
  font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 6rem; max-width: 1150px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- hero ---------- */
.hero { position: relative; overflow: hidden; border-radius: 24px;
  padding: 26px 32px; margin-bottom: 18px; color: #fff;
  background: linear-gradient(120deg, #5B4BF5 0%, #8B5CF6 50%, #06B6D4 100%);
  box-shadow: 0 14px 34px rgba(91, 75, 245, .28); }
.hero::before, .hero::after { content: ""; position: absolute; border-radius: 50%;
  background: rgba(255,255,255,.13); }
.hero::before { width: 260px; height: 260px; right: -70px; top: -110px; }
.hero::after  { width: 160px; height: 160px; right: 140px; bottom: -100px; }
.hero h1 { color: #fff !important; font-size: 2.15rem; font-weight: 800;
  margin: 8px 0 4px; padding: 0; letter-spacing: -.5px; }
.hero p { margin: 0; opacity: .93; font-size: 1.02rem; max-width: 640px; }
.chip-light { display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: .74rem; font-weight: 600; letter-spacing: .4px;
  background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.32); }

/* ---------- stat cards ---------- */
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px; margin: 2px 0 16px; }
.stat { background: #fff; border: 1px solid #E7E8F3; border-radius: 16px;
  padding: 14px 16px; box-shadow: 0 2px 10px rgba(31,35,64,.04); }
.stat .k { font-size: .72rem; color: #6B6F8E; font-weight: 600;
  text-transform: uppercase; letter-spacing: .6px; }
.stat .v { font-size: 1.3rem; font-weight: 700; color: #1E2140; margin-top: 3px;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* ---------- feature cards (empty state) ---------- */
.features { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 14px; margin-top: 6px; }
.feature { background: #fff; border: 1px solid #E7E8F3; border-radius: 18px;
  padding: 20px; box-shadow: 0 4px 16px rgba(31,35,64,.05); }
.feature .ic { font-size: 1.6rem; width: 46px; height: 46px; border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #EEEBFF, #E0F7FB); margin-bottom: 10px; }
.feature h4 { margin: 0 0 4px; font-size: 1.02rem; color: #1E2140; }
.feature p { margin: 0; color: #5D6180; font-size: .9rem; line-height: 1.45; }

/* ---------- badges ---------- */
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px;
  border-radius: 999px; font-size: .74rem; font-weight: 600; margin-bottom: 6px; }
.badge.pdf { background: #FFF1F2; color: #BE123C; }
.badge.calc { background: #ECFDF5; color: #047857; }
.badge.rows { background: #EFF6FF; color: #1D4ED8; }
.badge.sum { background: #F5F3FF; color: #6D28D9; }
.badge.warn { background: #FFFBEB; color: #B45309; }

/* ---------- source cards ---------- */
.src { border: 1px solid #E7E8F3; border-radius: 14px; padding: 12px 14px;
  margin-bottom: 10px; background: #FBFBFE; }
.src.cited { border-left: 4px solid #6D5DFC; background: #fff; }
.src-head { display: flex; align-items: center; gap: 8px; margin-bottom: 6px;
  flex-wrap: wrap; }
.pg { background: #6D5DFC; color: #fff; border-radius: 8px; padding: 1px 8px;
  font-size: .74rem; font-weight: 700; }
.fn { color: #5D6180; font-size: .8rem; font-weight: 500; }
.rel { margin-left: auto; display: flex; align-items: center; gap: 6px;
  font-size: .72rem; color: #6B6F8E; }
.bar { width: 70px; height: 6px; border-radius: 6px; background: #E9E9F5; overflow: hidden; }
.bar span { display: block; height: 100%;
  background: linear-gradient(90deg, #8B5CF6, #06B6D4); }
.src-body { color: #33374F; font-size: .87rem; line-height: 1.5; white-space: pre-wrap; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background: #fff; border: 1px solid #ECEDF6;
  border-radius: 18px; padding: 14px 18px; margin-bottom: 10px;
  box-shadow: 0 2px 12px rgba(31,35,64,.04); }
[data-testid="stChatMessage"] code { background: #EEEBFF; color: #5B4BF5;
  border-radius: 6px; padding: 1px 6px; font-size: .78rem; font-weight: 600; }
[data-testid="stChatInput"] textarea { font-size: 1rem; }
.stButton > button { border-radius: 12px; font-weight: 600;
  border: 1px solid #E1E2F0; transition: all .15s ease; }
.stButton > button:hover { border-color: #6D5DFC; color: #5B4BF5;
  transform: translateY(-1px); box-shadow: 0 4px 12px rgba(91,75,245,.15); }
[data-testid="stExpander"] { border-radius: 14px; border: 1px solid #E7E8F3; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px;
  font-weight: 600; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #FFFFFF 0%, #F6F5FF 100%);
  border-right: 1px solid #ECEBFA; }
.brand { display: flex; align-items: center; gap: 10px; margin: 2px 0 14px; }
.brand .logo { width: 40px; height: 40px; border-radius: 12px; font-size: 1.3rem;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #5B4BF5, #06B6D4); box-shadow: 0 6px 16px rgba(91,75,245,.3); }
.brand .name { font-weight: 800; font-size: 1.25rem; color: #1E2140; line-height: 1.1; }
.brand .tag { font-size: .74rem; color: #6B6F8E; }
.side-title { font-size: .72rem; font-weight: 700; color: #6B6F8E; text-transform: uppercase;
  letter-spacing: .7px; margin: 8px 0 6px; }
.file { display: flex; align-items: center; gap: 8px; background: #fff;
  border: 1px solid #E7E8F3; border-radius: 12px; padding: 8px 10px; font-size: .86rem;
  color: #1E2140; overflow: hidden; }
.file .nm { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.file .lk { font-size: .66rem; background: #E0F7FB; color: #0E7490;
  border-radius: 6px; padding: 0 6px; margin-left: auto; }
.status { display: flex; align-items: center; gap: 8px; font-size: .82rem; font-weight: 600;
  padding: 8px 12px; border-radius: 12px; }
.status.ok { background: #ECFDF5; color: #047857; }
.status.bad { background: #FEF2F2; color: #B91C1C; }
.dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.page-box { background: #fff; border: 1px solid #E7E8F3; border-radius: 16px;
  padding: 18px 20px; white-space: pre-wrap; line-height: 1.6; color: #33374F;
  font-size: .92rem; max-height: 520px; overflow-y: auto; }
</style>
""", unsafe_allow_html=True)


# =====================================================
# KNOWLEDGE BASE (files persist between restarts)
# =====================================================

def load_links():
    try:
        return json.loads(LINKS_FILE.read_text())
    except Exception:
        return []


def save_links(links):
    LINKS_FILE.write_text(json.dumps(sorted(set(links)), indent=2))


def list_sources():
    items = []
    for p in sorted(KB_DIR.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower().lstrip(".") in FILE_TYPES:
            items.append({"path": str(p), "name": p.name, "linked": False})
    for link in load_links():
        if Path(link).exists():
            items.append({"path": link, "name": Path(link).name, "linked": True})
    for it in items:
        ext = Path(it["path"]).suffix.lower().lstrip(".")
        it["ext"] = ext
        it["kind"] = "pdf" if ext == "pdf" else "table"
        it["icon"] = ICONS.get(ext, "📄")
    return items


def save_upload(uploaded):
    dest = KB_DIR / Path(uploaded.name).name
    data = uploaded.getvalue()
    # Rewrite only if the content changed, so the index stays warm.
    if not dest.exists() or hashlib.sha256(dest.read_bytes()).digest() != \
            hashlib.sha256(data).digest():
        dest.write_bytes(data)


def remove_source(src):
    try:
        rc.forget_file(src["path"])
    except Exception:
        pass
    if src["linked"]:
        save_links([p for p in load_links() if p != src["path"]])
    else:
        Path(src["path"]).unlink(missing_ok=True)
    st.session_state.chats.pop(src["path"], None)


# =====================================================
# CACHED ENGINES
# =====================================================

@st.cache_resource(show_spinner=False, max_entries=30)
def get_pdf_engine(path):
    return pe.PDFRAG(path)


@st.cache_resource(show_spinner=False, max_entries=30)
def get_table_engine(path, sheet, key_column):
    return te.TableRAG(path, sheet=sheet, key_column=key_column)


@st.cache_data(ttl=15, show_spinner=False)
def ollama_status(model):
    return rc.check_ollama(model)


@st.cache_data(show_spinner=False)
def sheets_of(path, signature):
    return te.list_sheets(path)


@st.cache_data(show_spinner=False)
def columns_of(path, sheet, signature):
    return te.read_columns(path, sheet)


def refresh_engine(engine, label):
    first = not getattr(engine, "_signature", None)
    msg = (f"Reading and indexing **{label}** (first time only)..."
           if first else "Checking for changes...")
    with st.spinner(msg):
        stats = engine.refresh()
    if stats and (stats["added"] or stats["updated"] or stats["deleted"]):
        unit = "passages" if isinstance(engine, pe.PDFRAG) else "rows"
        st.toast(f"{label}: {stats['added']} new, {stats['updated']} changed, "
                 f"{stats['deleted']} removed {unit} indexed", icon="✅")
    return stats


# =====================================================
# SMALL UI HELPERS
# =====================================================

def stat_cards(items):
    cells = "".join(f'<div class="stat"><div class="k">{html.escape(k)}</div>'
                    f'<div class="v">{html.escape(str(v))}</div></div>'
                    for k, v in items)
    st.markdown(f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def badge(kind, text):
    st.markdown(f'<span class="badge {kind}">{text}</span>',
                unsafe_allow_html=True)


def source_cards(sources):
    for s in sources:
        score = s.get("score")
        rel = ""
        if score is not None:
            pct = rc.relevance_percent(score)
            rel = (f'<span class="rel">relevance <span class="bar">'
                   f'<span style="width:{pct}%"></span></span>{pct}%</span>')
        text = s["text"] if len(s["text"]) < 900 else s["text"][:900] + " …"
        st.markdown(
            f'<div class="src {"cited" if s.get("cited") else ""}">'
            f'<div class="src-head"><span class="pg">p. {s["page"]}</span>'
            f'<span class="fn">{html.escape(s["file"])}</span>'
            f'{"<span class=fn>· cited</span>" if s.get("cited") else ""}{rel}</div>'
            f'<div class="src-body">{html.escape(text)}</div></div>',
            unsafe_allow_html=True)


def render_answer(ans, key):
    if ans.route == "error":
        st.error(ans.text, icon="⚠️")
        return

    labels = {"pdf": ("pdf", "📕 Answer from your PDF"),
              "summary": ("sum", "📝 Whole-document reading"),
              "pandas": ("calc", "🧮 Exact calculation on the full sheet"),
              "rows": ("rows", "🔎 Looked up in matching rows")}
    kind, label = labels.get(ans.route, ("rows", ans.route))
    if not ans.ok:
        kind, label = "warn", "⚠️ Not found in your file"
    badge(kind, label)

    text = ans.text
    if ans.route == "pandas":
        text = text.replace("\n", "  \n")
    st.markdown(text)

    for note in ans.notes:
        st.caption(f"ℹ️ {note}")

    if ans.exact and st.session_state.get("show_details", True):
        with st.expander("🧮 Exact result and how it was calculated"):
            st.code(ans.exact, language=None)
            if ans.plan:
                st.caption("Query plan the AI produced (run by Pandas):")
                st.json(ans.plan, expanded=False)

    if ans.table is not None and not ans.table.empty:
        with st.expander(f"📋 Rows behind this answer ({len(ans.table):,})"):
            st.dataframe(ans.table, hide_index=True, width="stretch")
            st.download_button("⬇️ Download CSV",
                               ans.table.to_csv(index=False).encode("utf-8"),
                               "answer_rows.csv", "text/csv", key=f"dl_{key}")

    if ans.sources:
        with st.expander(f"📎 Sources ({len(ans.sources)} passages)",
                         expanded=not ans.ok):
            source_cards(ans.sources)


def table_suggestions(engine):
    df, kinds = engine.df, engine.kinds
    cats = [c for c in df.columns if kinds[c] == "text"
            and 2 <= df[c][df[c] != ""].nunique() <= 25]
    nums = [c for c in df.columns if kinds[c] == "number"
            and df[c].nunique() > 5]
    out = ["How many rows are there in total?"]
    if cats:
        out.append(f"How many rows are there for each {cats[0]}?")
    if nums:
        out.append(f"Show the top 5 by {nums[0]}")
    if cats and nums:
        out.append(f"What is the average {nums[0]} per {cats[0]}?")
    elif nums:
        out.append(f"What is the average {nums[0]}?")
    return out[:4]


PDF_SUGGESTIONS = ["Summarize this document",
                   "What are the key findings?",
                   "What important numbers or dates are mentioned?",
                   "What are the conclusions or recommendations?"]


# =====================================================
# SESSION STATE
# =====================================================

ss = st.session_state
ss.setdefault("chats", {})
ss.setdefault("uploader_n", 0)
ss.setdefault("insights", {})
ss.setdefault("show_details", True)
ss.setdefault("natural", True)
ss.setdefault("llm_model", rc.LLM_MODEL)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">🧠</div><div>'
                '<div class="name">DocuMind</div>'
                '<div class="tag">Grounded answers from your files</div>'
                '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="side-title">Add files</div>',
                unsafe_allow_html=True)
    uploads = st.file_uploader(
        "PDF, Excel or CSV", type=FILE_TYPES, accept_multiple_files=True,
        key=f"uploader_{ss.uploader_n}", label_visibility="collapsed")
    if uploads:
        for f in uploads:
            save_upload(f)
        ss.uploader_n += 1
        ss.scope = str(KB_DIR / Path(uploads[-1].name).name)
        st.rerun()

    with st.expander("🔗 Link a live Excel/CSV on this PC"):
        st.caption("Linked files are read in place, so edits you save in "
                   "Excel show up in your answers automatically.")
        link = st.text_input("Full path", placeholder=r"C:\data\students.xlsx",
                             label_visibility="collapsed")
        if st.button("Link file", width="stretch"):
            p = Path(link.strip().strip('"'))
            if p.exists() and p.suffix.lower().lstrip(".") in FILE_TYPES:
                save_links(load_links() + [str(p.resolve())])
                ss.scope = str(p.resolve())
                st.rerun()
            else:
                st.error("File not found or unsupported type.")

    sources = list_sources()
    st.markdown(f'<div class="side-title">Your files ({len(sources)})</div>',
                unsafe_allow_html=True)
    if not sources:
        st.caption("No files yet. Upload a PDF, Excel or CSV above.")
    for i, src in enumerate(sources):
        c1, c2 = st.columns([0.84, 0.16], vertical_alignment="center")
        c1.markdown(
            f'<div class="file">{src["icon"]}<span class="nm" '
            f'title="{html.escape(src["path"])}">{html.escape(src["name"])}'
            f'</span>{"<span class=lk>LIVE</span>" if src["linked"] else ""}'
            f'</div>', unsafe_allow_html=True)
        if c2.button("✕", key=f"rm_{i}_{src['name']}",
                     help="Remove this file and its index"):
            remove_source(src)
            st.rerun()

    st.markdown('<div class="side-title">Settings</div>',
                unsafe_allow_html=True)
    with st.expander("⚙️ Model & answers"):
        ss.llm_model = st.text_input("Ollama model", value=ss.llm_model)
        ss.natural = st.toggle("Friendly wording for calculations",
                               value=ss.natural,
                               help="Numbers are always checked against the "
                                    "exact result.")
        ss.show_details = st.toggle("Show calculation details",
                                    value=ss.show_details)
    ok, msg = ollama_status(ss.llm_model)
    st.markdown(f'<div class="status {"ok" if ok else "bad"}"><span class="dot">'
                f'</span>{html.escape(msg)}</div>', unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="hero">
  <span class="chip-light">🔒 100% local · Ollama + ChromaDB</span>
  <h1>Ask your documents anything</h1>
  <p>Chat with PDFs, Excel sheets and CSV files. Answers come only from your
  files, with page references and exact calculations you can check.</p>
</div>""", unsafe_allow_html=True)

if not sources:
    st.markdown("""
<div class="features">
  <div class="feature"><div class="ic">📕</div><h4>PDF Q&amp;A</h4>
    <p>Hybrid search + reranking finds the right passages; every answer cites
    its page numbers.</p></div>
  <div class="feature"><div class="ic">🧮</div><h4>Exact spreadsheet maths</h4>
    <p>Counts, totals, averages and top-N are computed by Pandas on the whole
    sheet, never guessed.</p></div>
  <div class="feature"><div class="ic">✨</div><h4>Instant insights</h4>
    <p>One-click summaries and key findings for PDFs, and an automatic profile
    of your data.</p></div>
</div>""", unsafe_allow_html=True)
    st.info("👈 Upload a PDF, Excel or CSV file in the sidebar to begin.")
    st.stop()


# =====================================================
# PICK WHAT TO CHAT WITH
# =====================================================

pdfs = [s for s in sources if s["kind"] == "pdf"]
options = [s["path"] for s in sources] + ([ALL_PDFS] if len(pdfs) > 1 else [])
by_path = {s["path"]: s for s in sources}

if ss.get("scope") not in options:
    ss.scope = options[0]


def scope_label(opt):
    if opt == ALL_PDFS:
        return f"📚 All PDFs ({len(pdfs)})"
    return f"{by_path[opt]['icon']} {by_path[opt]['name']}"


chosen = st.pills("Chat with", options, format_func=scope_label,
                  key="scope", selection_mode="single")
scope = chosen or options[0]

# ---------- build / refresh the engine(s) ----------
try:
    if scope == ALL_PDFS:
        engines = [get_pdf_engine(s["path"]) for s in pdfs]
        for e, s in zip(engines, pdfs):
            refresh_engine(e, s["name"])
        mode = "pdf"
    elif by_path[scope]["kind"] == "pdf":
        engines = [get_pdf_engine(scope)]
        refresh_engine(engines[0], by_path[scope]["name"])
        mode = "pdf"
    else:
        signature = rc.file_signature(scope)
        sheets = sheets_of(scope, signature)
        sheet_key, id_key = f"sheet::{scope}", f"id::{scope}"
        with st.popover("⚙️ Sheet & ID column"):
            sheet = st.selectbox("Sheet", sheets, key=sheet_key)
            cols = columns_of(scope, sheet, signature)
            id_choice = st.selectbox(
                "Unique ID column (optional)", ["(row number)"] + cols,
                key=id_key,
                help="e.g. Roll No or Employee ID. Inserting or sorting "
                     "rows then won't trigger re-indexing.")
        key_column = None if id_choice == "(row number)" else id_choice
        engine = get_table_engine(scope, sheet, key_column)
        refresh_engine(engine, by_path[scope]["name"])
        engines = [engine]
        mode = "table"
except Exception as exc:
    st.error(f"Could not read this file: {exc}", icon="🚫")
    st.stop()

for e in engines:
    for w in e.warnings:
        st.warning(w, icon="⚠️")

# ---------- stat cards ----------
if mode == "pdf":
    pages = sum(len(e.pages) for e in engines)
    words = sum(e.word_count for e in engines)
    stat_cards([("Documents", len(engines)), ("Pages", f"{pages:,}"),
                ("Words", f"{words:,}"),
                ("Indexed passages", f"{sum(len(e.chunks) for e in engines):,}"),
                ("Reading time", f"~{max(1, round(words / 230))} min")])
else:
    df = engines[0].df
    stat_cards([("Rows", f"{len(df):,}"), ("Columns", len(df.columns)),
                ("Number columns",
                 sum(k == "number" for k in engines[0].kinds.values())),
                ("Sheet", engines[0].sheet),
                ("Indexed rows", f"{engines[0].last_sync.get('total', 0):,}")])

chat_key = scope if mode == "pdf" else f"{scope}::{engines[0].sheet}"
messages = ss.chats.setdefault(chat_key, [])


# =====================================================
# TABS
# =====================================================

tab_chat, tab_insights, tab_explore = st.tabs(
    ["💬  Chat", "✨  Insights", "🔍  Explore"])

prompt = st.chat_input("Ask a question about your file...")
if ss.get("pending_prompt"):
    prompt = prompt or ss.pop("pending_prompt")


def previous_question():
    return next((m["text"] for m in reversed(messages)
                 if m["role"] == "user"), None)


def answer_pdf(question):
    multi = len(engines) > 1
    previous = previous_question()

    if not multi and SUMMARY_RE.search(question):
        mode_ = "insights" if re.search(r"insight|finding|takeaway",
                                        question, re.I) else "summary"
        bar = st.progress(0.0, text="Reading the whole document...")
        stream = pe.summarize(engines[0], ss.llm_model, mode_,
                              progress=lambda f, t: bar.progress(f, text=t))
        badge("sum", "📝 Whole-document reading")
        text = st.write_stream(stream)
        bar.empty()
        return Answer(text, "summary"), True

    with st.status("🔎 Searching your document...", expanded=False) as status:
        sources_, msgs, search_q = pe.prepare_answer(
            engines, question, ss.llm_model, previous)
        if search_q != question:
            st.write(f"Searched for: _{search_q}_")
        status.update(label=(f"Found {len(sources_)} relevant passage(s)"
                             if msgs else "No passage matched strongly"),
                      state="complete")

    if msgs is None:
        return Answer(pe.NOT_FOUND + " Try rephrasing with words used in the "
                      "document. The closest passages are below.", "pdf",
                      ok=False, sources=sources_), False

    holder = st.empty()
    with holder.container():
        raw = st.write_stream(rc.llm_stream(ss.llm_model, msgs))
    final, notes, ok_ = pe.finalize(raw, sources_, multi)
    holder.empty()
    if not ok_:
        return Answer(pe.NOT_FOUND, "pdf", ok=False, sources=sources_), False
    return Answer(final, "pdf", sources=sources_, notes=notes), False


with tab_chat:
    if not messages and not prompt:
        st.markdown("##### 💡 Try asking")
        suggestions = PDF_SUGGESTIONS if mode == "pdf" \
            else table_suggestions(engines[0])
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if cols[i % 2].button(s, key=f"sg_{i}", width="stretch"):
                ss.pending_prompt = s
                st.rerun()

    for i, m in enumerate(messages):
        with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user"
                             else "🧠"):
            if m["role"] == "user":
                st.markdown(m["text"])
            else:
                render_answer(m["answer"], f"{chat_key}_{i}")

    if prompt:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        with st.chat_message("assistant", avatar="🧠"):
            try:
                if mode == "pdf":
                    ans, already_shown = answer_pdf(prompt)
                else:
                    with st.spinner("🧮 Working it out on the full sheet..."):
                        ans = engines[0].ask(prompt, ss.llm_model,
                                             previous_question(), ss.natural)
                    already_shown = False
            except rc.OllamaUnavailable as exc:
                ans, already_shown = Answer(str(exc), "error", ok=False), False
            if not already_shown:
                render_answer(ans, f"{chat_key}_{len(messages) + 1}")
        messages.append({"role": "user", "text": prompt})
        messages.append({"role": "assistant", "answer": ans})

    if messages:
        if st.button("🧹 Clear this chat"):
            messages.clear()
            st.rerun()


# =====================================================
# INSIGHTS
# =====================================================

with tab_insights:
    if mode == "pdf":
        if len(engines) > 1:
            st.info("Pick a single PDF above to generate its summary.")
        else:
            eng = engines[0]
            st.markdown("##### One-click reading of the whole document")
            st.caption("Reads every page (not just a few passages), so it "
                       "can take a minute on long PDFs.")
            c1, c2 = st.columns(2)
            run = None
            if c1.button("📝 Summarize document", width="stretch"):
                run = "summary"
            if c2.button("💡 Key insights & numbers", width="stretch"):
                run = "insights"
            for mode_ in ("summary", "insights"):
                cache_key = (eng.file_path, mode_, eng._signature)
                if run == mode_:
                    bar = st.progress(0.0, text="Starting...")
                    try:
                        text = st.write_stream(pe.summarize(
                            eng, ss.llm_model, mode_,
                            progress=lambda f, t: bar.progress(f, text=t)))
                        ss.insights[cache_key] = text
                    except rc.OllamaUnavailable as exc:
                        st.error(str(exc), icon="⚠️")
                    bar.empty()
                elif cache_key in ss.insights:
                    with st.container(border=True):
                        badge("sum", "📝 Summary" if mode_ == "summary"
                              else "💡 Key insights")
                        st.markdown(ss.insights[cache_key])
                        st.download_button(
                            "⬇️ Download", ss.insights[cache_key],
                            f"{Path(eng.name).stem}_{mode_}.md",
                            key=f"dl_ins_{mode_}")
    else:
        eng = engines[0]
        df = eng.df
        nums = [c for c in df.columns if eng.kinds[c] == "number"]
        cats = [c for c in df.columns if eng.kinds[c] == "text"
                and 2 <= df[c][df[c] != ""].nunique() <= 25]

        blanks = (df == "").sum()
        stat_cards([("Complete rows", f"{int((df != '').all(axis=1).sum()):,}"),
                    ("Blank cells", f"{int(blanks.sum()):,}"),
                    ("Duplicate rows", f"{int(df.duplicated().sum()):,}"),
                    ("Category columns", len(cats))])

        if nums:
            st.markdown("##### 🔢 Number columns")
            rows = []
            for c in nums:
                v = te.to_number(df[c]).dropna()
                rows.append({"Column": c, "Values": len(v),
                             "Total": round(v.sum(), 2),
                             "Average": round(v.mean(), 2),
                             "Min": v.min(), "Max": v.max()})
            st.dataframe(rows, hide_index=True, width="stretch")

        if cats:
            st.markdown("##### 📊 How values are distributed")
            cols = st.columns(2)
            for i, c in enumerate(cats[:6]):
                with cols[i % 2], st.container(border=True):
                    st.markdown(f"**{c}**")
                    counts = (df[c].replace("", "(blank)").str.strip()
                              .value_counts().head(10))
                    st.bar_chart(counts, color="#6D5DFC", horizontal=True,
                                 height=240)

        if blanks.sum():
            st.markdown("##### 🧹 Data quality")
            q = blanks[blanks > 0].sort_values(ascending=False)
            st.dataframe(
                [{"Column": c, "Blank cells": int(n),
                  "% blank": f"{100 * n / len(df):.1f}%"} for c, n in q.items()],
                hide_index=True, width="stretch")


# =====================================================
# EXPLORE
# =====================================================

with tab_explore:
    if mode == "pdf":
        query = st.text_input("🔎 Search inside the document",
                              placeholder="Type keywords, e.g. revenue 2024")
        if query:
            hits = []
            for e in engines:
                for c in e.search(query, k=8):
                    hits.append(c)
            hits.sort(key=lambda c: c["rank"])
            if hits:
                source_cards(hits[:8])
            else:
                st.caption("No matches.")
        eng = engines[0] if len(engines) == 1 else None
        if eng and eng.pages:
            st.markdown("##### 📄 Read a page")
            page_no = st.number_input("Page", 1, len(eng.pages), 1)
            st.markdown(f'<div class="page-box">'
                        f'{html.escape(eng.pages[page_no - 1]["text"] or "(no text on this page)")}'
                        f'</div>', unsafe_allow_html=True)
    else:
        eng = engines[0]
        query = st.text_input("🔎 Filter rows",
                              placeholder="Type any value, e.g. CSE")
        view = eng.with_row_ids(eng.df)
        if query:
            mask = eng.df.apply(lambda col: col.str.contains(
                query, case=False, regex=False)).any(axis=1)
            view = view[mask.values]
            st.caption(f"{len(view):,} matching rows")
        st.dataframe(view, hide_index=True, width="stretch", height=480)
