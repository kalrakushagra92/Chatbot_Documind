"""
Excel / CSV question answering without guessed numbers.

Every question first goes to a "query planner": the LLM only translates
the question into a small JSON plan (filters + operation). Pandas runs
the plan on the WHOLE sheet, so counts, totals, averages and lookups are
exact. Safeguards against wrong answers:

  * The planner is told which real cell values the question mentions.
  * Filter values are checked against the column; near-misses are
    corrected ("cse" -> "CSE") and anything not found is reported, not
    silently counted as 0.
  * The friendly wording is checked: if it contains a number that is
    not in the computed result, the exact result is shown instead.
  * Only if the plan fails, a row-retrieval answer is used, and its row
    citations are verified.

ChromaDB is synced incrementally (content hashes), never rebuilt.
"""

from __future__ import annotations

import difflib
import json
import re
import threading
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

import rag_core as rc

TOP_K_ROWS = 8
MAX_ROWS_IN_TEXT = 15
MAX_SAMPLE_VALUES = 12
NOT_FOUND = "I could not find this information in the spreadsheet."


# =====================================================
# READING FILES
# =====================================================

def list_sheets(path):
    if Path(path).suffix.lower() == ".csv":
        return ["CSV"]
    return pd.ExcelFile(path).sheet_names


def read_table(path, sheet="CSV", nrows=None):
    """Read as text so IDs/phone/PIN codes keep their leading zeros."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         encoding_errors="replace", nrows=nrows,
                         sep=None, engine="python")
    else:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str,
                           keep_default_na=False, nrows=nrows)

    df.columns = [str(c).strip() for c in df.columns]
    empty_unnamed = [c for c in df.columns if c.startswith("Unnamed:")
                     and (nrows == 0 or not df[c].astype(str).str.strip().any())]
    df = df.drop(columns=empty_unnamed)
    df = df.apply(lambda col: col.astype(str).str.strip())
    df = df.replace(r"^(\d{4}-\d{2}-\d{2}) 00:00:00$", r"\1", regex=True)
    df = df.replace(r"^(-?\d+)\.0$", r"\1", regex=True)      # 80.0 -> 80
    return df


def read_columns(path, sheet="CSV"):
    return list(read_table(path, sheet, nrows=0).columns)


# =====================================================
# VALUE HELPERS
# =====================================================

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6,
         "vii": 7, "viii": 8, "ix": 9, "x": 10, "xi": 11, "xii": 12}
DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")


def to_number(series):
    """'1,250' -> 1250, 'Sem 3' -> 3, 'III' -> 3; otherwise NaN."""
    s = series.astype(str).str.strip().str.replace(",", "", regex=False)
    s = s.str.replace(r"^[₹$€£]\s*|\s*%$", "", regex=True)
    result = pd.to_numeric(s, errors="coerce").fillna(pd.to_numeric(
        s.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce"))

    roman = (s.str.casefold()
             .str.extract(r"^(?:sem(?:ester)?\.?\s*)?"
                          r"(xii|xi|x|ix|viii|vii|vi|v|iv|iii|ii|i)$")[0]
             .map(ROMAN))
    non_blank = s != ""
    if non_blank.any() and roman.notna().any():
        numeric_share = (result.notna() | roman.notna())[non_blank].mean()
        roman_share = roman.notna()[non_blank].mean()
        long_roman = s.str.len()[roman.notna()].gt(1).any()
        if numeric_share >= 0.6 and (roman_share >= 0.25 or long_roman):
            result = result.fillna(roman)
    return result


def to_date(series):
    s = series.astype(str).str.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso = pd.to_datetime(s, errors="coerce", format="ISO8601")
        rest = pd.to_datetime(s.where(iso.isna()), errors="coerce",
                              dayfirst=True, format="mixed")
    return iso.fillna(rest)


def parse_number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = re.sub(r"[,₹$€£%]", "", str(value)).strip()
    try:
        return float(text)
    except ValueError:
        if DATE_RE.match(text):
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        if m:
            return float(m.group())
        r = ROMAN.get(text.casefold())
        return float(r) if r else None


def parse_date(value):
    text = str(value).strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if re.fullmatch(r"\d{4}", text):
            return pd.Timestamp(year=int(text), month=1, day=1)
        try:
            return pd.to_datetime(text, format="ISO8601")
        except (ValueError, TypeError):
            return pd.to_datetime(text, dayfirst=True)


def column_kind(series):
    values = series[series != ""]
    if values.empty:
        return "text"
    if values.str.match(DATE_RE).mean() >= 0.6:
        return "date"
    if to_number(values).notna().mean() >= 0.7:
        return "number"
    return "text"


def fmt(x):
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    return f"{int(x):,}" if x.is_integer() else f"{x:,.2f}"


def numbers_in(text):
    return {n.replace(",", "").rstrip(".").rstrip("0").rstrip(".")
            if "." in n else n.replace(",", "")
            for n in re.findall(r"\d[\d,]*\.?\d*", text)}


# =====================================================
# PROMPTS
# =====================================================

PLAN_PROMPT = """You turn a question about a spreadsheet into a JSON query
plan. You do NOT answer it; Pandas runs your plan on the full sheet.

COLUMNS (name, type, distinct count, example values):
{schema}

Return ONLY this JSON:
{{"filters": [{{"column": "...", "op": "...", "value": ...}}],
  "logic": "and",
  "operation": "...",
  "target_column": null,
  "columns": [],
  "group_by": null,
  "n": null}}

ops: eq, ne, contains, not_contains, in (value = list), gt, gte, lt, lte,
     between (value = [low, high]), odd, even, is_empty, not_empty
operations:
  lookup        values of "columns" for the matching row(s)
  list          show matching rows ("columns" optional)
  count         number of matching rows
  percent       matching rows as % of all rows
  count_distinct, sum, mean, median, min, max   on target_column
  group_count   rows per group_by value
  group_sum, group_mean   target_column per group_by value
  top_n, bottom_n   n rows with highest/lowest target_column
  unanswerable  the sheet has no column that can answer this

Rules:
- Use exact column names from the list. Prefer values exactly as they
  appear in the examples or in VALUES MENTIONED.
- A person/item name in the question is usually an eq filter on the
  name/ID column, with operation lookup.
- Dates as YYYY-MM-DD. "odd semester" = op odd on the semester column.
- No condition -> "filters": [].

Examples:
Q: Give me the count of the students having odd semester?
{{"filters": [{{"column": "Semester", "op": "odd", "value": null}}], "logic": "and", "operation": "count", "target_column": null, "columns": [], "group_by": null, "n": null}}
Q: What are Meena's marks and department?
{{"filters": [{{"column": "Name", "op": "eq", "value": "Meena"}}], "logic": "and", "operation": "lookup", "target_column": null, "columns": ["Marks", "Department"], "group_by": null, "n": null}}
Q: Average marks per department
{{"filters": [], "logic": "and", "operation": "group_mean", "target_column": "Marks", "columns": [], "group_by": "Department", "n": null}}
Q: Top 3 students by marks in semester 5
{{"filters": [{{"column": "Semester", "op": "eq", "value": "5"}}], "logic": "and", "operation": "top_n", "target_column": "Marks", "columns": [], "group_by": null, "n": 3}}
"""

PHRASE_PROMPT = """Rewrite the COMPUTED RESULT as a short, friendly answer to
the question (1-3 sentences, or a few bullets for several items).
Use ONLY facts in the computed result. Never add, round, change or invent
any number, name or value. Do not mention filters, row IDs or Pandas."""

ROWS_PROMPT = """Answer using ONLY the spreadsheet rows given. Copy values
exactly. Cite the row ID(s) you used, like (row_12). If the rows do not
answer the question, reply exactly: {not_found}"""


# =====================================================
# ENGINE
# =====================================================

class TableRAG:

    def __init__(self, file_path, sheet="CSV", key_column=None,
                 embedding_model=rc.EMBEDDING_MODEL, db_path=rc.DB_PATH,
                 embedding_function=None):
        self.file_path = str(Path(file_path).resolve())
        self.name = Path(file_path).name
        self.sheet = sheet
        self.key_column = key_column or None
        self.embedding_model = embedding_model
        self.db_path = db_path
        self._ef = embedding_function
        self._lock = threading.RLock()
        self._signature = None
        self.collection = None
        self.df = None
        self.warnings = []
        self.last_sync = {}

    # ---------------- load + sync ----------------

    def refresh(self, force=False):
        with self._lock:
            signature = rc.file_signature(self.file_path)
            if not force and signature == self._signature:
                return None
            self._load()
            if self.collection is None:
                self.collection = rc.get_collection(
                    rc.collection_name("tab", self.file_path, self.sheet,
                                       self.embedding_model),
                    self.file_path, self.embedding_model, self.db_path, self._ef)
            stats = rc.sync_collection(
                self.collection, list(self.doc_ids), list(self.documents),
                [{"row_id": r, "file": self.name} for r in self.row_ids])
            self._signature = signature
            self.last_sync = stats
            return stats

    def _load(self):
        df = read_table(self.file_path, self.sheet)
        df = df[df.apply(lambda r: any(v for v in r), axis=1)]
        if df.empty:
            raise ValueError("The sheet has no data rows.")
        row_ids = [f"row_{i + 2}" for i in df.index]       # header = row 1
        df = df.reset_index(drop=True)

        documents = ["\n".join(f"{c}: {row[c]}" for c in df.columns if row[c])
                     for _, row in df.iterrows()]

        self.warnings = []
        doc_ids = row_ids
        if self.key_column:
            keys = df.get(self.key_column)
            if keys is not None and (keys != "").all() and keys.is_unique:
                doc_ids = [f"id_{k}" for k in keys]
            else:
                self.warnings.append(f"'{self.key_column}' has blank or "
                                     "duplicate values; using row numbers.")

        value_rows, value_cols = defaultdict(set), defaultdict(set)
        uniques = {}
        for col in df.columns:
            low = df[col].str.casefold()
            uniques[col] = dict(zip(low, df[col]))
            for i, v in enumerate(low):
                if 2 <= len(v) <= 80:
                    value_rows[v].add(i)
                    value_cols[v].add(col)

        self.df = df
        self.row_ids = row_ids
        self.row_pos = {r: i for i, r in enumerate(row_ids)}
        self.documents = documents
        self.doc_ids = doc_ids
        self.value_rows = dict(value_rows)
        self.value_cols = dict(value_cols)
        self.uniques = uniques
        self.kinds = {c: column_kind(df[c]) for c in df.columns}
        self.schema = "\n".join(
            f'- "{c}" ({self.kinds[c]}, {df[c][df[c] != ""].nunique()} '
            f"distinct): " + ", ".join(
                df[c][df[c] != ""].unique()[:MAX_SAMPLE_VALUES])
            for c in df.columns)

    # ---------------- helpers ----------------

    def with_row_ids(self, sub, columns=None):
        out = (sub[columns] if columns else sub).copy()
        out.insert(0, "Row", [self.row_ids[i] for i in sub.index])
        return out

    def resolve_column(self, name):
        if name is None or str(name).strip().lower() in ("", "null", "none"):
            return None
        name = str(name).strip()
        if name in self.df.columns:
            return name
        lookup = {c.casefold(): c for c in self.df.columns}
        if name.casefold() in lookup:
            return lookup[name.casefold()]
        close = difflib.get_close_matches(name.casefold(), list(lookup),
                                          n=1, cutoff=0.6)
        if close:
            return lookup[close[0]]
        raise ValueError(f"Column '{name}' does not exist. "
                         f"Columns: {list(self.df.columns)}")

    def mentioned_values(self, question):
        """Real cell values that appear in the question (planner hints)."""
        q = question.casefold()
        hits = []
        for value, cols in self.value_cols.items():
            if len(value) < 3 or value not in q:
                continue
            if re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", q):
                col = sorted(cols)[0]
                hits.append((len(value), f'"{self.uniques[col][value]}" '
                                         f"(column {', '.join(sorted(cols))})"))
        hits.sort(reverse=True)
        return [h for _, h in hits[:12]]

    def _ground(self, col, value, notes):
        """Map a filter value to a value that really exists in the column."""
        text = str(value).strip()
        low = text.casefold()
        uniques = self.uniques[col]
        if low in uniques or self.kinds[col] != "text":
            return text
        contains = [u for u in uniques if u and
                    re.search(r"(?<!\w)" + re.escape(low) + r"(?!\w)", u)]
        if len(contains) == 1:
            notes.append(f"Matched '{text}' to '{uniques[contains[0]]}' "
                         f"in {col}.")
            return uniques[contains[0]]
        close = difflib.get_close_matches(low, [u for u in uniques if u],
                                          n=1, cutoff=0.82)
        if close:
            notes.append(f"Matched '{text}' to '{uniques[close[0]]}' in {col}.")
            return uniques[close[0]]
        available = [v for v in dict.fromkeys(self.df[col]) if v][:12]
        notes.append(f"No cell in '{col}' equals '{text}'. Values in this "
                     f"column include: {', '.join(available)}.")
        return text

    # ---------------- filters ----------------

    def _filter(self, flt, notes):
        df = self.df
        if not isinstance(flt, dict):
            raise ValueError(f"Bad filter: {flt!r}")
        col = self.resolve_column(flt.get("column"))
        if not col:
            raise ValueError("A filter has no column.")
        op = str(flt.get("op") or "eq").lower().strip()
        op = {"=": "eq", "==": "eq", "!=": "ne", ">": "gt", ">=": "gte",
              "<": "lt", "<=": "lte"}.get(op, op)
        value = flt.get("value")
        text, low = df[col], df[col].str.casefold()

        if op in ("eq", "ne"):
            value = self._ground(col, value, notes)
            mask = low == str(value).strip().casefold()
            num = parse_number(value)
            if num is not None and self.kinds[col] == "number":
                mask |= to_number(text) == num
            return ~mask if op == "ne" else mask

        if op in ("contains", "not_contains"):
            mask = low.str.contains(str(value).casefold(), regex=False)
            if op == "contains" and not mask.any():
                notes.append(f"No cell in '{col}' contains '{value}'.")
            return mask if op == "contains" else ~mask

        if op in ("in", "not_in"):
            values = value if isinstance(value, list) else [value]
            mask = pd.Series(False, index=df.index)
            for v in values:
                mask |= self._filter({"column": col, "op": "eq", "value": v},
                                     notes)
            return mask if op == "in" else ~mask

        if op in ("odd", "even"):
            nums = to_number(text)
            whole = nums.notna() & (nums % 1 == 0)
            return whole & (nums % 2 == (1 if op == "odd" else 0))

        if op == "is_empty":
            return text == ""
        if op == "not_empty":
            return text != ""

        if op in ("gt", "gte", "lt", "lte", "between"):
            if op == "between":
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError("'between' needs [low, high].")
                lo, hi = value
            else:
                lo = hi = value
            if self.kinds[col] == "date" or DATE_RE.match(str(lo).strip()):
                series, a, b = to_date(text), parse_date(lo), parse_date(hi)
            else:
                series, a, b = to_number(text), parse_number(lo), parse_number(hi)
                if a is None or b is None:
                    raise ValueError(f"'{value}' is not a number.")
            return {"gt": series > a, "gte": series >= a, "lt": series < a,
                    "lte": series <= a,
                    "between": (series >= a) & (series <= b)}[op]

        raise ValueError(f"Unknown operator '{op}'.")

    @staticmethod
    def describe(plan):
        filters = [f for f in (plan.get("filters") or [])
                   if isinstance(f, dict)]
        if not filters:
            return "all rows"
        word = f" {str(plan.get('logic') or 'and').upper()} "
        return word.join(
            f"{f.get('column')} {f.get('op')}"
            + ("" if f.get("value") is None else f" {f.get('value')!r}")
            for f in filters)

    # ---------------- execute a plan ----------------

    def execute(self, plan):
        """Run the plan with Pandas. Returns (exact_text, table, notes)."""
        df, notes = self.df, []
        filters = plan.get("filters") or []
        if isinstance(filters, dict):
            filters = [filters]
        logic = str(plan.get("logic") or "and").lower()

        sub = df
        if filters:
            masks = [self._filter(f, notes) for f in filters]
            mask = masks[0]
            for m in masks[1:]:
                mask = (mask | m) if logic == "or" else (mask & m)
            sub = df[mask.fillna(False).astype(bool)]

        op = str(plan.get("operation") or "count").lower().strip()
        op = {"average": "mean", "avg": "mean", "top": "top_n",
              "bottom": "bottom_n"}.get(op, op)
        target = self.resolve_column(plan.get("target_column"))
        group_by = self.resolve_column(plan.get("group_by"))
        columns = [c for c in (self.resolve_column(c)
                               for c in (plan.get("columns") or [])) if c]
        try:
            n = max(1, int(plan.get("n") or 5))
        except (TypeError, ValueError):
            n = 5
        scope = f"Rows matching ({self.describe(plan)}): {len(sub):,} of {len(df):,}"

        if op == "unanswerable":
            raise LookupError("The planner says no column answers this.")

        if op in ("lookup", "list"):
            cols = columns or ([target] if target else None)
            if sub.empty:
                return f"No matching rows. {scope}", None, notes
            if op == "lookup" and cols and len(sub) <= MAX_ROWS_IN_TEXT:
                lines = [f"{self.row_ids[i]}: " + "; ".join(
                    f"{c} = {sub.at[i, c] or '(blank)'}" for c in cols)
                    for i in sub.index]
                return ("\n".join(lines) + f"\n{scope}",
                        self.with_row_ids(sub), notes)
            shown = sub.head(MAX_ROWS_IN_TEXT)
            lines = [f"{self.row_ids[i]}: " + "; ".join(
                f"{c} = {shown.at[i, c]}" for c in (cols or df.columns))
                for i in shown.index]
            more = f"\n... and {len(sub) - len(shown):,} more rows" \
                if len(sub) > len(shown) else ""
            return (f"{len(sub):,} matching rows:\n" + "\n".join(lines)
                    + more + f"\n{scope}", self.with_row_ids(sub, cols), notes)

        if op == "count":
            return f"Count = {len(sub):,}\n{scope}", self.with_row_ids(sub), notes

        if op == "percent":
            pct = 100 * len(sub) / len(df)
            return (f"Percentage = {pct:.2f}% ({len(sub):,} of {len(df):,} rows)"
                    f"\n{scope}", self.with_row_ids(sub), notes)

        if op == "count_distinct":
            if not target:
                raise ValueError("count_distinct needs target_column.")
            vals = sub[target][sub[target] != ""]
            table = (vals.groupby(vals.str.casefold()).agg(["first", "size"])
                     .rename(columns={"first": target, "size": "Rows"})
                     .sort_values("Rows", ascending=False)
                     .reset_index(drop=True))
            return (f"Distinct values of {target} = {len(table):,}\n{scope}",
                    table, notes)

        if op in ("sum", "mean", "median", "min", "max"):
            if not target:
                raise ValueError(f"{op} needs target_column.")
            nums = to_number(sub[target]).dropna()
            if nums.empty:
                return (f"No numeric values in {target} for these rows.\n{scope}",
                        None, notes)
            value = getattr(nums, op)()
            skipped = len(sub) - len(nums)
            if skipped:
                notes.append(f"Skipped {skipped} blank/non-numeric cells "
                             f"in {target}.")
            label = {"mean": "Average", "sum": "Total", "median": "Median",
                     "min": "Minimum", "max": "Maximum"}[op]
            text = f"{label} {target} = {fmt(value)} (from {len(nums):,} values)"
            if op in ("min", "max"):
                hit = sub.loc[nums[nums == value].index]
                text += "\n" + "\n".join(
                    f"{self.row_ids[i]}: " + "; ".join(
                        f"{c} = {hit.at[i, c]}" for c in df.columns)
                    for i in hit.index[:MAX_ROWS_IN_TEXT])
                return f"{text}\n{scope}", self.with_row_ids(hit), notes
            return (f"{text}\n{scope}", self.with_row_ids(sub.loc[nums.index]),
                    notes)

        if op in ("group_count", "group_sum", "group_mean"):
            if not group_by:
                raise ValueError(f"{op} needs group_by.")
            keys = sub[group_by].str.casefold()
            labels = sub[group_by].groupby(keys).first()
            if op == "group_count":
                result, metric = sub.groupby(keys).size(), "Rows"
            else:
                if not target:
                    raise ValueError(f"{op} needs target_column.")
                grouped = to_number(sub[target]).groupby(keys)
                if op == "group_sum":
                    result, metric = grouped.sum(), f"Total {target}"
                else:
                    result, metric = grouped.mean().round(2), f"Average {target}"
            result = result.sort_values(ascending=False)
            table = pd.DataFrame({group_by: [labels[k] or "(blank)"
                                             for k in result.index],
                                  metric: result.values})
            lines = [f"{g}: {fmt(v)}" for g, v in
                     zip(table[group_by], table[metric])]
            return (f"{metric} by {group_by}:\n" + "\n".join(lines)
                    + f"\n{scope}", table, notes)

        if op in ("top_n", "bottom_n"):
            if not target:
                raise ValueError(f"{op} needs target_column.")
            nums = to_number(sub[target]).dropna()
            order = nums.sort_values(ascending=(op == "bottom_n"))
            hit = sub.loc[order.index[:n]]
            word = "Lowest" if op == "bottom_n" else "Highest"
            lines = [f"{k}. {self.row_ids[i]}: " + "; ".join(
                f"{c} = {hit.at[i, c]}" for c in df.columns)
                for k, i in enumerate(hit.index, 1)]
            return (f"{word} {len(hit)} by {target}:\n" + "\n".join(lines)
                    + f"\n{scope}", self.with_row_ids(hit), notes)

        raise ValueError(f"Unknown operation '{op}'.")

    # ---------------- LLM steps ----------------

    def make_plan(self, question, llm_model, previous=None, error=None):
        hints = self.mentioned_values(question)
        user = f"Q: {question}"
        if hints:
            user = "VALUES MENTIONED (exist in the sheet): " + \
                   "; ".join(hints) + "\n" + user
        messages = [{"role": "system",
                     "content": PLAN_PROMPT.format(schema=self.schema)}]
        if previous:
            messages.append({"role": "user", "content":
                             "Context only, for follow-up questions. "
                             f"Previous question: {previous}"})
        messages.append({"role": "user", "content": user})
        if error:
            messages.append({"role": "user", "content":
                             f"Your previous plan failed: {error}. "
                             "Return a corrected JSON plan."})
        plan = rc.parse_json(rc.llm_chat(llm_model, messages, json_mode=True))
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a JSON object.")
        return plan

    def phrase(self, question, exact, llm_model):
        """Friendly wording, rejected if it introduces any new number."""
        if len(exact) > 900:
            return None
        try:
            text = rc.llm_chat(llm_model, [
                {"role": "system", "content": PHRASE_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\n"
                                            f"COMPUTED RESULT:\n{exact}"},
            ], num_predict=250)
        except Exception:
            return None
        if not text or not numbers_in(text) <= numbers_in(exact):
            return None
        return text

    def answer_with_rows(self, question, llm_model):
        """Fallback: retrieve rows, answer from them, verify citations."""
        q = question.casefold()
        exact = []
        for value, rows in self.value_rows.items():
            if len(value) >= 3 and len(rows) <= 25 and value in q and \
                    re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", q):
                exact.extend(self.doc_ids[i] for i in rows)
        vector = rc.vector_search(self.collection, question,
                                  self.embedding_model, 20, self._ef)
        id_to_pos = {d: i for i, d in enumerate(self.doc_ids)}
        ranked = [id_to_pos[d] for d in rc.rrf_merge(exact, vector)
                  if d in id_to_pos][:TOP_K_ROWS]
        if not ranked:
            return Answer(NOT_FOUND, "rows", ok=False)

        context = "\n\n".join(f"ROW ID: {self.row_ids[i]}\n{self.documents[i]}"
                              for i in ranked)
        text = rc.llm_chat(llm_model, [
            {"role": "system", "content": ROWS_PROMPT.format(not_found=NOT_FOUND)},
            {"role": "user", "content": f"ROWS:\n\n{context}\n\nQUESTION: {question}"},
        ])
        valid = {self.row_ids[i] for i in ranked}
        cited = set(re.findall(r"\brow_\d+\b", text))
        if not text or text.startswith(NOT_FOUND[:25]) or not cited or \
                not cited <= valid:
            return Answer(NOT_FOUND, "rows", ok=False)
        table = self.with_row_ids(self.df.loc[sorted(self.row_pos[r]
                                                     for r in cited)])
        return Answer(text, "rows", table=table,
                      notes=["Answered from matching rows (not a calculation)."])

    def ask(self, question, llm_model, previous=None, natural=True):
        question = (question or "").strip()
        if not question:
            return Answer("Please type a question.", "error", ok=False)
        try:
            self.refresh()
            error, plan = None, None
            for _ in range(2):                    # one self-correction
                try:
                    plan = self.make_plan(question, llm_model, previous, error)
                    exact, table, notes = self.execute(plan)
                    break
                except rc.OllamaUnavailable:
                    raise
                except LookupError:
                    plan = None
                    break
                except Exception as exc:
                    error, plan = str(exc), None
            if plan is None:
                fallback = self.answer_with_rows(question, llm_model)
                if not fallback.ok and error:
                    fallback.notes.append(f"Could not build a calculation: {error}")
                return fallback

            missing = [n for n in notes if n.startswith("No cell")]
            if missing and (table is None or table.empty
                            or exact.startswith("Count = 0")):
                return Answer("I couldn't find that value in the sheet. "
                              + " ".join(missing), "pandas", ok=False,
                              plan=plan, exact=exact)

            text = (self.phrase(question, exact, llm_model) if natural else None)
            return Answer(text or exact.split("\nRows matching")[0], "pandas",
                          table=table, plan=plan, notes=notes, exact=exact)
        except rc.OllamaUnavailable as exc:
            return Answer(str(exc), "error", ok=False)


Answer = rc.Answer
