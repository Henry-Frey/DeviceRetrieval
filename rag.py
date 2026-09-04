"""On-device RAG testbed -- phase 0 spike.

One file, hardcoded config. All seven stages are implemented, and stage 7 is a
human loop rather than an API call:
  stage 1  build_index / chunk_document
  stage 2  rewrite_query
  stage 3  retrieve
  stage 4  rerank
  stage 5  build_context
  stage 6  assemble_prompt
  stage 7  deliver_prompt / run_generation_loop / assemble_judge_prompt

Retrieval is local and generation is manual: nothing in this file may ever call
a generation API.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

# Bump by hand whenever a change to chunking, embedding composition or the
# on-disk layout changes what an index *contains* without changing any config
# key. Without this a chunker bugfix silently reuses a stale index.
INDEX_FORMAT_VERSION = 1


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
# Flat declarative data, no logic. Attribute `chunk_size` is the flat key
# `chunk.size` when this moves to YAML in phase 2.

@dataclass(frozen=True)
class Config:
    corpus_dir: str = "corpus"
    corpus_globs: tuple[str, ...] = ("**/*.md", "**/*.txt")

    chunk_strategy: str = "recursive"
    # tokens, counted by the embedder's tokenizer. 510 = bge's 512-token limit
    # minus its two special tokens, so a full-size chunk is not truncated at
    # embed time. Raise this only alongside a model with a longer context.
    chunk_size: int = 510
    chunk_overlap: int = 51          # tokens, 10%
    chunk_separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ")
    chunk_header_mode: str = "none"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    # TODO pin to a 40-hex commit sha before running any ablation; `main` moves.
    embed_revision: str = "main"
    embed_dim: int = 384
    embed_prefix_scheme: str = "bge"
    embed_batch_size: int = 32
    embed_normalize: bool = True
    device: str = "cpu"

    rewrite_mode: str = "off"

    retrieve_mode: str = "dense"
    retrieve_top_k: int = 20

    rerank_mode: str = "off"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_revision: str = "main"
    rerank_query: str = "original"   # original | rewrite0 | each
    rerank_depth: int = 100          # how many candidates get scored, not how many survive
    rerank_batch_size: int = 32

    context_top_n: int = 8
    context_budget_tokens: int = 6000
    context_neighbours: int = 0          # +/- chunks by ordinal, same document
    context_dedup: str = "exact"         # off | exact
    context_mmr: str = "off"             # off | mmr
    context_mmr_lambda: float = 0.7      # 1.0 is pure relevance
    context_mmr_pool: int = 50           # candidates MMR chooses among
    context_order: str = "score_desc"    # score_desc | doc_order | inverted_v

    prompt_template: str = "v0"
    prompt_delimiters: str = "xml"       # xml | markdown | plain
    prompt_question_position: str = "last"   # first | last | both
    prompt_show_scores: bool = False
    prompt_metadata: str = "source"      # none | source | full

    clipboard: bool = True               # falls back to files when unavailable
    judge_template: str = "v0"
    judge_scale: tuple[int, ...] = (0, 1, 2)

    index_root: str = "indexes"
    runs_root: str = "runs"


CFG = Config()


# --------------------------------------------------------------------------
# data contracts
# --------------------------------------------------------------------------
# Every char offset in this file indexes into Document.text (never into the
# raw file on disk, which may differ by newline normalisation). Offsets are
# half-open [start_char, end_char) and survive to ContextBlock, because that is
# what makes document/span-level qrels and answer-span containment possible.

@dataclass(frozen=True)
class Document:
    doc_id: str          # corpus-relative posix path; stable across edits
    source_path: str
    title: str
    text: str
    text_sha256: str     # lets tooling detect qrel spans invalidated by an edit
    n_chars: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str        # f"{doc_id}:{start_char}-{end_char}", deterministic
    doc_id: str
    ordinal: int         # position within the document
    text: str            # always exactly Document.text[start_char:end_char]
    start_char: int
    end_char: int
    n_tokens: int
    header: Optional[str] = None   # contextual header, kept OUT of .text so
                                   # "embed with header, show without" is a
                                   # config axis rather than a rewrite


@dataclass(frozen=True)
class Rewrite:
    rewrite_id: int      # 0 is the user's own query
    text: str
    mode: str
    is_original: bool    # derived from the user query rather than generated


@dataclass
class Candidate:
    """Scores accumulate; a stage never overwrites another stage's field."""
    chunk_id: str
    doc_id: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    retrievers: list[str] = dataclasses.field(default_factory=list)
    rewrite_ids: list[int] = dataclasses.field(default_factory=list)
    ranks: dict[str, int] = dataclasses.field(default_factory=dict)  # "dense@rw0" -> rank


@dataclass
class ContextBlock:
    block_id: int
    doc_id: str
    source_path: str
    title: str
    text: str
    start_char: int      # span of the assembled text, after any expansion
    end_char: int
    chunk_ids: list[str]
    scores: dict[str, Optional[float]]


@dataclass
class Context:
    blocks: list[ContextBlock]
    total_tokens: int
    dropped: list[tuple[str, str]]   # (chunk_id, reason)
    index_hash: str = ""             # so stage 6 never needs the index itself


@dataclass
class PromptBundle:
    text: str            # the string that goes on the clipboard, and nothing else
    n_tokens: int
    query_id: str
    query: str
    block_ids: list[int]
    index_hash: str
    config_hash: str


# --------------------------------------------------------------------------
# embedding adapter
# --------------------------------------------------------------------------
# The pipeline must never learn how a model prefixes queries vs documents.

PREFIX_SCHEMES: dict[str, tuple[str, str]] = {
    "none": ("", ""),
    "bge": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
}


class SentenceTransformerAdapter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prefix_scheme_id = cfg.embed_prefix_scheme
        self._q_prefix, self._d_prefix = PREFIX_SCHEMES[cfg.embed_prefix_scheme]
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # torch import is slow
            self._model = SentenceTransformer(
                self.cfg.embed_model,
                revision=self.cfg.embed_revision,
                device=self.cfg.device,
            )
            dim = self._model.get_sentence_embedding_dimension()
            if dim != self.cfg.embed_dim:
                raise SystemExit(
                    f"embed.dim={self.cfg.embed_dim} but {self.cfg.embed_model} "
                    f"produces {dim}. MRL truncation is not implemented yet."
                )
        return self._model

    @property
    def max_content_tokens(self) -> int:
        """Tokens of *content* that survive; the specials eat the rest."""
        try:
            specials = self.model.tokenizer.num_special_tokens_to_add()
        except Exception:
            specials = 2
        return int(self.model.max_seq_length) - specials

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._q_prefix)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self._d_prefix)

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        vecs = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.cfg.embed_batch_size,
            normalize_embeddings=self.cfg.embed_normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def resolved_revision(self) -> Optional[str]:
        """Best effort: the commit sha actually on disk, for meta.json."""
        for probe in (
            getattr(getattr(self.model[0], "auto_model", None), "name_or_path", ""),
            getattr(getattr(getattr(self.model[0], "auto_model", None), "config", None),
                    "_name_or_path", ""),
            getattr(self.model, "model_card_data", None)
            and getattr(self.model.model_card_data, "base_model_revision", ""),
        ):
            m = re.search(r"[0-9a-f]{40}", str(probe or ""))
            if m:
                return m.group(0)
        return None


# Registry: name -> constructor. This dict is the entire plugin system.
EMBEDDERS: dict[str, Callable[[Config], SentenceTransformerAdapter]] = {
    "BAAI/bge-small-en-v1.5": SentenceTransformerAdapter,
}


_ADAPTERS: dict[tuple, object] = {}     # keep loaded models across queries in one process


def make_embedder(cfg: Config):
    if cfg.embed_model not in EMBEDDERS:
        raise SystemExit(
            f"unknown embed.model {cfg.embed_model!r}; "
            f"known: {sorted(EMBEDDERS)} (add an entry to EMBEDDERS)"
        )
    key = ("embed", cfg.embed_model, cfg.embed_revision, cfg.embed_prefix_scheme,
           cfg.embed_dim, cfg.embed_batch_size, cfg.embed_normalize, cfg.device)
    if key not in _ADAPTERS:
        _ADAPTERS[key] = EMBEDDERS[cfg.embed_model](cfg)
    return _ADAPTERS[key]


def composed_text(chunk: Chunk) -> str:
    """The chunk as a model sees it. Prefixes stay the adapter's business."""
    return chunk.text if not chunk.header else f"{chunk.header}\n\n{chunk.text}"


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

_TITLE_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)


def load_corpus(cfg: Config) -> list[Document]:
    root = Path(cfg.corpus_dir)
    if not root.is_dir():
        raise SystemExit(f"corpus.dir {root} does not exist")
    paths = sorted({p for g in cfg.corpus_globs for p in root.glob(g) if p.is_file()})
    docs: list[Document] = []
    for p in paths:
        raw = p.read_text(encoding="utf-8", errors="replace")
        text = raw.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            continue
        m = _TITLE_RE.search(text[:2000])
        docs.append(Document(
            doc_id=p.relative_to(root).as_posix(),
            source_path=str(p),
            title=m.group(1) if m else p.stem,
            text=text,
            text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            n_chars=len(text),
        ))
    if not docs:
        raise SystemExit(f"no documents matched {list(cfg.corpus_globs)} under {root}")
    return docs


def corpus_fingerprint(docs: list[Document]) -> str:
    """Feeds the index hash: editing or adding a document must invalidate."""
    blob = "\n".join(f"{d.doc_id}\t{d.text_sha256}" for d in sorted(docs, key=lambda d: d.doc_id))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# stage 1a: chunking
# --------------------------------------------------------------------------

Span = tuple[int, int]


def _memo_counter(text: str, count_tokens: Callable[[str], int]) -> Callable[[Span], int]:
    cache: dict[Span, int] = {}

    def tok(span: Span) -> int:
        if span not in cache:
            cache[span] = count_tokens(text[span[0]:span[1]])
        return cache[span]
    return tok


def _split_on(text: str, start: int, end: int, sep: str) -> list[Span]:
    """Tile [start,end) on `sep`, keeping the separator with the part before it."""
    spans: list[Span] = []
    pos = start
    while True:
        hit = text.find(sep, pos, end)
        if hit == -1:
            break
        cut = hit + len(sep)
        spans.append((pos, cut))
        pos = cut
    if pos < end:
        spans.append((pos, end))
    return spans


def _fit_end(pos: int, cap: int, max_tokens: int, tok: Callable[[Span], int]) -> int:
    """Largest e in (pos, cap] whose span fits the budget.

    Gallops out from pos before bisecting, so an unbreakable megabyte costs work
    proportional to one chunk rather than to the whole region.
    """
    if tok((pos, cap)) <= max_tokens:
        return cap
    step, hi = 64, min(pos + 64, cap)
    while hi < cap and tok((pos, hi)) <= max_tokens:
        step *= 2
        hi = min(pos + step, cap)
    lo = pos + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tok((pos, mid)) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _fit_start(end: int, floor: int, max_tokens: int, tok: Callable[[Span], int]) -> int:
    """Smallest s in (floor, end] whose span fits the budget."""
    lo, hi = floor + 1, end
    while lo < hi:
        mid = (lo + hi) // 2
        if tok((mid, end)) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _split_spans(text: str, start: int, end: int, seps: list[str], max_tokens: int,
                 tok: Callable[[Span], int]) -> list[Span]:
    """Tile [start,end) at separator boundaries.

    A region with no applicable separator is returned oversized rather than cut
    at an arbitrary character: the packer cuts it, so every boundary in `bounds`
    is a real separator boundary and crossing one is always a semantic choice.
    """
    if start >= end:
        return []
    if tok((start, end)) <= max_tokens:
        return [(start, end)]
    for i, sep in enumerate(seps):
        parts = _split_on(text, start, end, sep)
        if len(parts) > 1:
            out: list[Span] = []
            for ps, pe in parts:
                out.extend(_split_spans(text, ps, pe, seps[i + 1:], max_tokens, tok))
            return out
    return [(start, end)]


def _trim(text: str, span: Span) -> Span:
    s, e = span
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return (s, e)


def _overlap_start(text: str, pos: int, end: int, cfg: Config, bounds: list[int], i0: int,
                   tok: Callable[[Span], int]) -> int:
    """Where the next chunk begins: the earliest cut point whose tail fits chunk.overlap."""
    if cfg.chunk_overlap <= 0:
        return end
    for b in bounds[i0:]:
        if b <= pos:
            continue
        if b >= end:
            break
        if tok((b, end)) <= cfg.chunk_overlap:
            return b
    # no interior cut point (one long paragraph, or an unbreakable run): bisect on
    # characters rather than silently dropping a configured overlap
    p = _fit_start(end, pos, cfg.chunk_overlap, tok)
    q = p
    while q < end and not text[q - 1].isspace():
        q += 1                       # prefer not to start the next chunk mid-word
    if q < end and tok((q, end)) > 0:
        p = q                        # ...but never let that snap eat the overlap:
                                     # inside a long unbroken run it would skip the
                                     # whole run and leave no overlap at all
    return p if pos < p < end else end


def _chunk_recursive(doc: Document, cfg: Config, count_tokens: Callable[[str], int]) -> list[Chunk]:
    text = doc.text
    tok = _memo_counter(text, count_tokens)
    leaves = [s for s in _split_spans(text, 0, len(text), list(cfg.chunk_separators),
                                      cfg.chunk_size, tok)
              if text[s[0]:s[1]].strip()]
    if not leaves:
        return []

    bounds = [e for _, e in leaves]          # separator boundaries, ascending
    chunks: list[Chunk] = []
    pos, last, i0, prev_end = leaves[0][0], bounds[-1], 0, -1
    while pos < last:
        while bounds[i0] <= pos:
            i0 += 1
        j = i0                               # extend to the last boundary that still fits
        while j + 1 < len(bounds) and tok((pos, bounds[j + 1])) <= cfg.chunk_size:
            j += 1
        end = bounds[j]
        if tok((pos, end)) > cfg.chunk_size:
            end = _fit_end(pos, end, cfg.chunk_size, tok)   # oversized leaf
        if end <= prev_end:
            # the previous chunk already reached this boundary; push past it so every
            # chunk adds new material instead of re-emitting the overlap as a fragment
            cap = bounds[j + 1] if j + 1 < len(bounds) else last
            end = max(_fit_end(pos, cap, cfg.chunk_size, tok), prev_end + 1)
        span = _trim(text, (pos, end))
        if span[0] < span[1] and (not chunks or span != (chunks[-1].start_char, chunks[-1].end_char)):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}:{span[0]}-{span[1]}",
                doc_id=doc.doc_id,
                ordinal=len(chunks),
                text=text[span[0]:span[1]],
                start_char=span[0],
                end_char=span[1],
                n_tokens=tok(span),
            ))
        prev_end = end
        if end >= last:
            break
        pos = _overlap_start(text, pos, end, cfg, bounds, i0, tok)
    return chunks


CHUNKERS: dict[str, Callable[[Document, Config, Callable[[str], int]], list[Chunk]]] = {
    "recursive": _chunk_recursive,
}


def chunk_document(doc: Document, cfg: Config, count_tokens: Callable[[str], int]) -> list[Chunk]:
    if cfg.chunk_header_mode != "none":
        raise SystemExit(f"chunk.header_mode={cfg.chunk_header_mode!r} not implemented")
    if cfg.chunk_strategy not in CHUNKERS:
        raise SystemExit(f"unknown chunk.strategy {cfg.chunk_strategy!r}; known: {sorted(CHUNKERS)}")
    chunks = CHUNKERS[cfg.chunk_strategy](doc, cfg, count_tokens)
    for c in chunks:   # the invariant the whole span contract rests on
        assert doc.text[c.start_char:c.end_char] == c.text, c.chunk_id
    return chunks


# --------------------------------------------------------------------------
# stage 1b: index identity and build
# --------------------------------------------------------------------------

def index_identity(cfg: Config, fingerprint: str) -> tuple[str, dict]:
    payload = {
        "index_format_version": INDEX_FORMAT_VERSION,
        "chunk.strategy": cfg.chunk_strategy,
        "chunk.size": cfg.chunk_size,
        "chunk.overlap": cfg.chunk_overlap,
        "chunk.separators": list(cfg.chunk_separators),
        "chunk.header_mode": cfg.chunk_header_mode,
        "embed.model": cfg.embed_model,
        "embed.revision": cfg.embed_revision,
        "embed.dim": cfg.embed_dim,
        "embed.prefix_scheme": cfg.embed_prefix_scheme,
        "corpus.fingerprint": fingerprint,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest(), payload


def config_hash(cfg: Config) -> str:
    """Identifies a whole run. Unlike the index hash this covers query-time keys too."""
    blob = json.dumps(dataclasses.asdict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class Index:
    path: Path
    meta: dict
    docs: dict[str, Document]
    chunks: list[Chunk]
    matrix: np.ndarray                 # [n_chunks, dim], L2-normalised, row i <-> chunks[i]
    _by_id: dict[str, int] = dataclasses.field(default_factory=dict)
    _by_doc: dict[str, list[Chunk]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._by_doc = {}
        for c in self.chunks:
            self._by_doc.setdefault(c.doc_id, []).append(c)
        for cs in self._by_doc.values():
            cs.sort(key=lambda c: c.ordinal)

    @property
    def index_hash(self) -> str:
        return self.meta["index_hash"]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[self._by_id[chunk_id]]

    def get_doc(self, doc_id: str) -> Document:
        return self.docs[doc_id]

    def row_of(self, chunk_id: str) -> int:
        return self._by_id[chunk_id]

    def doc_chunks(self, doc_id: str) -> list[Chunk]:
        return self._by_doc[doc_id]


def _write_jsonl(path: Path, rows: Iterator[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _overlap_stats(chunks: list[Chunk]) -> dict:
    """Realised overlap, so a config asking for 10% that cannot get it is visible."""
    chars, zero = [], 0
    for a, b in zip(chunks, chunks[1:]):
        if a.doc_id != b.doc_id:
            continue
        ov = a.end_char - b.start_char
        chars.append(max(ov, 0))
        zero += ov <= 0
    return {"p50_chars": int(np.percentile(chars, 50)) if chars else 0,
            "zero_overlap_boundaries": int(zero),
            "boundaries": len(chars)}


def _token_stats(chunks: list[Chunk]) -> dict:
    n = np.array([c.n_tokens for c in chunks]) if chunks else np.array([0])
    return {
        "min": int(n.min()), "p50": int(np.percentile(n, 50)),
        "mean": round(float(n.mean()), 1), "p95": int(np.percentile(n, 95)),
        "max": int(n.max()),
        # the 43-token-fragment failure mode, counted directly
        "under_64_tokens": int((n < 64).sum()),
    }


def build_index(cfg: Config = CFG, rebuild: bool = False, verbose: bool = True) -> Index:
    t0 = time.time()
    docs = load_corpus(cfg)
    fingerprint = corpus_fingerprint(docs)
    ihash, payload = index_identity(cfg, fingerprint)
    path = Path(cfg.index_root) / ihash[:12]

    if path.is_dir() and not rebuild:
        if verbose:
            print(f"index {ihash[:12]} already built at {path}")
        return load_index(cfg)

    adapter = make_embedder(cfg)
    if cfg.chunk_size > adapter.max_content_tokens:
        print(f"WARNING chunk.size={cfg.chunk_size} exceeds the {cfg.embed_model} content "
              f"budget of {adapter.max_content_tokens} tokens; the tail of full-size chunks "
              f"is silently dropped at embed time. Set chunk.size<={adapter.max_content_tokens}.",
              file=sys.stderr)

    chunks: list[Chunk] = []
    for n, doc in enumerate(docs, 1):
        chunks.extend(chunk_document(doc, cfg, adapter.count_tokens))
        if verbose and (n % 25 == 0 or n == len(docs)):
            print(f"  chunked {n}/{len(docs)} docs -> {len(chunks)} chunks", end="\r")
    if verbose:
        print()

    vecs = np.empty((len(chunks), cfg.embed_dim), dtype=np.float32)
    slice_size = max(cfg.embed_batch_size * 16, 256)
    t_embed = time.time()
    for start in range(0, len(chunks), slice_size):
        batch = chunks[start:start + slice_size]
        vecs[start:start + len(batch)] = adapter.encode_documents([composed_text(c) for c in batch])
        if verbose:
            done = start + len(batch)
            rate = done / max(time.time() - t_embed, 1e-6)
            print(f"  embedded {done}/{len(chunks)} chunks ({rate:.0f}/s)", end="\r")
    if verbose:
        print()

    meta = {
        "index_hash": ihash,
        "identity": payload,          # the exact inputs, spelled out for diffing
        "embed.resolved_revision": adapter.resolved_revision(),
        "embed.prefix_scheme_id": adapter.prefix_scheme_id,
        "embed.max_content_tokens": adapter.max_content_tokens,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "dim": cfg.embed_dim,
        "normalized": cfg.embed_normalize,
        "chunk_tokens": _token_stats(chunks),
        "chunk_overlap": _overlap_stats(chunks),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
    }

    # write to a temp dir and rename, so an interrupted build never leaves a
    # directory whose name claims a hash its contents do not have
    tmp = path.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _write_jsonl(tmp / "docs.jsonl", (dataclasses.asdict(d) for d in docs))
    _write_jsonl(tmp / "chunks.jsonl", (dataclasses.asdict(c) for c in chunks))
    np.save(tmp / "embeddings.npy", vecs)
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        shutil.rmtree(path)
    os.replace(tmp, path)

    if verbose:
        print(f"built index {ihash[:12]} at {path} in {meta['build_seconds']}s")
    return load_index(cfg)


def load_index(cfg: Config = CFG) -> Index:
    docs = load_corpus(cfg)
    ihash, payload = index_identity(cfg, corpus_fingerprint(docs))
    path = Path(cfg.index_root) / ihash[:12]
    if not path.is_dir():
        raise SystemExit(f"no index for this config (want {ihash[:12]}); run: python rag.py index")

    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if meta.get("index_hash") != ihash:
        diff = [k for k, v in payload.items() if meta.get("identity", {}).get(k) != v]
        raise SystemExit(
            f"index at {path} was built for {str(meta.get('index_hash'))[:12]}, config wants "
            f"{ihash[:12]}; differing keys: {diff or ['<unknown>']}. Run: python rag.py index"
        )

    doc_rows = [json.loads(l) for l in (path / "docs.jsonl").read_text(encoding="utf-8").splitlines()]
    chunk_rows = [json.loads(l) for l in (path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    # mmap keeps peak RSS honest, which is the metric the brief asks for
    matrix = np.load(path / "embeddings.npy", mmap_mode="r")
    if matrix.shape != (len(chunk_rows), cfg.embed_dim):
        raise SystemExit(f"index at {path} is corrupt: embeddings {matrix.shape} vs "
                         f"{len(chunk_rows)} chunks x {cfg.embed_dim}")
    return Index(
        path=path,
        meta=meta,
        docs={r["doc_id"]: Document(**r) for r in doc_rows},
        chunks=[Chunk(**r) for r in chunk_rows],
        matrix=matrix,
    )


# --------------------------------------------------------------------------
# stage 2: query rewriting
# --------------------------------------------------------------------------
# Always returns a list. In `off` mode that list is [original_query], so
# multi-query and decomposition are a registry entry, not a signature change.

def _rewrite_off(query: str, cfg: Config) -> list[Rewrite]:
    return [Rewrite(rewrite_id=0, text=query, mode="off", is_original=True)]


def _rewrite_normalize(query: str, cfg: Config) -> list[Rewrite]:
    text = re.sub(r"\s+", " ", query).strip()
    return [Rewrite(rewrite_id=0, text=text, mode="normalize", is_original=True)]


REWRITERS: dict[str, Callable[[str, Config], list[Rewrite]]] = {
    "off": _rewrite_off,
    "normalize": _rewrite_normalize,
}


def rewrite_query(query: str, cfg: Config = CFG) -> list[Rewrite]:
    if cfg.rewrite_mode not in REWRITERS:
        raise SystemExit(f"unknown rewrite.mode {cfg.rewrite_mode!r}; known: {sorted(REWRITERS)}")
    rewrites = REWRITERS[cfg.rewrite_mode](query, cfg)
    if not rewrites:
        raise SystemExit(f"rewrite.mode {cfg.rewrite_mode!r} returned no rewrites")
    return rewrites


# --------------------------------------------------------------------------
# stage 3: embedding & retrieval
# --------------------------------------------------------------------------
# Brute-force numpy over the whole matrix. No ANN index: at 50k x 384 a full
# scan is single-digit milliseconds, and query latency here is human-scale.

def _cosine_scores(index: Index, qvecs: np.ndarray) -> np.ndarray:
    """[n_rewrites, n_chunks]. How vectors are compared stays inside the retriever."""
    scores = qvecs @ index.matrix.T
    if not index.meta.get("normalized", True):
        qn = np.linalg.norm(qvecs, axis=1, keepdims=True)
        dn = np.linalg.norm(index.matrix, axis=1)[None, :]
        scores = scores / np.maximum(qn * dn, 1e-12)
    return np.asarray(scores, dtype=np.float32)


def _retrieve_dense(rewrites: list[Rewrite], index: Index, cfg: Config) -> list[Candidate]:
    if not index.chunks:
        return []
    adapter = make_embedder(cfg)
    scores = _cosine_scores(index, adapter.encode_queries([r.text for r in rewrites]))
    k = max(1, min(cfg.retrieve_top_k, len(index.chunks)))

    merged: dict[str, Candidate] = {}
    for row, rw in zip(scores, rewrites):
        top = np.argpartition(-row, k - 1)[:k]
        top = top[np.argsort(-row[top], kind="stable")]
        for rank, i in enumerate(top, 1):          # 1-based: RRF wants it that way
            chunk = index.chunks[int(i)]
            cand = merged.get(chunk.chunk_id)
            if cand is None:
                cand = Candidate(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
                merged[chunk.chunk_id] = cand
            score = float(row[int(i)])
            # accumulate, never overwrite: a chunk surfaced by two rewrites keeps
            # its best dense score and the provenance of both
            cand.dense_score = score if cand.dense_score is None else max(cand.dense_score, score)
            if "dense" not in cand.retrievers:
                cand.retrievers.append("dense")
            if rw.rewrite_id not in cand.rewrite_ids:
                cand.rewrite_ids.append(rw.rewrite_id)
            cand.ranks[f"dense@rw{rw.rewrite_id}"] = rank
    return sorted(merged.values(), key=lambda c: (-(c.dense_score or 0.0), c.chunk_id))


RETRIEVERS: dict[str, Callable[[list[Rewrite], Index, Config], list[Candidate]]] = {
    "dense": _retrieve_dense,
}


def retrieve(rewrites: list[Rewrite], index: Index, cfg: Config = CFG) -> list[Candidate]:
    """retrieve.top_k is depth *per rewrite*; the merged pool is returned whole."""
    if cfg.retrieve_mode not in RETRIEVERS:
        raise SystemExit(f"unknown retrieve.mode {cfg.retrieve_mode!r}; known: {sorted(RETRIEVERS)}")
    return RETRIEVERS[cfg.retrieve_mode](rewrites, index, cfg)


# --------------------------------------------------------------------------
# stage 4: reranking
# --------------------------------------------------------------------------
# Scores candidates and reorders them. It never drops one: keeping the demoted
# tail is what lets you ask whether a reranker pushed a gold chunk out of the
# context, which is the whole reason to measure a reranker separately.

class CrossEncoderAdapter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.cfg.rerank_model,
                revision=self.cfg.rerank_revision,
                device=self.cfg.device,
            )
        return self._model

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        out = self.model.predict(pairs, batch_size=self.cfg.rerank_batch_size,
                                 show_progress_bar=False, convert_to_numpy=True)
        return [float(x) for x in np.asarray(out).reshape(-1)]


RERANKER_ADAPTERS: dict[str, Callable[[Config], CrossEncoderAdapter]] = {
    "cross-encoder/ms-marco-MiniLM-L-6-v2": CrossEncoderAdapter,
    "cross-encoder/ms-marco-MiniLM-L-12-v2": CrossEncoderAdapter,
    "BAAI/bge-reranker-base": CrossEncoderAdapter,
}


def make_reranker(cfg: Config) -> CrossEncoderAdapter:
    if cfg.rerank_model not in RERANKER_ADAPTERS:
        raise SystemExit(
            f"unknown rerank.model {cfg.rerank_model!r}; "
            f"known: {sorted(RERANKER_ADAPTERS)} (add an entry to RERANKER_ADAPTERS)"
        )
    key = ("rerank", cfg.rerank_model, cfg.rerank_revision, cfg.rerank_batch_size, cfg.device)
    if key not in _ADAPTERS:
        _ADAPTERS[key] = RERANKER_ADAPTERS[cfg.rerank_model](cfg)
    return _ADAPTERS[key]


def _rerank_query_texts(query: str, rewrites: list[Rewrite], cfg: Config) -> list[str]:
    """Which query the reranker scores against -- an ablation axis in its own right."""
    if cfg.rerank_query == "original":
        return [query]
    if cfg.rerank_query == "rewrite0":
        return [rewrites[0].text]
    if cfg.rerank_query == "each":
        return [r.text for r in rewrites]
    raise SystemExit(f"unknown rerank.query {cfg.rerank_query!r}; "
                     f"known: ['original', 'rewrite0', 'each']")


def _rerank_off(query: str, rewrites: list[Rewrite], candidates: list[Candidate],
                index: Index, cfg: Config) -> list[Candidate]:
    return candidates


def _rerank_cross_encoder(query: str, rewrites: list[Rewrite], candidates: list[Candidate],
                          index: Index, cfg: Config) -> list[Candidate]:
    if not candidates:
        return candidates
    depth = max(0, cfg.rerank_depth)
    head, tail = candidates[:depth], candidates[depth:]
    if not head:
        return candidates

    adapter = make_reranker(cfg)
    queries = _rerank_query_texts(query, rewrites, cfg)
    docs = [composed_text(index.get_chunk(c.chunk_id)) for c in head]
    per_query = [adapter.score([(q, d) for d in docs]) for q in queries]
    for i, cand in enumerate(head):
        cand.rerank_score = max(scores[i] for scores in per_query)

    head = sorted(head, key=lambda c: (-(c.rerank_score or 0.0), c.chunk_id))
    # the unscored tail is retained, after the scored head, in its prior order
    return head + tail


RERANKERS: dict[str, Callable[..., list[Candidate]]] = {
    "off": _rerank_off,
    "cross_encoder": _rerank_cross_encoder,
}


def rerank(query: str, rewrites: list[Rewrite], candidates: list[Candidate],
           index: Index, cfg: Config = CFG) -> list[Candidate]:
    """Sets rerank_score and reorders. Every candidate in, every candidate out;
    keeping only the top n is stage 5's decision, made against a full list."""
    if cfg.rerank_mode not in RERANKERS:
        raise SystemExit(f"unknown rerank.mode {cfg.rerank_mode!r}; known: {sorted(RERANKERS)}")
    return RERANKERS[cfg.rerank_mode](query, rewrites, candidates, index, cfg)


# --------------------------------------------------------------------------
# stage 5: context construction
# --------------------------------------------------------------------------
# Chunks overlap by design, so selected chunks are merged into contiguous
# spans before anything is measured or rendered. A block's text stays exactly
# Document.text[start_char:end_char], which is what keeps answer-span
# containment a span-arithmetic question rather than a string search.

def _relevance(c: Candidate) -> float:
    return c.rerank_score if c.rerank_score is not None else (c.dense_score or 0.0)


def _scores_of(c: Candidate) -> dict[str, Optional[float]]:
    return {"dense": c.dense_score, "bm25": c.bm25_score,
            "rrf": c.rrf_score, "rerank": c.rerank_score}


def _best_scores(a: dict, b: dict) -> dict:
    return {k: (a[k] if b[k] is None else b[k] if a[k] is None else max(a[k], b[k])) for k in a}


def _dedup(candidates: list[Candidate], index: Index, cfg: Config,
           dropped: list[tuple[str, str]]) -> list[Candidate]:
    if cfg.context_dedup == "off":
        return list(candidates)
    if cfg.context_dedup != "exact":
        raise SystemExit(f"unknown context.dedup {cfg.context_dedup!r}; known: ['off', 'exact']")
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        key = " ".join(index.get_chunk(c.chunk_id).text.split()).casefold()
        if key in seen:
            dropped.append((c.chunk_id, "dedup"))
            continue
        seen.add(key)
        out.append(c)
    return out


def _select_head(candidates: list[Candidate], index: Index, cfg: Config) -> list[Candidate]:
    return candidates[:cfg.context_top_n]


def _select_mmr(candidates: list[Candidate], index: Index, cfg: Config) -> list[Candidate]:
    """Maximal marginal relevance over the candidate pool."""
    pool = candidates[:max(cfg.context_mmr_pool, cfg.context_top_n)]
    if not pool:
        return []
    rel = np.array([_relevance(c) for c in pool], dtype=np.float32)
    lo, hi = float(rel.min()), float(rel.max())
    # min-max first: a cross-encoder logit and a cosine similarity are not on the
    # same scale, and mixing them raw would make lambda meaningless
    rel = (rel - lo) / (hi - lo) if hi > lo else np.ones_like(rel)
    vecs = np.asarray([index.matrix[index.row_of(c.chunk_id)] for c in pool], dtype=np.float32)
    sim = vecs @ vecs.T
    lam = cfg.context_mmr_lambda

    chosen: list[int] = []
    remaining = list(range(len(pool)))
    while remaining and len(chosen) < cfg.context_top_n:
        if not chosen:
            best = max(remaining, key=lambda i: rel[i])
        else:
            best = max(remaining,
                       key=lambda i: lam * rel[i] - (1 - lam) * max(sim[i][j] for j in chosen))
        chosen.append(best)
        remaining.remove(best)
    return [pool[i] for i in chosen]


SELECTORS: dict[str, Callable[[list[Candidate], Index, Config], list[Candidate]]] = {
    "off": _select_head,
    "mmr": _select_mmr,
}


def _expand(selected: list[Candidate], index: Index, cfg: Config) -> list[dict]:
    """One span group per selected candidate, widened by neighbour expansion."""
    groups = []
    for priority, c in enumerate(selected):
        chunk = index.get_chunk(c.chunk_id)
        members = [chunk]
        if cfg.context_neighbours > 0:
            siblings = index.doc_chunks(c.doc_id)      # sorted by ordinal
            lo = max(0, chunk.ordinal - cfg.context_neighbours)
            hi = min(len(siblings), chunk.ordinal + cfg.context_neighbours + 1)
            members = siblings[lo:hi]
        groups.append({
            "doc_id": c.doc_id,
            "start": min(m.start_char for m in members),
            "end": max(m.end_char for m in members),
            "chunk_ids": [m.chunk_id for m in members],
            "priority": priority,
            "scores": _scores_of(c),
        })
    return groups


def _merge(groups: list[dict]) -> list[dict]:
    """Union overlapping or touching spans within a document."""
    out: list[dict] = []
    for g in sorted(groups, key=lambda g: (g["doc_id"], g["start"], g["end"])):
        prev = out[-1] if out else None
        if prev and prev["doc_id"] == g["doc_id"] and g["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], g["end"])
            prev["chunk_ids"] = list(dict.fromkeys(prev["chunk_ids"] + g["chunk_ids"]))
            prev["priority"] = min(prev["priority"], g["priority"])
            prev["scores"] = _best_scores(prev["scores"], g["scores"])
        else:
            out.append(dict(g))
    return out


def _order_blocks(blocks: list[ContextBlock], cfg: Config) -> list[ContextBlock]:
    if cfg.context_order == "score_desc":
        return blocks                                    # already in priority order
    if cfg.context_order == "doc_order":
        return sorted(blocks, key=lambda b: (b.doc_id, b.start_char))
    if cfg.context_order == "inverted_v":
        front, back = [], []                             # strongest at both ends
        for i, b in enumerate(blocks):
            (front if i % 2 == 0 else back).append(b)
        return front + back[::-1]
    raise SystemExit(f"unknown context.order {cfg.context_order!r}; "
                     f"known: ['score_desc', 'doc_order', 'inverted_v']")


def build_context(candidates: list[Candidate], index: Index, cfg: Config = CFG) -> Context:
    if cfg.context_mmr not in SELECTORS:
        raise SystemExit(f"unknown context.mmr {cfg.context_mmr!r}; known: {sorted(SELECTORS)}")
    dropped: list[tuple[str, str]] = []

    kept = _dedup(candidates, index, cfg, dropped)
    selected = SELECTORS[cfg.context_mmr](kept, index, cfg)
    chosen = {c.chunk_id for c in selected}
    dropped.extend((c.chunk_id, "top_n") for c in kept if c.chunk_id not in chosen)

    count_tokens = make_embedder(cfg).count_tokens
    blocks: list[ContextBlock] = []
    total = 0
    for g in sorted(_merge(_expand(selected, index, cfg)), key=lambda g: g["priority"]):
        doc = index.get_doc(g["doc_id"])
        start, end = g["start"], g["end"]
        n = count_tokens(doc.text[start:end])
        if total + n > cfg.context_budget_tokens:
            if blocks:
                dropped.extend((cid, "budget") for cid in g["chunk_ids"])
                continue
            # the top block alone busts the budget: truncate rather than return an
            # empty context, and keep the span honest about what was kept
            tok = _memo_counter(doc.text, count_tokens)
            end = _fit_end(start, end, cfg.context_budget_tokens, tok)
            n = tok((start, end))
            dropped.extend((cid, "budget_truncated") for cid in g["chunk_ids"])
        blocks.append(ContextBlock(
            block_id=0,
            doc_id=doc.doc_id,
            source_path=doc.source_path,
            title=doc.title,
            text=doc.text[start:end],
            start_char=start,
            end_char=end,
            chunk_ids=g["chunk_ids"],
            scores=g["scores"],
        ))
        total += n

    blocks = _order_blocks(blocks, cfg)
    for i, b in enumerate(blocks, 1):
        b.block_id = i
    return Context(blocks=blocks, total_tokens=total, dropped=dropped,
                   index_hash=index.index_hash)


# --------------------------------------------------------------------------
# stage 6: prompt assembly
# --------------------------------------------------------------------------
# Templates are versioned functions in this file, so two runs are comparable and
# a template change is a diff. Config and retrieval scores stay out of the text
# unless prompt.show_scores asks for them.

def _block_meta(b: ContextBlock, cfg: Config) -> dict[str, str]:
    meta: dict[str, str] = {"id": str(b.block_id)}
    if cfg.prompt_metadata not in ("none", "source", "full"):
        raise SystemExit(f"unknown prompt.metadata {cfg.prompt_metadata!r}; "
                         f"known: ['none', 'source', 'full']")
    if cfg.prompt_metadata in ("source", "full"):
        meta["source"] = b.doc_id            # corpus-relative, not a local path
    if cfg.prompt_metadata == "full":
        meta["title"] = b.title
        meta["span"] = f"{b.start_char}-{b.end_char}"
    if cfg.prompt_show_scores:
        best = max((v for v in b.scores.values() if v is not None), default=None)
        if best is not None:
            meta["score"] = f"{best:.4f}"
    return meta


def _render_block(b: ContextBlock, cfg: Config) -> str:
    meta = _block_meta(b, cfg)
    if cfg.prompt_delimiters == "xml":
        attrs = " ".join(f'{k}="{v}"' for k, v in meta.items())
        body = b.text.replace("</excerpt>", "<\\/excerpt>")   # never break the framing
        return f"<excerpt {attrs}>\n{body}\n</excerpt>"
    if cfg.prompt_delimiters == "markdown":
        head = " · ".join(f"{k}: {v}" for k, v in meta.items())
        return f"### Excerpt {meta['id']}\n{head}\n\n{b.text}"
    if cfg.prompt_delimiters == "plain":
        head = " | ".join(f"{k}: {v}" for k, v in meta.items())
        return f"[{meta['id']}] {head}\n{b.text}\n" + "-" * 40
    raise SystemExit(f"unknown prompt.delimiters {cfg.prompt_delimiters!r}; "
                     f"known: ['xml', 'markdown', 'plain']")


def _prompt_v0(query: str, context: Context, cfg: Config) -> str:
    question = f"<question>\n{query}\n</question>" if cfg.prompt_delimiters == "xml" \
        else f"Question: {query}"
    excerpts = "\n\n".join(_render_block(b, cfg) for b in context.blocks)
    instruction = (
        "Answer the question using only the excerpts above. Cite the excerpt ids you "
        "used, like [2]. If the excerpts do not contain the answer, say so plainly "
        "instead of guessing."
    )
    parts = ["Below are excerpts retrieved from a document collection."]
    if cfg.prompt_question_position in ("first", "both"):
        parts.append(question)
    parts.append(excerpts)
    parts.append(instruction)
    if cfg.prompt_question_position in ("last", "both"):
        parts.append(question)
    if cfg.prompt_question_position not in ("first", "last", "both"):
        raise SystemExit(f"unknown prompt.question_position {cfg.prompt_question_position!r}; "
                         f"known: ['first', 'last', 'both']")
    return "\n\n".join(parts) + "\n"


PROMPT_TEMPLATES: dict[str, Callable[[str, Context, Config], str]] = {
    "v0": _prompt_v0,
}


def assemble_prompt(query: str, context: Context, cfg: Config = CFG,
                    query_id: Optional[str] = None) -> PromptBundle:
    if cfg.prompt_template not in PROMPT_TEMPLATES:
        raise SystemExit(f"unknown prompt.template {cfg.prompt_template!r}; "
                         f"known: {sorted(PROMPT_TEMPLATES)}")
    text = PROMPT_TEMPLATES[cfg.prompt_template](query, context, cfg)
    return PromptBundle(
        text=text,
        n_tokens=make_embedder(cfg).count_tokens(text),
        query_id=query_id or hashlib.sha256(query.encode("utf-8")).hexdigest()[:8],
        query=query,
        block_ids=[b.block_id for b in context.blocks],
        index_hash=context.index_hash,
        config_hash=config_hash(cfg),
    )


# --------------------------------------------------------------------------
# the whole pipeline, stages 2-6
# --------------------------------------------------------------------------

def run_pipeline(query: str, index: Index, cfg: Config = CFG,
                 query_id: Optional[str] = None
                 ) -> tuple[PromptBundle, Context, list[Candidate], dict]:
    timing = {}
    t = time.time()
    rewrites = rewrite_query(query, cfg)
    timing["rewrite_ms"] = round((time.time() - t) * 1000, 1)
    t = time.time()
    candidates = retrieve(rewrites, index, cfg)
    timing["retrieve_ms"] = round((time.time() - t) * 1000, 1)
    t = time.time()
    candidates = rerank(query, rewrites, candidates, index, cfg)
    timing["rerank_ms"] = round((time.time() - t) * 1000, 1)
    t = time.time()
    context = build_context(candidates, index, cfg)
    timing["context_ms"] = round((time.time() - t) * 1000, 1)
    t = time.time()
    bundle = assemble_prompt(query, context, cfg, query_id=query_id)
    timing["prompt_ms"] = round((time.time() - t) * 1000, 1)
    return bundle, context, candidates, timing


# --------------------------------------------------------------------------
# stage 7: manual generation
# --------------------------------------------------------------------------
# Nothing here may call a generation API, and nothing ever will: the manual step
# is the architecture. This tool puts a prompt on the clipboard, a human pastes
# it into a frontier model, copies the answer back and presses Enter.
#
# Generation is strictly one query at a time. Several queries' contexts in one
# prompt would let the model answer query 7 from chunks retrieved for query 3,
# which inflates scores unpredictably. Judging is the opposite: one batched
# paste for every triple, from a versioned template, with nothing in it that
# identifies which config produced which answer.

def _clipboard() -> tuple[Optional[Callable[[str], None]], Optional[Callable[[], str]]]:
    """(copy, paste), or (None, None) when there is no usable clipboard."""
    try:
        import pyperclip
        pyperclip.paste()          # imports fine with no backend; this is the real probe
    except Exception:
        return None, None
    return pyperclip.copy, pyperclip.paste


def deliver_prompt(bundle: PromptBundle, cfg: Config = CFG,
                   out_dir: Optional[Path] = None) -> str:
    """Put the prompt where a human can paste it. Returns where it went."""
    copy, _ = _clipboard() if cfg.clipboard else (None, None)
    if copy is not None:
        copy(bundle.text)
        return "clipboard"
    path = Path(out_dir or ".") / "prompts" / f"{bundle.query_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bundle.text, encoding="utf-8")
    return str(path)


def collect_answer(bundle: PromptBundle, cfg: Config = CFG,
                   out_dir: Optional[Path] = None) -> Optional[str]:
    """Read the human's answer back. Returns None if there is nothing to read."""
    _, paste = _clipboard() if cfg.clipboard else (None, None)
    if paste is not None:
        return paste()
    path = Path(out_dir or ".") / "answers" / f"{bundle.query_id}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


@dataclass(frozen=True)
class QueryEntry:
    query_id: str
    query: str
    ground_truth: Optional[str] = None


def load_queries(path: str) -> list[QueryEntry]:
    """One query per line. Optionally `id<TAB>query` or `id<TAB>query<TAB>ground truth`."""
    entries: list[QueryEntry] = []
    seen: set[str] = set()
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) == 1:
            qid, query, truth = hashlib.sha256(line.encode()).hexdigest()[:8], parts[0], None
        else:
            qid, query = parts[0].strip(), parts[1].strip()
            truth = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        if qid in seen:
            raise SystemExit(f"{path}:{n}: duplicate query_id {qid!r}")
        seen.add(qid)
        entries.append(QueryEntry(qid, query, truth))
    if not entries:
        raise SystemExit(f"no queries in {path}")
    return entries


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())      # answers are expensive to produce; do not lose them


def _latest_by(rows: list[dict], key: str) -> list[dict]:
    """per_query.jsonl is append-only, so a retried query has more than one row;
    the last one is the authoritative record for that query."""
    out: dict[str, dict] = {}
    for row in rows:
        out[row[key]] = row
    return list(out.values())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def open_run(cfg: Config = CFG, run: Optional[str] = None) -> Path:
    """Runs are directories, not database rows."""
    if run:
        path = Path(run)
        if not path.is_dir():
            raise SystemExit(f"no run directory at {path}")
        return path
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    base = Path(cfg.runs_root) / f"{stamp}-{config_hash(cfg)[:8]}"
    path, n = base, 1
    while path.exists():        # two runs in the same second must not share a directory
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(dataclasses.asdict(cfg), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _peak_rss_mb() -> float:
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _write_timing(run_dir: Path, rows: list[dict]) -> None:
    rows = _latest_by(rows, "query_id")
    stages = ["rewrite_ms", "retrieve_ms", "rerank_ms", "context_ms", "prompt_ms"]
    out: dict = {"n_queries": len(rows), "peak_rss_mb": _peak_rss_mb()}
    for stage in stages:
        vals = [r["timing"][stage] for r in rows if "timing" in r]
        if vals:
            out[stage] = {"p50": round(float(np.percentile(vals, 50)), 1),
                          "p95": round(float(np.percentile(vals, 95)), 1)}
    (run_dir / "timing.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


def run_generation_loop(entries: list[QueryEntry], index: Index, run_dir: Path,
                        cfg: Config = CFG) -> None:
    """One query at a time. Never batch this."""
    answers = {r["query_id"]: r for r in _read_jsonl(run_dir / "answers.jsonl")}
    per_query = _read_jsonl(run_dir / "per_query.jsonl")
    todo = [e for e in entries if e.query_id not in answers]
    if not todo:
        print(f"all {len(entries)} queries already answered in {run_dir}")
        return
    print(f"{len(todo)} of {len(entries)} queries to go in {run_dir}\n"
          f"paste the prompt into your model, copy the answer, then press Enter "
          f"(s = skip, q = quit)\n")

    last_answer = None
    for n, entry in enumerate(todo, 1):
        bundle, context, candidates, timing = run_pipeline(entry.query, index, cfg,
                                                           query_id=entry.query_id)
        row = {
            "query_id": entry.query_id,
            "query": entry.query,
            "index_hash": bundle.index_hash,
            "config_hash": bundle.config_hash,
            "prompt_tokens": bundle.n_tokens,
            "context_tokens": context.total_tokens,
            "candidates": [dataclasses.asdict(c) for c in candidates],
            "blocks": [{"block_id": b.block_id, "doc_id": b.doc_id,
                        "start_char": b.start_char, "end_char": b.end_char,
                        "chunk_ids": b.chunk_ids, "scores": b.scores}
                       for b in context.blocks],
            "dropped": context.dropped,
            "timing": timing,
        }
        _append_jsonl(run_dir / "per_query.jsonl", row)
        per_query.append(row)

        sink = deliver_prompt(bundle, cfg, run_dir)
        print(f"[{n}/{len(todo)}] {entry.query_id}  {entry.query[:60]!r}  "
              f"{len(context.blocks)} blocks, {bundle.n_tokens} tokens -> {sink}")
        if sink != "clipboard":
            print(f"  no clipboard: save the answer to "
                  f"{run_dir / 'answers' / (entry.query_id + '.txt')}")

        while True:
            try:
                key = input("  answer ready? [Enter/s/q] ").strip().lower()
            except EOFError:
                key = "q"
            if key == "q":
                _write_timing(run_dir, per_query)
                print(f"stopped; resume with: python rag.py walk <queries> --run {run_dir}")
                return
            if key == "s":
                break
            answer = collect_answer(bundle, cfg, run_dir)
            if not answer or not answer.strip():
                print("  nothing to read; copy the answer first (or s to skip)")
                continue
            # the two ways a human silently poisons an eval set
            if answer.strip() == bundle.text.strip():
                print("  that is the prompt, not an answer; copy the model's reply")
                continue
            if last_answer is not None and answer.strip() == last_answer:
                print("  identical to the previous answer; press Enter again to keep it")
                last_answer = None
                continue
            _append_jsonl(run_dir / "answers.jsonl", {
                "query_id": entry.query_id,
                "answer": answer.strip(),
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            last_answer = answer.strip()
            break

    _write_timing(run_dir, per_query)
    print(f"\ndone: {run_dir}/answers.jsonl")


# --- batched judging -------------------------------------------------------

def _judge_v0(items: list[dict], cfg: Config) -> str:
    """Versioned in the repo so scores are comparable across runs. No config,
    no retrieval scores, no hint about which answer came from where."""
    head = (
        "You are grading answers to questions about a document collection.\n\n"
        "For each item you get the question, a reference answer, and a candidate "
        "answer. Score the candidate against the reference:\n\n"
        "  2 - correct: carries the reference answer's substance, contradicts nothing\n"
        "  1 - partial: some of the reference answer, or correct but incomplete\n"
        "  0 - wrong: contradicts the reference, or does not answer the question\n\n"
        "A candidate that says the information was not available scores 0 unless "
        "the reference says so too. Judge content only: ignore length, style, "
        "formatting and citations. Items are unrelated to each other; do not let "
        "one item's answer inform another's score.\n\n"
        "Return only JSON, one object per item, no prose around it:\n"
        '[{"item": 1, "score": 2, "note": "<= 12 words"}]\n'
    )
    body = []
    for it in items:
        body.append(
            f"<item n=\"{it['item']}\">\n"
            f"<question>\n{it['query']}\n</question>\n"
            f"<reference>\n{it['reference']}\n</reference>\n"
            f"<candidate>\n{it['candidate']}\n</candidate>\n"
            f"</item>"
        )
    return head + "\n" + "\n\n".join(body) + "\n"


JUDGE_TEMPLATES: dict[str, Callable[[list[dict], Config], str]] = {
    "v0": _judge_v0,
}


def assemble_judge_prompt(items: list[dict], cfg: Config = CFG) -> str:
    if cfg.judge_template not in JUDGE_TEMPLATES:
        raise SystemExit(f"unknown judge.template {cfg.judge_template!r}; "
                         f"known: {sorted(JUDGE_TEMPLATES)}")
    return JUDGE_TEMPLATES[cfg.judge_template](items, cfg)


def build_judge_items(entries: list[QueryEntry], run_dir: Path,
                      cfg: Config = CFG) -> tuple[list[dict], list[str]]:
    """Returns (items, query_ids_in_item_order). Order is shuffled: presentation
    position would otherwise identify a config across runs judged in one session."""
    answers = {r["query_id"]: r["answer"] for r in _read_jsonl(run_dir / "answers.jsonl")}
    usable = [e for e in entries if e.query_id in answers and e.ground_truth]
    missing_answer = [e.query_id for e in entries if e.query_id not in answers]
    missing_truth = [e.query_id for e in entries
                     if e.query_id in answers and not e.ground_truth]
    if missing_answer:
        print(f"skipping {len(missing_answer)} queries with no answer yet", file=sys.stderr)
    if missing_truth:
        print(f"skipping {len(missing_truth)} answered queries with no ground truth "
              f"(add a third tab-separated column)", file=sys.stderr)
    if not usable:
        raise SystemExit("nothing to judge: need answers and ground truths")

    order = list(usable)
    random.Random(f"{run_dir.name}:{cfg.judge_template}").shuffle(order)
    items = [{"item": i, "query": e.query, "reference": e.ground_truth,
              "candidate": answers[e.query_id]} for i, e in enumerate(order, 1)]
    return items, [e.query_id for e in order]


def parse_judgements(text: str, n_items: int, cfg: Config = CFG) -> dict[int, dict]:
    """Tolerates markdown fences and prose around the JSON array."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        raise SystemExit("no JSON array found in the pasted text")
    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise SystemExit(f"pasted text is not valid JSON: {e}")

    out: dict[int, dict] = {}
    for row in rows:
        try:
            item, score = int(row["item"]), int(row["score"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"malformed judgement row: {row!r}")
        if not 1 <= item <= n_items:
            raise SystemExit(f"judgement for item {item}, but there are {n_items} items")
        if score not in cfg.judge_scale:
            raise SystemExit(f"item {item}: score {score} is outside {list(cfg.judge_scale)}")
        out[item] = {"score": score, "note": str(row.get("note", ""))[:200]}
    return out


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------

def _cmd_index(args) -> None:
    idx = build_index(CFG, rebuild=args.rebuild)
    m = idx.meta
    print(f"index_hash   {m['index_hash']}")
    print(f"path         {idx.path}")
    print(f"embedder     {CFG.embed_model} @ {CFG.embed_revision} "
          f"(resolved {m.get('embed.resolved_revision') or 'unknown'}), "
          f"dim {m['dim']}, prefix {m['embed.prefix_scheme_id']}")
    print(f"corpus       {m['n_docs']} docs -> {m['n_chunks']} chunks")
    t = m["chunk_tokens"]
    print(f"chunk tokens min {t['min']} / p50 {t['p50']} / mean {t['mean']} / "
          f"p95 {t['p95']} / max {t['max']}; under 64 tokens: {t['under_64_tokens']}")
    o = m["chunk_overlap"]
    print(f"overlap      p50 {o['p50_chars']} chars; "
          f"{o['zero_overlap_boundaries']}/{o['boundaries']} boundaries with none")
    if not re.fullmatch(r"[0-9a-f]{40}", CFG.embed_revision):
        print(f"WARNING embed.revision={CFG.embed_revision!r} is not a pinned commit sha; "
              f"the index hash cannot detect the model changing under you.", file=sys.stderr)


def _cmd_rewrite(args) -> None:
    for r in rewrite_query(args.query, CFG):
        print(json.dumps(dataclasses.asdict(r), ensure_ascii=False))


def _cmd_inspect(args) -> None:
    index = load_index(CFG)
    t0 = time.time()
    rewrites = rewrite_query(args.query, CFG)
    t1 = time.time()
    candidates = retrieve(rewrites, index, CFG)
    t2 = time.time()
    candidates = rerank(args.query, rewrites, candidates, index, CFG)
    t3 = time.time()

    print(f"index    {index.index_hash[:12]}  {index.meta['n_chunks']} chunks "
          f"from {index.meta['n_docs']} docs")
    print(f"rewrite  {CFG.rewrite_mode}: {len(rewrites)} query/queries "
          f"({(t1 - t0) * 1000:.0f} ms)")
    for r in rewrites:
        print(f"   rw{r.rewrite_id}  {r.text!r}")
    print(f"retrieve {CFG.retrieve_mode} top_k={CFG.retrieve_top_k} -> {len(candidates)} candidates "
          f"({(t2 - t1) * 1000:.0f} ms)")
    print(f"rerank   {CFG.rerank_mode} ({(t3 - t2) * 1000:.0f} ms)")
    print(f"{'#':>3}  {'dense':>7}  {'rerank':>8}  {'rw':>4}  chunk")
    for n, c in enumerate(candidates[:args.n], 1):
        chunk = index.get_chunk(c.chunk_id)
        preview = " ".join(chunk.text.split())[:72]
        print(f"{n:>3}  {c.dense_score if c.dense_score is None else round(c.dense_score, 4):>7}  "
              f"{c.rerank_score if c.rerank_score is None else round(c.rerank_score, 4):>8}  "
              f"{','.join(str(i) for i in c.rewrite_ids):>4}  "
              f"{c.doc_id}:{chunk.start_char}-{chunk.end_char}")
        print(f"{'':>27}  {preview!r}")


def _cmd_query(args) -> None:
    index = load_index(CFG)
    bundle, context, candidates, timing = run_pipeline(args.query, index, CFG)
    print(bundle.text)                      # stdout is the prompt and nothing else
    sink = deliver_prompt(bundle, CFG, Path(CFG.runs_root))
    reasons: dict[str, int] = {}
    for _, why in context.dropped:
        reasons[why] = reasons.get(why, 0) + 1
    print(f"query {bundle.query_id} | {len(candidates)} candidates -> {len(context.blocks)} blocks "
          f"| {context.total_tokens} context tokens, {bundle.n_tokens} prompt tokens "
          f"(budget {CFG.context_budget_tokens}) | dropped {reasons or '{}'} | "
          f"{' '.join(f'{k}={v}' for k, v in timing.items())} | index {bundle.index_hash[:12]} "
          f"| config {bundle.config_hash[:12]} | -> {sink}", file=sys.stderr)


def _cmd_walk(args) -> None:
    cfg = dataclasses.replace(CFG, clipboard=not args.no_clipboard)
    entries = load_queries(args.queries)
    index = load_index(cfg)
    if not args.run:
        # answers are expensive; do not let a forgotten --run quietly restart one
        prior = sorted(Path(cfg.runs_root).glob(f"*-{config_hash(cfg)[:8]}*"))
        prior = [d for d in prior if (d / "answers.jsonl").exists()]
        if prior:
            n = len(_read_jsonl(prior[-1] / "answers.jsonl"))
            print(f"note: {prior[-1]} already holds {n} answers for this exact config; "
                  f"pass --run {prior[-1]} to continue it instead of starting over\n")
    run_dir = open_run(cfg, args.run)
    run_generation_loop(entries, index, run_dir, cfg)


def _cmd_judge(args) -> None:
    cfg = dataclasses.replace(CFG, clipboard=not args.no_clipboard)
    entries = load_queries(args.queries)
    run_dir = open_run(cfg, args.run)
    items, query_ids = build_judge_items(entries, run_dir, cfg)
    text = assemble_judge_prompt(items, cfg)

    (run_dir / "judge_prompt.txt").write_text(text, encoding="utf-8")
    (run_dir / "judge_items.json").write_text(
        json.dumps({"template": cfg.judge_template, "order": query_ids}, indent=2) + "\n",
        encoding="utf-8")
    copy, paste = _clipboard() if cfg.clipboard else (None, None)
    if copy is not None:
        copy(text)
    print(f"{len(items)} triples -> {'clipboard' if copy else run_dir / 'judge_prompt.txt'}\n"
          f"paste into a FRESH session, then copy the JSON back")

    try:
        input("  scores copied? [Enter] ")
    except EOFError:
        return
    raw = paste() if paste is not None else None
    if raw is None:
        path = run_dir / "judge_response.txt"
        if not path.exists():
            raise SystemExit(f"no clipboard; save the response to {path} and re-run")
        raw = path.read_text(encoding="utf-8")

    scored = parse_judgements(raw, len(items), cfg)
    rows = [{"query_id": query_ids[i - 1], **scored[i]} for i in sorted(scored)]
    values = [r["score"] for r in rows]
    dist = {str(v): values.count(v) for v in sorted(set(values))}
    out = {
        "judge_template": cfg.judge_template,
        "config_hash": config_hash(cfg),
        "n_items": len(items),
        "n_scored": len(rows),
        "mean": round(sum(values) / len(values), 3) if values else None,
        "distribution": dist,
        "scores": rows,
    }
    (run_dir / "judgements.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    missing = sorted(set(range(1, len(items) + 1)) - set(scored))
    if missing:
        print(f"WARNING no score for items {missing}", file=sys.stderr)
    print(f"mean {out['mean']} over {len(rows)} items, distribution {dist} "
          f"-> {run_dir / 'judgements.json'}")
    if len(rows) < 30:
        print("note: below ~30 items, differences under ~10 points are noise",
              file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="build or validate the index for the current config")
    p_index.add_argument("--rebuild", action="store_true", help="rebuild even if the hash matches")
    p_index.set_defaults(func=_cmd_index)

    p_rw = sub.add_parser("rewrite", help="show the rewrite list for a query")
    p_rw.add_argument("query")
    p_rw.set_defaults(func=_cmd_rewrite)

    p_ins = sub.add_parser("inspect", help="rewrite -> retrieve -> rerank, with scores")
    p_ins.add_argument("query")
    p_ins.add_argument("-n", type=int, default=20, help="how many candidates to print")
    p_ins.set_defaults(func=_cmd_inspect)

    p_q = sub.add_parser("query", help="full pipeline; the prompt goes to stdout")
    p_q.add_argument("query")
    p_q.set_defaults(func=_cmd_query)

    p_w = sub.add_parser("walk", help="one query at a time: prompt out, answer back in")
    p_w.add_argument("queries", help="one query per line, or id<TAB>query[<TAB>ground truth]")
    p_w.add_argument("--run", help="resume an existing run directory")
    p_w.add_argument("--no-clipboard", action="store_true", help="use files instead")
    p_w.set_defaults(func=_cmd_walk)

    p_j = sub.add_parser("judge", help="one batched judge prompt for every collected answer")
    p_j.add_argument("queries")
    p_j.add_argument("--run", required=True, help="the run directory holding answers.jsonl")
    p_j.add_argument("--no-clipboard", action="store_true")
    p_j.set_defaults(func=_cmd_judge)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
