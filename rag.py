"""On-device RAG testbed -- phase 0 spike.

One file, hardcoded config. All seven stage functions exist with their real
names and contracts; stages 3-7 are unimplemented stubs so that filling them in
is addition, not signature change.

Implemented so far:
  stage 1  build_index / chunk_document
  stage 2  rewrite_query

Retrieval is local and generation is manual: nothing in this file may ever call
a generation API.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
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
    chunk_size: int = 512            # tokens, counted by the embedder's tokenizer
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

    context_top_n: int = 8
    context_budget_tokens: int = 6000
    context_neighbours: int = 0
    context_dedup: str = "exact"
    context_order: str = "score_desc"

    prompt_template: str = "v0"
    prompt_delimiters: str = "xml"
    prompt_question_position: str = "last"
    prompt_show_scores: bool = False

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


def make_embedder(cfg: Config):
    if cfg.embed_model not in EMBEDDERS:
        raise SystemExit(
            f"unknown embed.model {cfg.embed_model!r}; "
            f"known: {sorted(EMBEDDERS)} (add an entry to EMBEDDERS)"
        )
    return EMBEDDERS[cfg.embed_model](cfg)


def embed_text(chunk: Chunk) -> str:
    """What actually gets embedded. Prefixes are the adapter's business."""
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


@dataclass
class Index:
    path: Path
    meta: dict
    docs: dict[str, Document]
    chunks: list[Chunk]
    matrix: np.ndarray                 # [n_chunks, dim], L2-normalised, row i <-> chunks[i]
    _by_id: dict[str, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self._by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}

    @property
    def index_hash(self) -> str:
        return self.meta["index_hash"]

    def get_chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[self._by_id[chunk_id]]

    def get_doc(self, doc_id: str) -> Document:
        return self.docs[doc_id]

    def row_of(self, chunk_id: str) -> int:
        return self._by_id[chunk_id]


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
        vecs[start:start + len(batch)] = adapter.encode_documents([embed_text(c) for c in batch])
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
# stages 3-7: names and signatures locked, bodies not written yet
# --------------------------------------------------------------------------

def retrieve(rewrites: list[Rewrite], index: Index, cfg: Config = CFG) -> list[Candidate]:
    raise NotImplementedError("stage 3: embedding & retrieval")


def rerank(query: str, candidates: list[Candidate], cfg: Config = CFG) -> list[Candidate]:
    """Scores every candidate and returns every candidate; truncation is stage 5."""
    raise NotImplementedError("stage 4: reranking")


def build_context(candidates: list[Candidate], index: Index, cfg: Config = CFG) -> Context:
    raise NotImplementedError("stage 5: context construction")


def assemble_prompt(query: str, context: Context, cfg: Config = CFG) -> PromptBundle:
    raise NotImplementedError("stage 6: prompt assembly")


def deliver_prompt(bundle: PromptBundle, cfg: Config = CFG) -> None:
    """Clipboard + status line. Never calls a generation API; nothing here ever will."""
    raise NotImplementedError("stage 7: manual generation loop")


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


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="build or validate the index for the current config")
    p_index.add_argument("--rebuild", action="store_true", help="rebuild even if the hash matches")
    p_index.set_defaults(func=_cmd_index)

    p_rw = sub.add_parser("rewrite", help="show the rewrite list for a query")
    p_rw.add_argument("query")
    p_rw.set_defaults(func=_cmd_rewrite)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
