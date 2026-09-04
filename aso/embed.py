"""Frozen text embeddings, computed once and cached.

No gradient ever flows through this module. That is the whole reason CPU
training is viable: the transformer runs at scrape time, not at train time.

Falls back to a deterministic character-trigram hash when sentence-transformers
is absent, so the pipeline runs end to end before you download a model.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

DIM = 384
# all-MiniLM-L6-v2: 384-d, ~90MB, English. The multilingual e5 models are five
# times the download for no gain on a US storefront. Override with ASO_EMBED_MODEL.
MODEL = os.environ.get("ASO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CACHE = Path(os.environ.get("ASO_EMBED_CACHE",
                            Path(__file__).resolve().parent.parent / "data" / "emb"))


def _trigrams(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return [t[i:i + 3] for i in range(max(len(t) - 2, 0))] or [t]


class HashingEncoder:
    """Crude but real lexical similarity. Deterministic, dependency-free, instant.
    Good enough to exercise the pipeline; replace before trusting kw_cosine."""

    name = "hashing-v1"

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype="float32")
        for i, t in enumerate(texts):
            for g in _trigrams(t):
                h = hashlib.blake2b(g.encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % DIM
                sign = 1.0 if h[4] & 1 else -1.0
                out[i, idx] += sign
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-9, None)


class SentenceEncoder:
    """Only the e5 family wants the 'query: ' / 'passage: ' prefixes; feeding
    them to a MiniLM model just adds noise, so they are dropped there."""

    def __init__(self, model_name: str = MODEL):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model_name)
        self.name = model_name
        self.wants_prefix = "e5" in model_name.lower()

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.m.encode(texts, normalize_embeddings=True,
                          batch_size=32, show_progress_bar=False)
        return np.asarray(v, dtype="float32")


_ENCODER = None


def encoder():
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = SentenceEncoder()
        except Exception:                                # noqa: BLE001
            _ENCODER = HashingEncoder()
    return _ENCODER


def _key(text: str, prefix: str) -> str:
    return hashlib.blake2b((prefix + (text or "")).encode(), digest_size=16).hexdigest()


def embed(texts: list[str], prefix: str = "") -> np.ndarray:
    """Cache per text, so re-scraping an unchanged listing costs nothing."""
    CACHE.mkdir(parents=True, exist_ok=True)
    enc = encoder()
    if not getattr(enc, "wants_prefix", False):
        prefix = ""
    tag = re.sub(r"[^a-z0-9]+", "-", enc.name.lower())
    out = np.zeros((len(texts), DIM), dtype="float32")
    missing_idx, missing_txt = [], []
    for i, t in enumerate(texts):
        f = CACHE / f"{tag}-{_key(t, prefix)}.npy"
        if f.exists():
            out[i] = np.load(f)
        else:
            missing_idx.append(i)
            missing_txt.append(prefix + (t or ""))
    if missing_txt:
        fresh = enc.encode(missing_txt)
        for j, i in enumerate(missing_idx):
            out[i] = fresh[j]
            np.save(CACHE / f"{tag}-{_key(texts[i], prefix)}.npy", fresh[j])
    return out


def keyword_vec(kw: str) -> np.ndarray:
    return embed([kw], "query: ")[0]


def app_vec(title: str, short_desc: str, description: str = "") -> np.ndarray:
    """Title and short description alone cannot separate "use your phone as a
    pocket mirror" from "cast your screen to a TV" - both are titled
    "...Mirror...". The opening of the description is where the two meanings
    actually differ, so it is part of the vector."""
    text = f"{title}. {short_desc}".strip()
    if description:
        text += " " + " ".join(description.split())[:240]
    return embed([text], "passage: ")[0]
