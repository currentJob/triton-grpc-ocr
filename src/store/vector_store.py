"""
Vector Store — OCR 텍스트를 벡터화하여 유사 문서 검색 (RAG 핵심 컴포넌트).

학습 포인트:
  - TF-IDF 벡터화 + 코사인 유사도 = 가장 기초적인 RAG
  - 한국어는 문자 n-gram(2~3자) 토크나이징이 효과적
  - 실제 서비스라면 sentence-transformers + Chroma/Faiss로 대체

RAG (Retrieval-Augmented Generation) 흐름:
  사용자 쿼리
    → VectorStore.search()  : 유사 OCR 결과 k개 검색
    → Retrieved context     : LLM 프롬프트에 주입
    → LLM 응답              : 근거 있는 답변 생성
"""
from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass


@dataclass
class SearchResult:
    id:         str
    text:       str
    score:      float           # 코사인 유사도 [0, 1]
    metadata:   dict


class VectorStore:
    def __init__(self):
        self._docs:   list[dict]             = []   # {id, text, metadata}
        self._tfidf:  list[dict[str, float]] = []   # 문서별 TF-IDF 벡터
        self._idf:    dict[str, float]       = {}   # 전체 IDF
        self._dirty:  bool                   = False

    # ── 문서 추가 ─────────────────────────────────────────────────────────────

    def add(self, text: str, metadata: dict | None = None) -> str:
        doc_id = str(uuid.uuid4())
        self._docs.append({"id": doc_id, "text": text, "metadata": metadata or {}})
        self._dirty = True
        return doc_id

    def add_many(self, texts: list[str], metadatas: list[dict] | None = None) -> list[str]:
        ids = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            ids.append(self.add(text, meta))
        return ids

    # ── 검색 ──────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        if not self._docs:
            return []
        if self._dirty:
            self._build_index()

        q_vec  = self._tfidf_vector(query)
        scores = [(i, self._cosine(q_vec, d_vec)) for i, d_vec in enumerate(self._tfidf)]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:k]:
            if score < 1e-9:
                break
            doc = self._docs[idx]
            results.append(SearchResult(
                id=doc["id"], text=doc["text"], score=round(score, 4),
                metadata=doc["metadata"],
            ))
        return results

    def size(self) -> int:
        return len(self._docs)

    # ── TF-IDF 인덱스 빌드 ───────────────────────────────────────────────────

    def _build_index(self) -> None:
        n = len(self._docs)
        # DF (문서 빈도) 계산
        df: dict[str, int] = Counter()
        tokenized = [self._tokenize(d["text"]) for d in self._docs]
        for tokens in tokenized:
            for term in set(tokens):
                df[term] += 1

        # IDF = log((N + 1) / (df + 1)) + 1  (smoothed)
        self._idf = {t: math.log((n + 1) / (cnt + 1)) + 1.0 for t, cnt in df.items()}

        # TF-IDF 벡터 (정규화)
        self._tfidf = [self._tfidf_vector(d["text"], tokenized[i]) for i, d in enumerate(self._docs)]
        self._dirty = False

    def _tfidf_vector(self, text: str, tokens: list[str] | None = None) -> dict[str, float]:
        if tokens is None:
            tokens = self._tokenize(text)
        tf   = Counter(tokens)
        total = len(tokens) or 1
        vec  = {t: (cnt / total) * self._idf.get(t, 1.0) for t, cnt in tf.items()}
        # L2 정규화
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    # ── 토크나이저 (한국어 문자 n-gram) ─────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        한국어·영어 혼용 텍스트용 토크나이저.
        - 한국어: 2-gram (어절 단위보다 n-gram이 형태소 변화에 강함)
        - 영어  : 소문자 단어 토큰
        """
        text = re.sub(r"\s+", " ", text.strip())
        tokens: list[str] = []
        # 영어 단어
        tokens += re.findall(r"[a-z]+", text.lower())
        # 한국어 문자 n-gram (2, 3)
        korean = re.sub(r"[^가-힣]", "", text)
        for n in (2, 3):
            tokens += [korean[i:i+n] for i in range(len(korean) - n + 1)]
        return tokens

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        return sum(a.get(t, 0.0) * v for t, v in b.items())
