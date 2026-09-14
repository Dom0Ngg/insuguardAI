import re
from pathlib import Path
from typing import Any, Dict, List

from backend.agents.base_agent import BaseAgent
from backend.rag.vector_store import PostgresVectorStore, RAGUnavailableError
from backend.services.policy_repository import PolicyRepository


class RAGCoverageAgent(BaseAgent):
    # Cosine similarity is a retrieval signal, not a probability of coverage.
    # These thresholds are intentionally conservative and are only used to
    # describe retrieval quality / decide whether the retrieved evidence is
    # strong enough to support an orientative policy review.
    MIN_USABLE_SCORE = 0.25
    MEDIUM_SCORE = 0.30
    HIGH_SCORE = 0.45

    COVERAGE_TERMS = (
        "we will pay",
        "we will cover",
        "is covered",
        "are covered",
        "covered under this section",
        "cover for",
        "indemnify",
        "indemnity",
        "queda cubierto",
        "está cubierto",
        "estan cubiertos",
        "están cubiertos",
        "cubre",
        "indemnización",
    )
    EXCLUSION_TERMS = (
        "we will not cover",
        "not covered",
        "does not cover",
        "excluded",
        "exclusion",
        "exclusions",
        "shall not be liable",
        "not be liable",
        "no cover",
        "no está cubierto",
        "no esta cubierto",
        "no cubre",
        "exclusión",
        "exclusiones",
        "excluido",
    )
    CONDITION_TERMS = (
        "subject to",
        "provided that",
        "only if",
        "unless",
        "you must",
        "must be",
        "condition",
        "conditions",
        "siempre que",
        "salvo que",
        "condición",
        "condiciones",
        "deberá",
        "debe",
    )

    def __init__(self):
        super().__init__("Agente RAG de coberturas")
        root = Path(__file__).resolve().parents[2]
        self.policy_repository = PolicyRepository()
        self.vector_store = PostgresVectorStore(
            root / "data" / "policies",
            policy_repository=self.policy_repository,
        )

    @staticmethod
    def _page_refs(text: str) -> List[int]:
        return sorted({int(value) for value in re.findall(r"\[P(?:á|a)gina\s+(\d+)\]", text or "", flags=re.I)})

    @staticmethod
    def _sentences(text: str) -> List[str]:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        if not cleaned:
            return []
        return [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", cleaned) if part.strip()]

    @classmethod
    def _matching_evidence(
        cls,
        chunks: List[Dict[str, Any]],
        terms: tuple[str, ...],
        evidence_type: str,
        *,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for chunk in chunks:
            for sentence in cls._sentences(chunk.get("text", "")):
                lowered = sentence.lower()
                matched = next((term for term in terms if term in lowered), None)
                if not matched:
                    continue
                evidence.append(
                    {
                        "type": evidence_type,
                        "chunk_id": chunk.get("chunk_id"),
                        "source": chunk.get("source"),
                        "semantic_score": chunk.get("semantic_score"),
                        "pages": cls._page_refs(chunk.get("text", "")),
                        "matched_term": matched,
                        "text": sentence[:420],
                    }
                )
                if len(evidence) >= limit:
                    return evidence
        return evidence

    @classmethod
    def _retrieval_quality(cls, score: float) -> str:
        if score >= cls.HIGH_SCORE:
            return "alta"
        if score >= cls.MEDIUM_SCORE:
            return "media"
        if score >= cls.MIN_USABLE_SCORE:
            return "baja"
        return "insuficiente"

    @classmethod
    def _review_result(
        cls,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not chunks:
            return {
                "coverage_result": "Evidencia insuficiente",
                "retrieval_quality": "insuficiente",
                "top_semantic_score": None,
                "coverage_evidence": [],
                "exclusion_evidence": [],
                "condition_evidence": [],
                "recommended_action": "Reformular la consulta o revisar el clausulado manualmente.",
            }

        best_score = max(float(chunk.get("semantic_score") or 0.0) for chunk in chunks)
        quality = cls._retrieval_quality(best_score)
        # Only classify coverage from evidence reasonably close to the best hit.
        # Low-scoring tail chunks remain visible as sources but do not drive the verdict.
        evidence_floor = max(cls.MIN_USABLE_SCORE, best_score - 0.12)
        usable = [
            chunk
            for chunk in chunks
            if float(chunk.get("semantic_score") or 0.0) >= evidence_floor
        ]

        coverage = cls._matching_evidence(usable, cls.COVERAGE_TERMS, "coverage")
        exclusions = cls._matching_evidence(usable, cls.EXCLUSION_TERMS, "exclusion")
        conditions = cls._matching_evidence(usable, cls.CONDITION_TERMS, "condition")

        if best_score < cls.MIN_USABLE_SCORE:
            result = "Evidencia insuficiente"
            action = "La similitud recuperada es demasiado baja; reformula la consulta o revisa la póliza completa."
        elif exclusions and coverage:
            result = "Revisión manual requerida"
            action = "Hay indicios de cobertura y de exclusión/condición; contrasta ambas cláusulas antes de resolver."
        elif exclusions:
            result = "Posible exclusión"
            action = "Revisa la exclusión recuperada y sus condiciones antes de concluir que el siniestro no está cubierto."
        elif coverage:
            result = "Probablemente cubierto"
            action = "Existe evidencia orientativa de cobertura; confirma límites, franquicias, exclusiones y condiciones particulares."
        else:
            result = "Revisión manual requerida"
            action = "Hay fragmentos relacionados, pero no contienen evidencia explícita suficiente para clasificar la cobertura."

        return {
            "coverage_result": result,
            "retrieval_quality": quality,
            "top_semantic_score": round(best_score, 4),
            "coverage_evidence": coverage,
            "exclusion_evidence": exclusions,
            "condition_evidence": conditions,
            "recommended_action": action,
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        policy_id = input_data.get("policy_id")
        query = (input_data.get("coverage_query") or "").strip()
        if not policy_id or not query:
            return {
                "status": "success",
                "output": {
                    "coverage_result": "No evaluado",
                    "rag_answer": (
                        "Kaggle no incluye la póliza contractual ni una descripción de cobertura. "
                        "Indica policy_id y coverage_query para ejecutar el RAG semántico."
                    ),
                    "policy_id": policy_id,
                    "sources": [],
                },
            }

        policy = self.policy_repository.get_policy(policy_id)
        if policy is None:
            return {
                "status": "success",
                "output": {
                    "coverage_result": "No evaluado",
                    "rag_answer": f"No existe la póliza {policy_id} en data/policies/manifest.json.",
                    "policy_id": policy_id,
                    "sources": [],
                },
            }

        try:
            top_k = int(input_data.get("top_k") or 5)
            top_k = min(max(top_k, 1), 10)
            chunks = self.vector_store.search(query=query, policy_id=policy_id, top_k=top_k)
        except RAGUnavailableError as exc:
            return {
                "status": "error",
                "output": {
                    "coverage_result": "RAG no disponible",
                    "rag_answer": str(exc),
                    "policy_id": policy_id,
                    "policy": policy,
                    "query": query,
                    "sources": [],
                    "retrieval_method": "PostgreSQL + pgvector + Sentence Transformers",
                },
            }

        review = self._review_result(chunks)
        sources = [
            {
                "source": chunk.get("source"),
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "semantic_score": chunk.get("semantic_score"),
                "vector_score": chunk.get("semantic_score"),
                "hybrid_score": chunk.get("semantic_score"),
                "pages": self._page_refs(chunk.get("text", "")),
                "relevance_level": self._retrieval_quality(float(chunk.get("semantic_score") or 0.0)),
                "snippet": chunk.get("text", "")[:600],
                "exact_text": chunk.get("text", ""),
                "page": (self._page_refs(chunk.get("text", "")) or [None])[0],
            }
            for chunk in chunks
        ]
        if chunks and review["retrieval_quality"] != "insuficiente":
            answer = (
                "Se han recuperado fragmentos únicamente de la póliza seleccionada y se ha realizado "
                "una revisión orientativa de cobertura. La similitud semántica mide relevancia de búsqueda, "
                "no probabilidad de cobertura; la resolución final corresponde al especialista."
            )
        elif chunks:
            answer = (
                "Se recuperaron fragmentos, pero su similitud semántica es insuficiente para sostener una "
                "interpretación de cobertura. Reformula la consulta o revisa el clausulado completo."
            )
        else:
            answer = "No se localizaron fragmentos semánticamente relevantes dentro de la póliza seleccionada."
        return {
            "status": "success",
            "output": {
                **review,
                "rag_answer": answer,
                "policy_id": policy_id,
                "policy": policy,
                "query": query,
                "sources": sources,
                "score_notice": (
                    "semantic_score es similitud coseno del retrieval y no representa una probabilidad "
                    "de que el siniestro esté cubierto."
                ),
                "retrieval_method": (
                    "Sentence Transformers multilingüe + PostgreSQL/pgvector cosine similarity; "
                    "filtrado estricto por policy_id"
                ),
            },
        }
