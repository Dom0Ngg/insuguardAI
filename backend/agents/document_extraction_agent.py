from pathlib import Path
from typing import Any, Dict, List
import mimetypes

from backend.agents.base_agent import BaseAgent


class DocumentExtractionAgent(BaseAgent):
    """Inventory optional customer attachments as original case evidence.

    The agent records file availability and metadata so the supervisor can open
    the original evidence from the case view. Policy-document extraction is a
    separate responsibility of the RAG indexing pipeline.
    """

    def __init__(self):
        super().__init__("Agente documental")
        self.project_root = Path(__file__).resolve().parents[2]

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        claim_id = Path(str(input_data.get("claim_id", ""))).name
        documents: List[str] = input_data.get("documents", []) or []
        documents_dir = self.project_root / "data" / "claims" / claim_id / "documents"

        if not documents:
            return {
                "status": "success",
                "output": {
                    "documents_requested": [],
                    "documents_available": [],
                    "documents_missing": [],
                    "document_files": [],
                    "content_processing": "disabled",
                    "ocr_used": False,
                    "note": (
                        "No hay documentos adjuntos. Los documentos del cliente son opcionales "
                        "y el análisis de fraude puede continuar con los datos estructurados."
                    ),
                },
            }

        available: List[str] = []
        missing: List[str] = []
        files: List[Dict[str, Any]] = []
        for raw_name in documents:
            name = Path(str(raw_name)).name
            path = (documents_dir / name).resolve()
            if path.exists() and path.is_file() and path.parent == documents_dir.resolve():
                available.append(name)
                mime_type, _ = mimetypes.guess_type(name)
                files.append(
                    {
                        "document_name": name,
                        "mime_type": mime_type or "application/octet-stream",
                        "size_bytes": path.stat().st_size,
                        "status": "available",
                    }
                )
            else:
                missing.append(name)
                files.append(
                    {
                        "document_name": name,
                        "mime_type": None,
                        "size_bytes": None,
                        "status": "missing",
                    }
                )

        return {
            "status": "success",
            "output": {
                "documents_requested": documents,
                "documents_available": available,
                "documents_missing": missing,
                "document_files": files,
                "content_processing": "disabled",
                "ocr_used": False,
                "note": (
                    "Los adjuntos del cliente se conservan como evidencia original para revisión del supervisor."
                ),
            },
        }
