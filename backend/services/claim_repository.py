import json
from pathlib import Path
from typing import Any, Dict, List
import shutil


class ClaimRepository:
    ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".txt"}

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.claims_dir = self.project_root / "data" / "claims"

    def list_claims(self) -> List[Dict[str, Any]]:
        rows = []
        if not self.claims_dir.exists():
            return rows
        for claim_file in sorted(self.claims_dir.glob("*/claim.json")):
            try:
                claim = json.loads(claim_file.read_text(encoding="utf-8"))
                rows.append({
                    "claim_id": claim.get("claim_id"),
                    "source_row": claim.get("source", {}).get("row_index"),
                    "ground_truth": claim.get("ground_truth", {}).get("FraudFound_P"),
                    "policy_id": claim.get("policy_id"),
                    "documents": claim.get("documents", []),
                })
            except Exception:
                continue
        return rows

    def get_claim(self, claim_id: str) -> Dict[str, Any]:
        safe_id = Path(claim_id).name
        path = self.claims_dir / safe_id / "claim.json"
        if not path.exists():
            raise FileNotFoundError(claim_id)
        return json.loads(path.read_text(encoding="utf-8"))


    def save_claim(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        claim_id = str(claim.get("claim_id") or "").strip()
        if not claim_id:
            raise ValueError("claim_id is required")
        safe_id = Path(claim_id).name
        claim_dir = self.claims_dir / safe_id
        claim_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(claim)
        payload["claim_id"] = safe_id
        claim_path = claim_dir / "claim.json"
        claim_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload

    @classmethod
    def get_supported_document_extensions(cls) -> list[str]:
        """Formats accepted as original claim evidence for supervisor review."""
        return sorted(cls.ALLOWED_DOCUMENT_EXTENSIONS)

    def documents_dir(self, claim_id: str) -> Path:
        return self.claims_dir / Path(claim_id).name / "documents"

    def attach_document(self, claim_id: str, source_path: Path, original_name: str) -> Dict[str, Any]:
        claim = self.get_claim(claim_id)
        safe_name = Path(original_name).name
        if not safe_name:
            raise ValueError("Invalid document name")
        documents_dir = self.documents_dir(claim_id)
        documents_dir.mkdir(parents=True, exist_ok=True)
        destination = documents_dir / safe_name
        shutil.copyfile(source_path, destination)
        documents = list(claim.get("documents", []))
        if safe_name not in documents:
            documents.append(safe_name)
        claim["documents"] = documents
        claim_path = self.claims_dir / Path(claim_id).name / "claim.json"
        claim_path.write_text(json.dumps(claim, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"claim_id": claim_id, "document": safe_name, "documents": documents}
