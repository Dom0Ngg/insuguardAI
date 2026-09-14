import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class PolicyRepository:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]
        self.policies_dir = self.project_root / "data" / "policies"
        self.manifest_path = self.policies_dir / "manifest.json"

    def list_policies(self) -> List[Dict[str, Any]]:
        if not self.manifest_path.exists():
            return []
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return data.get("policies", [])

    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        for policy in self.list_policies():
            if policy.get("policy_id") == policy_id:
                return policy
        return None
