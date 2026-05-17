"""Rule registry that turns ``dataset/ontology/rules_*.json`` into a runtime engine.

The runtime evaluator consults this registry to:
  * decide whether a triggered ontology id should ``block``, ``clamp`` or ``warn``;
  * pull severity, ``hard_block`` and a human-readable ``explanation_template``;
  * fill `{placeholder}` slots in the template with live values supplied by the
    detector code (e.g. joint name, requested vs limit velocity).

The registry is intentionally side-effect free and safe to reuse across requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Rule:
    rule_id: str
    trigger_condition: str
    action: str
    severity: str
    hard_block: bool
    explanation_template: str
    threshold: dict[str, Any]
    disabled: bool

    def render(self, **kwargs: Any) -> str:
        """Best-effort template render; missing keys degrade gracefully."""
        try:
            return self.explanation_template.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return self.explanation_template


class RuleRegistry:
    """In-memory registry indexed by ``rule_id``."""

    def __init__(self, rules: Iterable[Rule]) -> None:
        self._rules: dict[str, Rule] = {r.rule_id: r for r in rules if not r.disabled}

    def __len__(self) -> int:
        return len(self._rules)

    def get(self, rule_id: str) -> Rule | None:
        return self._rules.get(rule_id)

    def render(self, rule_id: str, fallback: str = "", /, **kwargs: Any) -> str:
        rule = self._rules.get(rule_id)
        if rule is None:
            return fallback
        return rule.render(**kwargs)

    def is_hard_block(self, rule_id: str) -> bool:
        rule = self._rules.get(rule_id)
        return bool(rule and (rule.action == "block" or rule.hard_block))

    def severity(self, rule_id: str) -> str:
        rule = self._rules.get(rule_id)
        return rule.severity if rule else "medium"

    @classmethod
    def load(cls, ontology_dir: Path) -> "RuleRegistry":
        entries: list[Rule] = []
        for name in ("rules_physical.json", "rules_semantic.json"):
            path = ontology_dir / name
            if not path.exists():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            for item in raw:
                entries.append(
                    Rule(
                        rule_id=item["rule_id"],
                        trigger_condition=item.get("trigger_condition", ""),
                        action=item.get("action", "warn"),
                        severity=item.get("severity", "medium"),
                        hard_block=bool(item.get("hard_block", False)),
                        explanation_template=item.get("explanation_template", ""),
                        threshold=item.get("threshold", {}),
                        disabled=bool(item.get("disabled", False)),
                    )
                )
        return cls(entries)
