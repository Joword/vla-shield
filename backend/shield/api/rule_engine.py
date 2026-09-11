"""dataset/ontology/rules_*.json as a lookup.

Evaluator asks us: block vs clamp vs warn, severity, hard_block, and the
explanation_template with {placeholders} filled in. No I/O after load —
safe to reuse across requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Rule:
    """One row from rules_physical.json / rules_semantic.json."""
    rule_id: str
    trigger_condition: str
    action: str
    severity: str
    hard_block: bool
    explanation_template: str
    threshold: dict[str, Any]
    disabled: bool

    def render(self, **kwargs: Any) -> str:
        """Fill {placeholders}. Missing keys just return the raw template."""
        try:
            return self.explanation_template.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            return self.explanation_template


class RuleRegistry:
    """In-memory rules, keyed by rule_id. Disabled rules are dropped at load."""

    def __init__(self, rules: Iterable[Rule]) -> None:
        self._rules: dict[str, Rule] = {r.rule_id: r for r in rules if not r.disabled}

    def __len__(self) -> int:
        return len(self._rules)

    def get(self, rule_id: str) -> Rule | None:
        """None if the id isn't loaded (or was disabled)."""
        return self._rules.get(rule_id)

    def render(self, rule_id: str, fallback: str = "", /, **kwargs: Any) -> str:
        """Fill the explanation_template, or return fallback if the id is unknown."""
        rule = self._rules.get(rule_id)
        if rule is None:
            return fallback
        return rule.render(**kwargs)

    def is_hard_block(self, rule_id: str) -> bool:
        """True when action=block or hard_block is set."""
        rule = self._rules.get(rule_id)
        return bool(rule and (rule.action == "block" or rule.hard_block))

    def severity(self, rule_id: str) -> str:
        """info/low/medium/high/critical. Unknown ids → medium."""
        rule = self._rules.get(rule_id)
        return rule.severity if rule else "medium"

    def decide(self, rule_ids: Iterable[str]) -> str:
        """One verdict from many rule ids.

        Rank is block > clamp > warn > PASS. Unknown ids fail closed (BLOCK).
        Don't flatten clamp/warn into BLOCK just because something fired.
        """
        rank = 0
        for rid in rule_ids:
            rule = self._rules.get(rid)
            if rule is None:
                rank = max(rank, 3)
                continue
            action = "block" if rule.hard_block else rule.action
            rank = max(rank, {"block": 3, "clamp": 2, "warn": 1}.get(action, 3))
        if rank >= 3:
            return "BLOCK"
        if rank == 2:
            return "CLAMP"
        if rank == 1:
            return "WARN"
        return "PASS"

    @classmethod
    def load(cls, ontology_dir: Path) -> "RuleRegistry":
        """Read rules_physical.json + rules_semantic.json from ontology_dir."""
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
