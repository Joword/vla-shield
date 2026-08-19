"use client";

import { useEffect, useMemo, useState } from "react";

export interface RuleEntry {
  rule_id: string;
  trigger_condition: string;
  action: "block" | "clamp" | "warn";
  severity: "info" | "low" | "medium" | "high" | "critical";
  hard_block: boolean;
  explanation_template: string;
  disabled: boolean;
}

export function useRules() {
  const [rules, setRules] = useState<RuleEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchRules() {
      try {
        const resp = await fetch("/v1/rules");
        const payload: { rules: RuleEntry[] } = await resp.json();
        if (!cancelled) {
          setRules(payload.rules ?? []);
        }
      } catch (e) {
        if (!cancelled) setError(String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    fetchRules();
    return () => {
      cancelled = true;
    };
  }, []);

  const byId = useMemo(() => {
    const map = new Map<string, RuleEntry>();
    for (const rule of rules) {
      map.set(rule.rule_id, rule);
    }
    return map;
  }, [rules]);

  return { rules, byId, loading, error };
}
