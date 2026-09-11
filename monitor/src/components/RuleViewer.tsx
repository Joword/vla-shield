"use client";

import { useEffect, useState } from "react";
import { useRules, RuleEntry } from "@/hooks/useRules";

const ACTION_BADGE: Record<string, string> = {
  block: "bg-danger/20 text-danger border border-danger/40",
  clamp: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40",
  warn: "bg-blue-500/20 text-blue-400 border border-blue-500/40",
};

const SEVERITY_COLOR: Record<string, string> = {
  critical: "text-red-400",
  high: "text-orange-400",
  medium: "text-yellow-400",
  low: "text-blue-400",
  info: "text-gray-400",
};

function RuleRow({
  rule,
  active,
}: {
  rule: RuleEntry;
  active: boolean;
}) {
  const [expanded, setExpanded] = useState(active);

  useEffect(() => {
    // Currently firing — pop the row open so you don't have to hunt for it.
    if (active) setExpanded(true);
  }, [active]);

  return (
    <li
      className={`text-xs border-b border-gray-700/60 last:border-0 ${
        rule.disabled ? "opacity-40" : ""
      } ${active ? "bg-indigo-500/10" : ""}`}
    >
      <button
        className="w-full flex items-center gap-2 py-1.5 px-1 text-left hover:bg-white/5 transition-colors"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
      >
        {active && <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 flex-shrink-0" />}
        <code className="text-gray-300 flex-1">{rule.rule_id}</code>
        <span
          className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${ACTION_BADGE[rule.action] ?? ""}`}
        >
          {rule.action}
        </span>
        <span className={`text-[10px] uppercase ${SEVERITY_COLOR[rule.severity]}`}>
          {rule.severity}
        </span>
        <span className="text-gray-500 text-[10px]">{expanded ? "▴" : "▾"}</span>
      </button>

      {expanded && (
        <div className="pl-3 pr-1 pb-2 text-gray-500 space-y-1">
          <p>
            <span className="text-gray-400">Trigger: </span>
            {rule.trigger_condition}
          </p>
          <p className="font-mono text-gray-400 text-[10px] break-all">
            {rule.explanation_template}
          </p>
          {rule.hard_block && (
            <p className="text-danger text-[10px] font-semibold">
              ⬛ hard block — cannot be overridden
            </p>
          )}
          {rule.disabled && (
            <p className="text-gray-500 text-[10px]">⏸ rule is disabled</p>
          )}
        </div>
      )}
    </li>
  );
}

interface RuleViewerProps {
  activeRuleIds?: string[];
}

export default function RuleViewer({ activeRuleIds = [] }: RuleViewerProps) {
  const { rules, loading, error } = useRules();
  const [filter, setFilter] = useState<"all" | "block" | "active">("all");

  const displayed = rules.filter((r) => {
    if (filter === "block") return r.action === "block";
    if (filter === "active") return activeRuleIds.includes(r.rule_id); // firing now, not "enabled"
    return true;
  });

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        Active Rule Set
      </h2>

      <div className="flex gap-1 mb-3 text-xs">
        {(["all", "block", "active"] as const).map((f) => (
          <button
            key={f}
            className={`px-2 py-0.5 rounded capitalize transition-colors ${
              filter === f
                ? "bg-indigo-600 text-white"
                : "text-gray-500 hover:text-gray-300"
            }`}
            onClick={() => setFilter(f)}
          >
            {f === "active" ? `active (${activeRuleIds.length})` : f}
          </button>
        ))}
      </div>

      {loading && <p className="text-xs text-gray-500">Loading rules…</p>}
      {error && <p className="text-xs text-danger">{error}</p>}

      {!loading && !error && displayed.length === 0 && (
        <p className="text-xs text-gray-500">No rules match the current filter.</p>
      )}

      <ul>
        {displayed.map((rule) => (
          <RuleRow
            key={rule.rule_id}
            rule={rule}
            active={activeRuleIds.includes(rule.rule_id)}
          />
        ))}
      </ul>

      <p className="mt-2 text-[10px] text-gray-600">
        {rules.length} rules total · {rules.filter((r) => !r.disabled).length} enabled
      </p>
    </div>
  );
}
