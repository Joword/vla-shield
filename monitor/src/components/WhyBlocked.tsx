"use client";

import { useRules } from "@/hooks/useRules";
import { reasonsTitle } from "@/lib/decision";

const SEVERITY_DOT: Record<string, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-yellow-400",
  low: "bg-blue-400",
  info: "bg-gray-400",
};

interface WhyBlockedProps {
  ontologyIds: string[];
  ontologyDetails?: Record<string, string>;
  decision?: string;
}

export default function WhyBlocked({
  ontologyIds,
  ontologyDetails = {},
  decision = "BLOCK",
}: WhyBlockedProps) {
  const { byId } = useRules();

  if (ontologyIds.length === 0) return null;

  return (
    <div className="p-4 border-b border-gray-700">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-2">
        {reasonsTitle(decision)}
        <span className="ml-2 px-1.5 py-0.5 rounded bg-danger/20 text-danger text-[10px] font-bold">
          {ontologyIds.length}
        </span>
      </h2>
      <ul className="space-y-2">
        {ontologyIds.map((oid) => {
          // oid is a rule id when the catalog loaded; otherwise we still show the raw id.
          const rule = byId.get(oid);
          const detail = ontologyDetails[oid];
          const dot = SEVERITY_DOT[rule?.severity ?? ""] ?? "bg-danger";
          const trigger = rule?.trigger_condition;
          const action = rule?.action;
          return (
            <li key={oid} className="flex flex-col gap-0.5">
              <div className="flex items-center gap-2 text-sm">
                <span className={`w-2 h-2 rounded-full ${dot} flex-shrink-0`} />
                <code className="text-gray-200 font-semibold">{oid}</code>
                {action && (
                  <span className="text-[10px] uppercase text-gray-500">{action}</span>
                )}
              </div>
              {trigger && (
                <p className="pl-4 text-[10px] text-gray-500 font-mono">
                  trigger: {trigger}
                </p>
              )}
              {detail && (
                <p className="pl-4 text-xs text-yellow-300/80 font-mono break-all leading-relaxed">
                  {detail}
                </p>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
