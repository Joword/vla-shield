export type DecisionKind = "PASS" | "CLAMP" | "WARN" | "BLOCK";

export interface DecisionStyle {
  text: string;
  badge: string;
  bar: string;
  hex: string;
}

/** Colors for the badge / bar / 3D arm. Unknown strings get PASS green. */
export function decisionStyle(decision: string): DecisionStyle {
  switch (decision) {
    case "BLOCK":
      return {
        text: "text-danger",
        badge: "bg-red-900/40 text-danger",
        bar: "bg-danger",
        hex: "#ef4444",
      };
    case "CLAMP":
      return {
        text: "text-warning",
        badge: "bg-yellow-900/40 text-yellow-300",
        bar: "bg-warning",
        hex: "#f59e0b",
      };
    case "WARN":
      return {
        text: "text-blue-400",
        badge: "bg-blue-900/40 text-blue-300",
        bar: "bg-blue-400",
        hex: "#60a5fa",
      };
    default:
      return {
        text: "text-safe",
        badge: "bg-green-900/40 text-safe",
        bar: "bg-safe",
        hex: "#22c55e",
      };
  }
}

/** Sidebar heading. Keep these in sync with the labels on the panel. */
export function reasonsTitle(decision: string): string {
  switch (decision) {
    case "BLOCK":
      return "Block Reasons";
    case "CLAMP":
      return "Clamp Reasons";
    case "WARN":
      return "Warnings";
    default:
      return "Triggered Rules";
  }
}
