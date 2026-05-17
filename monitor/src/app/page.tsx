"use client";

import { useState } from "react";
import Header from "@/components/Header";
import RiskGauge from "@/components/RiskGauge";
import EventTimeline from "@/components/EventTimeline";
import SceneView from "@/components/SceneView";
import WhyBlocked from "@/components/WhyBlocked";
import LatencyChart from "@/components/LatencyChart";
import RuleViewer from "@/components/RuleViewer";
import { useTelemetryStore } from "@/store/telemetry";
import { useTelemetryWs } from "@/hooks/useTelemetryWs";

export default function MonitorPage() {
  const [robotId, setRobotId] = useState("ur5e-lab-01");
  useTelemetryWs(robotId);
  const { risk, decision, ontologyIds, ontologyDetails } = useTelemetryStore();

  return (
    <div className="flex flex-col h-screen">
      <Header robotId={robotId} onRobotIdChange={setRobotId} />
      <div className="flex flex-1 overflow-hidden">
        <div className="flex-1 min-w-0">
          <SceneView />
        </div>
        <aside className="w-96 bg-panel border-l border-gray-700 flex flex-col overflow-y-auto">
          <RiskGauge score={risk} decision={decision} />
          <WhyBlocked ontologyIds={ontologyIds} ontologyDetails={ontologyDetails} />
          <LatencyChart />
          <RuleViewer activeRuleIds={ontologyIds} />
          <EventTimeline robotId={robotId} />
        </aside>
      </div>
    </div>
  );
}
