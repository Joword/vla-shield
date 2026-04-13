import { useEffect, useRef } from "react";
import { useTelemetryStore } from "@/store/telemetry";

export function useTelemetryWs(robotId: string) {
  const update = useTelemetryStore((s) => s.update);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/telemetry/${robotId}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        update(msg);
      } catch {
        // ignore malformed frames
      }
    };

    ws.onclose = () => {
      setTimeout(() => {
        wsRef.current = null;
      }, 2000);
    };

    return () => {
      ws.close();
    };
  }, [robotId, update]);
}
