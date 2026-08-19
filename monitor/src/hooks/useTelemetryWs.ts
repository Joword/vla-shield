import { useEffect, useRef } from "react";
import { useTelemetryStore } from "@/store/telemetry";

const BACKOFF_MS = [500, 1000, 2000, 4000, 8000];

export function useTelemetryWs(robotId: string) {
  const update = useTelemetryStore((s) => s.update);
  const setWsStatus = useTelemetryStore((s) => s.setWsStatus);

  const updateRef = useRef(update);
  updateRef.current = update;

  useEffect(() => {
    let closed = false;
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    const connect = () => {
      if (closed) return;
      setWsStatus(attempt === 0 ? "connecting" : "reconnecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${protocol}//${window.location.host}/ws/telemetry/${robotId}`;
      const socket = new WebSocket(url);
      ws = socket;

      socket.onopen = () => {
        if (closed) return;
        attempt = 0;
        setWsStatus("connected");
      };

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          updateRef.current(msg);
        } catch {
          // ignore malformed frames
        }
      };

      socket.onclose = () => {
        if (closed) return;
        setWsStatus("reconnecting");
        const delay = BACKOFF_MS[Math.min(attempt, BACKOFF_MS.length - 1)];
        attempt += 1;
        timer = setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
      } else if (ws) {
        ws.onclose = null;
        ws.close();
      }
      setWsStatus("disconnected");
    };
  }, [robotId, setWsStatus]);
}
