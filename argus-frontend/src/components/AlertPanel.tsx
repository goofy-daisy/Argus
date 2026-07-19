import React, { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BellRing, X, AlertTriangle, Info } from "lucide-react";
import { WS_BASE } from "../api/client";
import { LiveAlert } from "../types";

const MAX_ALERTS = 200;
const PAGE = 20;

function severityColor(s: string) {
  if (s === "high") return "#ef4444";
  if (s === "medium") return "#f59e0b";
  return "#10b981";
}

function SeverityIcon({ s }: { s: string }) {
  if (s === "high") return <AlertTriangle size={12} className="text-red-400 shrink-0" />;
  if (s === "medium") return <AlertTriangle size={12} className="text-amber-400 shrink-0" />;
  return <Info size={12} className="text-emerald-400 shrink-0" />;
}

export default function AlertPanel() {
  const [alerts, setAlerts] = useState<LiveAlert[]>([]);
  const [visible, setVisible] = useState(PAGE);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(`${WS_BASE}/ws/alerts`);
      wsRef.current = ws;
      ws.onmessage = (e) => {
        try {
          const payload = JSON.parse(e.data) as LiveAlert;
          setAlerts((prev) => [payload, ...prev].slice(0, MAX_ALERTS));
        } catch {}
      };
      ws.onclose = () => setTimeout(connect, 3000);
      ws.onerror = () => ws.close();
    };
    connect();
    return () => wsRef.current?.close();
  }, []);

  const dismiss = (idx: number) =>
    setAlerts((prev) => prev.filter((_, i) => i !== idx));

  const shown = alerts.slice(0, visible);

  return (
    <div className="flex flex-col h-full" style={{ background: "var(--card)", borderLeft: "1px solid var(--border)" }}>
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b" style={{ borderColor: "var(--border)" }}>
        <BellRing size={14} className="text-blue-400" />
        <span className="text-sm font-semibold" style={{ color: "var(--text)" }}>Live Alerts</span>
        {alerts.length > 0 && (
          <span className="ml-auto text-xs px-1.5 py-0.5 rounded-full bg-red-600 text-white font-bold">
            {alerts.length}
          </span>
        )}
      </div>

      {/* Alert list */}
      <div className="flex-1 overflow-y-auto">
        <AnimatePresence initial={false}>
          {alerts.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-xs" style={{ color: "var(--muted)" }}>
              <span>No active alerts</span>
            </div>
          ) : (
            shown.map((a, i) => (
              <motion.div
                key={`${a.camera_id}-${a.track_id}-${a.timestamp}`}
                initial={{ opacity: 0, y: -10, height: 0 }}
                animate={{ opacity: 1, y: 0, height: "auto" }}
                exit={{ opacity: 0, x: 30, height: 0 }}
                transition={{ duration: 0.2 }}
                className="alert-row px-3 py-2 border-b"
                style={{ borderColor: "var(--border)" }}
              >
                <div className="flex items-start gap-2">
                  <SeverityIcon s={a.severity} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5 flex-wrap">
                      <span className="text-xs font-bold uppercase tracking-wide" style={{ color: severityColor(a.severity) }}>
                        {a.severity}
                      </span>
                      <span className="text-xs" style={{ color: "var(--muted)" }}>
                        CAM#{a.camera_id} · TRK#{a.track_id}
                      </span>
                    </div>
                    <div className="flex items-center gap-1.5 mt-0.5 flex-wrap">
                      <span className="text-xs px-1 py-0.5 rounded" style={{ background: "var(--border)", color: "var(--text)" }}>
                        {a.action}
                      </span>
                      {a.in_zone && (
                        <span className="text-xs px-1 py-0.5 rounded bg-red-900/40 text-red-300">
                          {a.zone_name || "zone"}
                        </span>
                      )}
                      <span className="text-xs" style={{ color: "var(--muted)" }}>
                        T:{(a.threat_score * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => dismiss(i)}
                    className="opacity-40 hover:opacity-100 transition-opacity p-0.5 rounded"
                    style={{ color: "var(--muted)" }}
                  >
                    <X size={10} />
                  </button>
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>

        {/* Show more */}
        {visible < alerts.length && (
          <div className="px-3 py-2 text-center border-t" style={{ borderColor: "var(--border)" }}>
            <button
              onClick={() => setVisible((v) => v + PAGE)}
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              Show {Math.min(PAGE, alerts.length - visible)} more ({alerts.length - visible} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  );
}