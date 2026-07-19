import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { BellRing, CheckCheck, Filter } from "lucide-react";
import { api } from "../api/client";
import { Alert } from "../types";

const PAGE = 20;

function severityBadge(sev: string | null) {
  const s = sev || "low";
  const colors: Record<string, string> = {
    high:   "bg-red-900/40 text-red-300 border-red-800/50",
    medium: "bg-amber-900/40 text-amber-300 border-amber-800/50",
    low:    "bg-emerald-900/40 text-emerald-300 border-emerald-800/50",
  };
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded border font-medium uppercase tracking-wide ${colors[s] || colors.low}`}>
      {s}
    </span>
  );
}

export default function Alerts(): React.ReactElement {
  const [alerts,  setAlerts]  = useState<Alert[]>([]);
  const [filter,  setFilter]  = useState<string>("all");
  const [loading, setLoading] = useState(true);
  const [visible, setVisible] = useState(PAGE);

  const load = () => {
    setLoading(true);
    api.get<Alert[]>("/api/v1/alerts", { params: { limit: 100 } })
      .then((r) => setAlerts(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 8000);
    return () => clearInterval(id);
  }, []);

  const ack = async (id: number) => {
    await api.patch(`/api/v1/alerts/${id}/acknowledge`, { acknowledged: true });
    setAlerts((prev) => prev.map((a) => a.id === id ? { ...a, acknowledged: true } : a));
  };

  const filtered = filter === "all"
    ? alerts
    : alerts.filter((a) => (a.severity || "low") === filter);

  const paged   = filtered.slice(0, visible);
  const hasMore = visible < filtered.length;

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold flex items-center gap-2" style={{ color: "var(--text)" }}>
          <BellRing size={16} className="text-blue-400" />
          Alerts
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-blue-900/40 text-blue-300">
            {filtered.length}
          </span>
        </h2>
        <div className="flex items-center gap-2">
          <Filter size={13} style={{ color: "var(--muted)" }} />
          {["all", "high", "medium", "low"].map((f) => (
            <button
              key={f}
              onClick={() => { setFilter(f); setVisible(PAGE); }}
              className={`text-xs px-2.5 py-1 rounded-full font-medium transition-colors ${
                filter === f ? "bg-blue-600 text-white" : "text-slate-400 hover:bg-white/5"
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
              {f !== "all" && (
                <span className="ml-1 opacity-60">
                  ({alerts.filter((a) => (a.severity || "low") === f).length})
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="rounded-lg overflow-hidden" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
        {loading ? (
          <div className="flex items-center justify-center h-32 text-xs" style={{ color: "var(--muted)" }}>
            Loading alerts…
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-xs" style={{ color: "var(--muted)" }}>
            <BellRing size={20} className="mb-2 opacity-30" />
            No alerts to display
          </div>
        ) : (
          <>
            <table className="w-full">
              <thead>
                <tr className="text-xs border-b" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
                  <th className="text-left px-4 py-2.5">ID</th>
                  <th className="text-left px-4 py-2.5">Severity</th>
                  <th className="text-left px-4 py-2.5">Type</th>
                  <th className="text-left px-4 py-2.5">Confidence</th>
                  <th className="text-left px-4 py-2.5">Track</th>
                  <th className="text-left px-4 py-2.5">Time</th>
                  <th className="text-right px-4 py-2.5">ACK</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence initial={false}>
                  {paged.map((a) => (
                    <motion.tr
                      key={a.id}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="alert-row border-b text-xs"
                      style={{ borderColor: "var(--border)", opacity: a.acknowledged ? 0.5 : 1 }}
                    >
                      <td className="px-4 py-2.5 font-mono" style={{ color: "var(--muted)" }}>#{a.id}</td>
                      <td className="px-4 py-2.5">{severityBadge(a.severity)}</td>
                      <td className="px-4 py-2.5" style={{ color: "var(--text)" }}>{a.type}</td>
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 rounded-full overflow-hidden w-16 bg-slate-700">
                            <div
                              className="h-full rounded-full"
                              style={{
                                width: `${a.confidence * 100}%`,
                                background: a.confidence >= 0.75 ? "#ef4444" : a.confidence >= 0.5 ? "#f59e0b" : "#10b981",
                              }}
                            />
                          </div>
                          <span style={{ color: "var(--muted)" }}>{(a.confidence * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5 font-mono" style={{ color: "var(--muted)" }}>#{a.track_id}</td>
                      <td className="px-4 py-2.5" style={{ color: "var(--muted)" }}>
                        {new Date(a.timestamp).toLocaleTimeString()}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        {a.acknowledged ? (
                          <CheckCheck size={13} className="text-emerald-500 ml-auto" />
                        ) : (
                          <button
                            onClick={() => ack(a.id)}
                            className="text-xs px-2 py-0.5 rounded text-blue-400 hover:bg-blue-900/30 transition-colors"
                          >
                            ACK
                          </button>
                        )}
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>

            {/* Pagination */}
            {hasMore && (
              <div className="flex items-center justify-center py-3 border-t" style={{ borderColor: "var(--border)" }}>
                <button
                  onClick={() => setVisible((v) => v + PAGE)}
                  className="text-xs px-4 py-1.5 rounded-lg transition-colors"
                  style={{ background: "var(--surface)", color: "var(--muted)", border: "1px solid var(--border)" }}
                >
                  Show {Math.min(PAGE, filtered.length - visible)} more
                  <span className="ml-1 opacity-60">({filtered.length - visible} remaining)</span>
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
}