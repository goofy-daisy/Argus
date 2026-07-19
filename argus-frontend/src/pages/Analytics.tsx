import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend
} from "recharts";
import { api } from "../api/client";
import { Alert } from "../types";

const CHART = {
  contentStyle: { background: "#1a2235", border: "1px solid #1e2d45", borderRadius: 6, fontSize: 11 },
  labelStyle:   { color: "#e2e8f0" },
};

export default function Analytics(): React.ReactElement {
  const [alerts, setAlerts] = useState<Alert[]>([]);

  const load = () =>
    api.get<Alert[]>("/api/v1/alerts", { params: { limit: 100 } })
      .then((r) => setAlerts(r.data))
      .catch(() => {});

  useEffect(() => {
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);

  // Timeline: group by hour
  const hourlyMap: Record<string, { high: number; medium: number; low: number }> = {};
  alerts.forEach((a) => {
    const d   = new Date(a.timestamp);
    const key = `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:00`;
    if (!hourlyMap[key]) hourlyMap[key] = { high: 0, medium: 0, low: 0 };
    const sev = (a.severity || "low") as "high" | "medium" | "low";
    hourlyMap[key][sev]++;
  });
  const timelineData = Object.entries(hourlyMap)
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-24)
    .map(([time, v]) => ({ time, ...v }));

  // Confidence buckets
  const confBuckets = Array.from({ length: 10 }, (_, i) => ({
    range: `${i * 10}-${(i + 1) * 10}%`,
    count: 0,
  }));
  alerts.forEach((a) => {
    const b = Math.min(9, Math.floor(a.confidence * 10));
    confBuckets[b].count++;
  });

  // Severity counts
  const highCount   = alerts.filter((a) => a.severity === "high").length;
  const medCount    = alerts.filter((a) => a.severity === "medium").length;
  const lowCount    = alerts.filter((a) => (a.severity || "low") === "low").length;
  const sevData = [
    { name: "High",   value: highCount, fill: "#ef4444" },
    { name: "Medium", value: medCount,  fill: "#f59e0b" },
    { name: "Low",    value: lowCount,  fill: "#10b981" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold" style={{ color: "var(--text)" }}>Analytics</h2>
        <span className="text-xs" style={{ color: "var(--muted)" }}>{alerts.length} total alerts</span>
      </div>

      {/* Timeline */}
      <div className="rounded-lg p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
        <h3 className="text-sm font-medium mb-4" style={{ color: "var(--text)" }}>Alert Timeline (last 24h)</h3>
        {timelineData.length === 0 ? (
          <div className="h-32 flex items-center justify-center text-xs" style={{ color: "var(--muted)" }}>
            No alert data yet
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={180}>
            <AreaChart data={timelineData}>
              <defs>
                <linearGradient id="highGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0}   />
                </linearGradient>
                <linearGradient id="medGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#f59e0b" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}   />
                </linearGradient>
                <linearGradient id="lowGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#10b981" stopOpacity={0.2} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0}   />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
              <XAxis dataKey="time" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip {...CHART} />
              <Legend wrapperStyle={{ fontSize: 11, color: "#64748b" }} />
              <Area type="monotone" dataKey="high"   stroke="#ef4444" fill="url(#highGrad)" strokeWidth={1.5} dot={false} name="High"   />
              <Area type="monotone" dataKey="medium" stroke="#f59e0b" fill="url(#medGrad)"  strokeWidth={1.5} dot={false} name="Medium" />
              <Area type="monotone" dataKey="low"    stroke="#10b981" fill="url(#lowGrad)"  strokeWidth={1.5} dot={false} name="Low"    />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Confidence distribution */}
        <div className="rounded-lg p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
          <h3 className="text-sm font-medium mb-4" style={{ color: "var(--text)" }}>Detection Confidence Distribution</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={confBuckets} barSize={18}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
              <XAxis dataKey="range" tick={{ fill: "#64748b", fontSize: 9 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip {...CHART} />
              <Bar dataKey="count" fill="#3b82f6" radius={[3, 3, 0, 0]} name="Count" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Severity breakdown */}
        <div className="rounded-lg p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
          <h3 className="text-sm font-medium mb-4" style={{ color: "var(--text)" }}>Severity Breakdown</h3>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={sevData} layout="vertical" barSize={20}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
              <XAxis type="number" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis dataKey="name" type="category" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} width={60} />
              <Tooltip {...CHART} />
              <Bar dataKey="value" radius={[0, 3, 3, 0]} name="Count"
                label={{ position: "right", fontSize: 10, fill: "#64748b" }}>
                {sevData.map((entry, i) => (
                  <rect key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-3">
        {sevData.map((s) => (
          <div key={s.name} className="rounded-lg p-3 text-center" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
            <p className="text-2xl font-bold" style={{ color: s.fill }}>{s.value}</p>
            <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>{s.name} Alerts</p>
          </div>
        ))}
      </div>
    </motion.div>
  );
}