import React, { useEffect, useState } from "react";
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "../api/client";
import { Track } from "../types";

interface Props {
  cameraId: number | null;
}

export default function TrackExplorer({ cameraId }: Props) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (cameraId === null) return;
    setLoading(true);
    api
      .get<Track[]>("/api/v1/tracks", { params: { camera_id: cameraId, limit: 100 } })
      .then((r) => setTracks(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [cameraId]);

  const chartData = tracks.map((t, i) => ({
    name: `#${t.id}`,
    score: t.anomaly_score ?? 0,
    frames: t.frame_end ? t.frame_end - t.frame_start : 0,
  }));

  return (
    <div className="rounded-lg p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
      <h3 className="text-sm font-semibold mb-3" style={{ color: "var(--text)" }}>
        Track Anomaly Scores
        {cameraId !== null && (
          <span className="ml-2 text-xs font-normal" style={{ color: "var(--muted)" }}>
            Camera #{cameraId}
          </span>
        )}
      </h3>

      {loading ? (
        <div className="h-32 flex items-center justify-center text-xs" style={{ color: "var(--muted)" }}>
          Loading…
        </div>
      ) : chartData.length === 0 ? (
        <div className="h-32 flex items-center justify-center text-xs" style={{ color: "var(--muted)" }}>
          No tracks recorded yet
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={140}>
          <AreaChart data={chartData}>
            <defs>
              <linearGradient id="anomGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="name" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 0.1]} tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{ background: "#1a2235", border: "1px solid #1e2d45", borderRadius: 6, fontSize: 11 }}
              labelStyle={{ color: "#e2e8f0" }}
            />
            <Area type="monotone" dataKey="score" stroke="#ef4444" fill="url(#anomGrad)" strokeWidth={1.5} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      )}

      {/* Track table */}
      {tracks.length > 0 && (
        <div className="mt-3 overflow-y-auto max-h-48">
          <table className="w-full text-xs">
            <thead>
              <tr style={{ color: "var(--muted)" }}>
                <th className="text-left py-1">ID</th>
                <th className="text-left py-1">Frames</th>
                <th className="text-left py-1">Label</th>
                <th className="text-right py-1">Anomaly</th>
              </tr>
            </thead>
            <tbody>
              {tracks.slice(0, 20).map((t) => (
                <tr key={t.id} className="border-t" style={{ borderColor: "var(--border)" }}>
                  <td className="py-1 font-mono" style={{ color: "var(--text)" }}>#{t.id}</td>
                  <td className="py-1" style={{ color: "var(--muted)" }}>
                    {t.frame_end ? t.frame_end - t.frame_start : "—"}
                  </td>
                  <td className="py-1" style={{ color: "var(--muted)" }}>{t.label ?? "normal"}</td>
                  <td className="py-1 text-right font-mono" style={{
                    color: (t.anomaly_score ?? 0) > 0.05 ? "#ef4444" : "#10b981"
                  }}>
                    {(t.anomaly_score ?? 0).toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
