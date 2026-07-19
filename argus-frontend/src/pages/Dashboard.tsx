import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Activity, Camera as CameraIcon, BellRing, Users } from "lucide-react";
import { api } from "../api/client";
import { Camera, Alert } from "../types";
import CameraGrid from "../components/CameraGrid";
import AlertPanel from "../components/AlertPanel";
import ThreatGauge from "../components/ThreatGauge";

function StatCard({ icon: Icon, label, value, color }: {
  icon: any; label: string; value: string | number; color: string;
}) {
  return (
    <div className="rounded-lg p-4 flex items-center gap-3" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
      <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: `${color}20` }}>
        <Icon size={16} style={{ color }} />
      </div>
      <div>
        <p className="text-xs" style={{ color: "var(--muted)" }}>{label}</p>
        <p className="text-lg font-bold" style={{ color: "var(--text)" }}>{value}</p>
      </div>
    </div>
  );
}

export default function Dashboard(): React.ReactElement {
  const [cameras,     setCameras]     = useState<Camera[]>([]);
  const [alerts,      setAlerts]      = useState<Alert[]>([]);
  const [threatScore, setThreatScore] = useState(0);
  const [activeCams,  setActiveCams]  = useState(0);

  const loadCameras = () =>
    Promise.all([
      api.get<Camera[]>("/api/v1/cameras"),
      api.get<{ active_pipelines: number[] }>("/health"),
    ])
      .then(([camsRes, healthRes]) => {
        const active = new Set(healthRes.data.active_pipelines ?? []);
        setActiveCams(active.size);
        setCameras(camsRes.data.filter((c) => active.has(c.id)));
      })
      .catch(() => {});

  const loadAlerts = () =>
    api.get<Alert[]>("/api/v1/alerts", { params: { limit: 100 } })
      .then((r) => {
        setAlerts(r.data);
        const highCount = r.data.filter((a) => a.severity === "high").length;
        const medCount  = r.data.filter((a) => a.severity === "medium").length;
        setThreatScore(Math.min((highCount * 2 + medCount) / 20, 1));
      })
      .catch(() => {});

  useEffect(() => {
    loadCameras();
    loadAlerts();
    const c1 = setInterval(loadCameras, 10000);
    const c2 = setInterval(loadAlerts,  8000);
    return () => { clearInterval(c1); clearInterval(c2); };
  }, []);

  const highAlerts   = alerts.filter((a) => a.severity === "high").length;
  const medAlerts    = alerts.filter((a) => a.severity === "medium").length;
  const activeTracks = new Set(alerts.map((a) => a.track_id).filter(Boolean)).size;

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="flex flex-col h-full gap-4"
    >
      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard icon={CameraIcon} label="Active Cameras" value={activeCams}    color="#3b82f6" />
        <StatCard icon={BellRing}   label="High Alerts"    value={highAlerts}    color="#ef4444" />
        <StatCard icon={Activity}   label="Active Tracks"  value={activeTracks || "—"} color="#10b981" />
        <StatCard icon={Users}      label="Med Alerts"     value={medAlerts}     color="#f59e0b" />
      </div>

      {/* Main layout */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-4 min-h-0">
        {/* Camera grid */}
        <div className="lg:col-span-3 overflow-auto">
          <CameraGrid cameras={cameras} />
        </div>

        {/* Right column: gauge + alert panel */}
        <div className="flex flex-col gap-3">
          <div className="rounded-lg p-4 flex flex-col items-center"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
            <p className="text-xs font-medium mb-3 self-start" style={{ color: "var(--muted)" }}>
              System Threat Level
            </p>
            <ThreatGauge score={threatScore} size={140} label="" />
          </div>

          <div className="flex-1 rounded-lg overflow-hidden" style={{ minHeight: 240 }}>
            <AlertPanel />
          </div>
        </div>
      </div>
    </motion.div>
  );
}