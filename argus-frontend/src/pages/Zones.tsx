import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Map, Plus, Trash2, Loader2 } from "lucide-react";
import { api } from "../api/client";
import { Zone, Camera } from "../types";
import ZoneEditor from "../components/ZoneEditor";

export default function Zones() {
  const [zones, setZones] = useState<Zone[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [adding, setAdding] = useState(false);
  const [selectedCam, setSelectedCam] = useState<number | null>(null);
  const [zoneName, setZoneName] = useState("");
  const [saving, setSaving] = useState(false);

  const load = () =>
    api.get<Zone[]>("/api/v1/zones").then((r) => setZones(r.data)).catch(() => {});

  useEffect(() => {
    load();
    api.get<Camera[]>("/api/v1/cameras").then((r) => {
      setCameras(r.data);
      if (r.data.length > 0) setSelectedCam(r.data[0].id);
    }).catch(() => {});
  }, []);

  const handleSave = async (polygon: number[][]) => {
    if (!selectedCam || !zoneName.trim()) return;
    setSaving(true);
    try {
      await api.post("/api/v1/zones", {
        camera_id: selectedCam,
        name: zoneName.trim(),
        polygon,
        alert_on_enter: true,
        alert_on_dwell: false,
        dwell_threshold_seconds: 30,
        active: true,
      });
      await load();
      setAdding(false);
      setZoneName("");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    await api.delete(`/api/v1/zones/${id}`);
    setZones((z) => z.filter((x) => x.id !== id));
  };

  const toggleActive = async (zone: Zone) => {
    await api.patch(`/api/v1/zones/${zone.id}`, { active: !zone.active });
    setZones((z) => z.map((x) => x.id === zone.id ? { ...x, active: !x.active } : x));
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold flex items-center gap-2" style={{ color: "var(--text)" }}>
          <Map size={16} className="text-blue-400" />
          Zones
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-blue-900/40 text-blue-300">{zones.length}</span>
        </h2>
        <button
          onClick={() => setAdding((v) => !v)}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          <Plus size={12} /> New Zone
        </button>
      </div>

      <AnimatePresence>
        {adding && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="rounded-lg p-4 space-y-3"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}
          >
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs mb-1" style={{ color: "var(--muted)" }}>Zone Name</label>
                <input
                  value={zoneName}
                  onChange={(e) => setZoneName(e.target.value)}
                  placeholder="Restricted Area"
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", outline: "none" }}
                />
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: "var(--muted)" }}>Camera</label>
                <select
                  value={selectedCam ?? ""}
                  onChange={(e) => setSelectedCam(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", outline: "none" }}
                >
                  {cameras.map((c) => (
                    <option key={c.id} value={c.id}>{c.name}</option>
                  ))}
                </select>
              </div>
            </div>
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              Click on the canvas below to place polygon vertices. Need at least 3 points.
            </p>
            <ZoneEditor onSave={handleSave} />
            {saving && (
              <div className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
                <Loader2 size={12} className="animate-spin" /> Saving zone…
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Zone list */}
      <div className="rounded-lg overflow-hidden" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
        {zones.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-xs" style={{ color: "var(--muted)" }}>
            <Map size={20} className="mb-2 opacity-30" />
            No zones configured
          </div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
                <th className="text-left px-4 py-2.5">Name</th>
                <th className="text-left px-4 py-2.5">Camera</th>
                <th className="text-left px-4 py-2.5">Vertices</th>
                <th className="text-left px-4 py-2.5">Alert on Enter</th>
                <th className="text-left px-4 py-2.5">Status</th>
                <th className="text-right px-4 py-2.5">Actions</th>
              </tr>
            </thead>
            <tbody>
              <AnimatePresence>
                {zones.map((z) => (
                  <motion.tr
                    key={z.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="border-b"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="px-4 py-2.5 font-medium" style={{ color: "var(--text)" }}>{z.name}</td>
                    <td className="px-4 py-2.5 font-mono" style={{ color: "var(--muted)" }}>#{z.camera_id}</td>
                    <td className="px-4 py-2.5" style={{ color: "var(--muted)" }}>{z.polygon.length}</td>
                    <td className="px-4 py-2.5">
                      <span className={`px-1.5 py-0.5 rounded text-xs ${z.alert_on_enter ? "bg-emerald-900/40 text-emerald-300" : "bg-slate-800 text-slate-500"}`}>
                        {z.alert_on_enter ? "Yes" : "No"}
                      </span>
                    </td>
                    <td className="px-4 py-2.5">
                      <button
                        onClick={() => toggleActive(z)}
                        className={`px-1.5 py-0.5 rounded text-xs cursor-pointer transition-colors ${z.active ? "bg-blue-900/40 text-blue-300 hover:bg-red-900/30 hover:text-red-300" : "bg-slate-800 text-slate-500 hover:bg-blue-900/30 hover:text-blue-300"}`}
                      >
                        {z.active ? "Active" : "Inactive"}
                      </button>
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <button
                        onClick={() => handleDelete(z.id)}
                        className="p-1 rounded hover:bg-red-900/30 text-red-400 transition-colors"
                      >
                        <Trash2 size={12} />
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </AnimatePresence>
            </tbody>
          </table>
        )}
      </div>
    </motion.div>
  );
}
