import React, { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, Plus, Play, Square, Trash2, Loader2 } from "lucide-react";
import { api } from "../api/client";
import { Camera as CameraType } from "../types";
import TrackExplorer from "../components/TrackExplorer";

function CameraCard({
  cam,
  onStart,
  onStop,
  onDelete,
  running,
  onSelect,
  selected,
}: {
  cam: CameraType;
  onStart: () => void;
  onStop: () => void;
  onDelete: () => void;
  running: boolean;
  onSelect: () => void;
  selected: boolean;
}) {
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="rounded-lg p-4 cursor-pointer transition-all"
      style={{
        background: "var(--card)",
        border: `1px solid ${selected ? "var(--accent)" : "var(--border)"}`,
      }}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Camera size={14} className="text-blue-400 shrink-0" />
          <span className="text-sm font-semibold truncate max-w-[150px]" style={{ color: "var(--text)" }}>
            {cam.name}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span
            className="text-xs px-1.5 py-0.5 rounded"
            style={{ background: running ? "#10b98120" : "#1e2d45", color: running ? "#10b981" : "var(--muted)" }}
          >
            {running ? "● Live" : "Off"}
          </span>
        </div>
      </div>

      <p className="text-xs mb-1" style={{ color: "var(--muted)" }}>
        Type: <span style={{ color: "var(--text)" }}>{cam.type.toUpperCase()}</span>
      </p>
      <p className="text-xs truncate mb-3" style={{ color: "var(--muted)" }}>
        Source: <span className="font-mono text-xs" style={{ color: "var(--text)" }}>{cam.location}</span>
      </p>

      <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
        {!running ? (
          <button
            onClick={onStart}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg bg-emerald-600/80 hover:bg-emerald-600 text-white transition-colors"
          >
            <Play size={11} /> Start
          </button>
        ) : (
          <button
            onClick={onStop}
            className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg bg-slate-600/80 hover:bg-slate-600 text-white transition-colors"
          >
            <Square size={11} /> Stop
          </button>
        )}
        <button
          onClick={onDelete}
          className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-lg hover:bg-red-900/30 text-red-400 transition-colors"
        >
          <Trash2 size={11} /> Delete
        </button>
      </div>
    </motion.div>
  );
}

export default function Cameras(): React.ReactElement {
  const [cameras, setCameras] = useState<CameraType[]>([]);
  const [running, setRunning] = useState<Set<number>>(new Set());
  const [selected, setSelected] = useState<number | null>(null);
  const [adding, setAdding] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState({ name: "", type: "rgb", location: "" });

  const load = async () => {
    try {
      const [camsRes, healthRes] = await Promise.all([
        api.get<CameraType[]>("/api/v1/cameras"),
        api.get<{ active_pipelines: number[] }>("/health"),
      ]);
      setCameras(camsRes.data);
      setRunning(new Set(healthRes.data.active_pipelines ?? []));
    } catch {}
  };

  useEffect(() => { load(); }, []);

  const handleStart = async (id: number) => {
    await api.post(`/api/v1/cameras/${id}/start`);
    setRunning((s) => new Set(s).add(id));
  };

  const handleStop = async (id: number) => {
    await api.post(`/api/v1/cameras/${id}/stop`);
    setRunning((s) => { const n = new Set(s); n.delete(id); return n; });
  };

  const handleDelete = async (id: number) => {
    await api.delete(`/api/v1/cameras/${id}`);
    setCameras((c) => c.filter((x) => x.id !== id));
    if (selected === id) setSelected(null);
  };

  const handleSave = async () => {
    if (!form.name || !form.location) return;
    setSaving(true);
    try {
      await api.post("/api/v1/cameras", form);
      await load();
      setAdding(false);
      setForm({ name: "", type: "rgb", location: "" });
    } finally {
      setSaving(false);
    }
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
          <Camera size={16} className="text-blue-400" />
          Cameras
          <span className="text-xs px-1.5 py-0.5 rounded-full bg-blue-900/40 text-blue-300">{cameras.length}</span>
        </h2>
        <button
          onClick={() => setAdding((v) => !v)}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors"
        >
          <Plus size={12} /> Add Camera
        </button>
      </div>

      {/* Add camera form */}
      <AnimatePresence>
        {adding && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="rounded-lg p-4 space-y-3"
            style={{ background: "var(--card)", border: "1px solid var(--border)" }}
          >
            <p className="text-sm font-medium" style={{ color: "var(--text)" }}>New Camera</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <label className="block text-xs mb-1" style={{ color: "var(--muted)" }}>Name</label>
                <input
                  value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                  placeholder="Front Door"
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", outline: "none" }}
                />
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: "var(--muted)" }}>Type</label>
                <select
                  value={form.type}
                  onChange={(e) => setForm((f) => ({ ...f, type: e.target.value }))}
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", outline: "none" }}
                >
                  <option value="rgb">RGB</option>
                  <option value="thermal">Thermal</option>
                </select>
              </div>
              <div>
                <label className="block text-xs mb-1" style={{ color: "var(--muted)" }}>Source (file path or RTSP URL)</label>
                <input
                  value={form.location}
                  onChange={(e) => setForm((f) => ({ ...f, location: e.target.value }))}
                  placeholder="/path/to/video.mp4 or rtsp://..."
                  className="w-full px-3 py-2 rounded-lg text-sm"
                  style={{ background: "var(--surface)", border: "1px solid var(--border)", color: "var(--text)", outline: "none" }}
                />
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors disabled:opacity-50"
              >
                {saving && <Loader2 size={11} className="animate-spin" />}
                Save
              </button>
              <button
                onClick={() => setAdding(false)}
                className="text-xs px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors"
                style={{ color: "var(--muted)" }}
              >
                Cancel
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Camera grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        <AnimatePresence>
          {cameras.map((cam) => (
            <CameraCard
              key={cam.id}
              cam={cam}
              running={running.has(cam.id)}
              selected={selected === cam.id}
              onSelect={() => setSelected(selected === cam.id ? null : cam.id)}
              onStart={() => handleStart(cam.id)}
              onStop={() => handleStop(cam.id)}
              onDelete={() => handleDelete(cam.id)}
            />
          ))}
        </AnimatePresence>
        {cameras.length === 0 && (
          <div className="col-span-3 flex flex-col items-center justify-center h-32 rounded-lg text-xs"
            style={{ border: "1px dashed var(--border)", color: "var(--muted)" }}>
            No cameras yet. Click "Add Camera" to get started.
          </div>
        )}
      </div>

      {/* Track explorer for selected camera */}
      {selected !== null && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          <TrackExplorer cameraId={selected} />
        </motion.div>
      )}
    </motion.div>
  );
}
