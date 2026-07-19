import React, { useRef, useState } from "react";
import { Trash2, Check, Undo } from "lucide-react";

interface Props {
  onSave?: (polygon: number[][]) => void;
  width?: number;
  height?: number;
}

export default function ZoneEditor({ onSave, width = 640, height = 360 }: Props) {
  const [points, setPoints] = useState<number[][]>([]);
  const svgRef = useRef<SVGSVGElement>(null);

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = svgRef.current!.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    setPoints((prev) => [...prev, [x, y]]);
  };

  const toPixel = (p: number[]) => [p[0] * width, p[1] * height];

  const polyPoints = points.map((p) => toPixel(p).join(",")).join(" ");

  return (
    <div className="rounded-lg overflow-hidden" style={{ border: "1px solid var(--border)", background: "#050b14" }}>
      <div className="px-3 py-2 flex items-center justify-between border-b" style={{ borderColor: "var(--border)" }}>
        <span className="text-xs font-medium" style={{ color: "var(--text)" }}>
          Zone Editor — click to place vertices
        </span>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setPoints((p) => p.slice(0, -1))}
            className="p-1 rounded hover:bg-white/10 transition-colors"
            title="Undo last point"
          >
            <Undo size={12} style={{ color: "var(--muted)" }} />
          </button>
          <button
            onClick={() => setPoints([])}
            className="p-1 rounded hover:bg-red-900/30 transition-colors"
            title="Clear"
          >
            <Trash2 size={12} className="text-red-400" />
          </button>
          <button
            onClick={() => points.length >= 3 && onSave?.(points)}
            disabled={points.length < 3}
            className="px-2 py-0.5 rounded text-xs font-medium bg-blue-600 text-white disabled:opacity-30 hover:bg-blue-500 transition-colors flex items-center gap-1"
          >
            <Check size={10} />
            Save Zone
          </button>
        </div>
      </div>

      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        style={{ display: "block", cursor: "crosshair" }}
        onClick={handleClick}
      >
        {/* Grid lines */}
        {Array.from({ length: 6 }, (_, i) => (
          <line key={`h${i}`} x1={0} y1={(i + 1) * (height / 7)} x2={width} y2={(i + 1) * (height / 7)}
            stroke="#1e2d45" strokeWidth={0.5} />
        ))}
        {Array.from({ length: 10 }, (_, i) => (
          <line key={`v${i}`} x1={(i + 1) * (width / 11)} y1={0} x2={(i + 1) * (width / 11)} y2={height}
            stroke="#1e2d45" strokeWidth={0.5} />
        ))}

        {/* Polygon fill */}
        {points.length >= 3 && (
          <polygon points={polyPoints} fill="rgba(59,130,246,0.15)" stroke="#3b82f6" strokeWidth={1.5} />
        )}

        {/* Edges for in-progress polygon */}
        {points.length >= 2 && points.length < 3 && (
          <polyline
            points={points.map((p) => toPixel(p).join(",")).join(" ")}
            fill="none"
            stroke="#3b82f6"
            strokeWidth={1.5}
            strokeDasharray="4 2"
          />
        )}

        {/* Vertex dots */}
        {points.map((p, i) => {
          const [px, py] = toPixel(p);
          return (
            <circle key={i} cx={px} cy={py} r={5} fill="#3b82f6" stroke="#fff" strokeWidth={1.5} />
          );
        })}
      </svg>

      <div className="px-3 py-1.5 text-xs" style={{ color: "var(--muted)", borderTop: "1px solid var(--border)" }}>
        {points.length} vertices placed
        {points.length >= 3 && " · polygon ready"}
        {points.length > 0 && points.length < 3 && " · need at least 3"}
      </div>
    </div>
  );
}
