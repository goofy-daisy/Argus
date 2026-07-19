import React, { useEffect, useRef } from "react";

interface Props {
  grid: number[][];
  width?: number;
  height?: number;
}

// Jet colormap: 0=blue → 0.5=green → 1=red
function jetColor(t: number): [number, number, number] {
  const v = Math.max(0, Math.min(1, t));
  const r = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export default function HeatmapOverlay({ grid, width = 640, height = 360 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || !grid.length) return;
    const rows = grid.length;
    const cols = grid[0].length;
    const canvas = canvasRef.current;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d")!;
    const cellW = width / cols;
    const cellH = height / rows;

    ctx.clearRect(0, 0, width, height);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[r][c];
        if (v < 0.01) continue;
        const [rv, gv, bv] = jetColor(v);
        ctx.fillStyle = `rgba(${rv},${gv},${bv},${Math.min(v * 0.85, 0.8)})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
      }
    }
  }, [grid, width, height]);

  return (
    <canvas
      ref={canvasRef}
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    />
  );
}
