import React from "react";

interface Props {
  score: number; // 0–1
  label?: string;
  size?: number;
}

function scoreColor(s: number) {
  if (s >= 0.75) return "#ef4444";
  if (s >= 0.5) return "#f59e0b";
  return "#10b981";
}

function scoreSeverity(s: number) {
  if (s >= 0.75) return "HIGH";
  if (s >= 0.5) return "MED";
  return "LOW";
}

export default function ThreatGauge({ score, label = "Threat", size = 120 }: Props) {
  const r = size / 2 - 10;
  const cx = size / 2;
  const cy = size / 2 + 10;
  const startAngle = Math.PI;      // 180°
  const sweepAngle = Math.PI;      // 180° arc (semicircle)
  const angle = startAngle + sweepAngle * Math.min(Math.max(score, 0), 1);
  const color = scoreColor(score);

  const arcPath = (a1: number, a2: number, fill = false) => {
    const x1 = cx + r * Math.cos(a1);
    const y1 = cy + r * Math.sin(a1);
    const x2 = cx + r * Math.cos(a2);
    const y2 = cy + r * Math.sin(a2);
    const large = a2 - a1 > Math.PI ? 1 : 0;
    return fill
      ? `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`
      : `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  const needleX = cx + (r - 8) * Math.cos(angle);
  const needleY = cy + (r - 8) * Math.sin(angle);
  const glassClass = score >= 0.75 ? "gauge-high" : score >= 0.5 ? "gauge-med" : "gauge-low";

  return (
    <div className="flex flex-col items-center select-none">
      <svg
        width={size}
        height={size / 2 + 20}
        className={glassClass}
        style={{ overflow: "visible" }}
      >
        {/* Track */}
        <path
          d={arcPath(Math.PI, 2 * Math.PI)}
          fill="none"
          stroke="#1e2d45"
          strokeWidth={8}
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d={arcPath(Math.PI, angle)}
          fill="none"
          stroke={color}
          strokeWidth={8}
          strokeLinecap="round"
        />
        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={needleX}
          y2={needleY}
          stroke={color}
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={4} fill={color} />
        {/* Score text */}
        <text
          x={cx}
          y={cy - 14}
          textAnchor="middle"
          fontSize={14}
          fontWeight={700}
          fill={color}
          fontFamily="JetBrains Mono, monospace"
        >
          {(score * 100).toFixed(0)}
        </text>
        <text
          x={cx}
          y={cy - 1}
          textAnchor="middle"
          fontSize={8}
          fill="#64748b"
        >
          {scoreSeverity(score)}
        </text>
      </svg>
      {label && (
        <span className="text-xs mt-1 font-medium" style={{ color: "var(--muted)" }}>
          {label}
        </span>
      )}
    </div>
  );
}
