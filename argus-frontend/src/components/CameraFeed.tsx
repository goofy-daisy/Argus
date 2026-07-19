import React, { useRef, useState } from "react";
import { Camera, Wifi, WifiOff } from "lucide-react";
import { useVideoStream } from "../hooks/useVideoStream";
import { Camera as CameraType } from "../types";

interface Props {
  camera: CameraType;
  onClick?: () => void;
}

export default function CameraFeed({ camera, onClick }: Props): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [active] = useState(true);
  useVideoStream(camera.id, canvasRef);

  return (
    <div
      className="relative rounded-lg overflow-hidden cursor-pointer group scanline"
      style={{
        background: "#050b14",
        border: "1px solid var(--border)",
        aspectRatio: "16/9",
      }}
      onClick={onClick}
    >
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }}
      />

      {/* Overlay header */}
      <div className="absolute top-0 left-0 right-0 px-3 py-1.5 flex items-center justify-between z-10"
        style={{ background: "linear-gradient(to bottom, rgba(0,0,0,0.7) 0%, transparent 100%)" }}>
        <div className="flex items-center gap-1.5">
          <Camera size={11} className="text-blue-400" />
          <span className="text-xs font-medium text-white/90 truncate max-w-[120px]">
            {camera.name}
          </span>
          <span className="text-xs px-1 py-0.5 rounded text-blue-300/70 bg-blue-900/30">
            {camera.type.toUpperCase()}
          </span>
        </div>
        {active ? (
          <Wifi size={11} className="text-emerald-400" />
        ) : (
          <WifiOff size={11} className="text-red-400" />
        )}
      </div>

      {/* Camera ID badge */}
      <div className="absolute bottom-2 right-2 z-10 text-xs font-mono px-1.5 py-0.5 rounded"
        style={{ background: "rgba(0,0,0,0.6)", color: "var(--muted)" }}>
        #{camera.id}
      </div>

      {/* Hover border glow */}
      <div className="absolute inset-0 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none"
        style={{ boxShadow: "inset 0 0 0 1.5px #3b82f6" }} />
    </div>
  );
}
