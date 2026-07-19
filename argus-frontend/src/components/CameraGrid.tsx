import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Camera as CameraType } from "../types";
import CameraFeed from "./CameraFeed";

interface Props {
  cameras: CameraType[];
  onSelect?: (cam: CameraType) => void;
}

export default function CameraGrid({ cameras, onSelect }: Props) {
  const cols = cameras.length <= 1 ? 1 : cameras.length <= 4 ? 2 : 3;

  return (
    <div
      className="grid gap-3"
      style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
    >
      <AnimatePresence>
        {cameras.map((cam, i) => (
          <motion.div
            key={cam.id}
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            transition={{ duration: 0.25, delay: i * 0.04 }}
          >
            <CameraFeed camera={cam} onClick={() => onSelect?.(cam)} />
          </motion.div>
        ))}
      </AnimatePresence>

      {cameras.length === 0 && (
        <div className="col-span-2 flex flex-col items-center justify-center h-48 rounded-lg"
          style={{ border: "1px dashed var(--border)", color: "var(--muted)" }}>
          <span className="text-sm">No cameras configured</span>
          <span className="text-xs mt-1">Go to Cameras → Add Camera</span>
        </div>
      )}
    </div>
  );
}
