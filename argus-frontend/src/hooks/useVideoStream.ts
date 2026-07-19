import { useEffect, useRef } from "react";
import { WS_BASE } from "../api/client";

export function useVideoStream(
  cameraId: number | null,
  canvasRef: React.RefObject<HTMLCanvasElement | null>
) {
  const wsRef    = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const deadRef  = useRef(false);

  useEffect(() => {
    if (cameraId === null) return;
    deadRef.current = false;

    function connect() {
      if (deadRef.current) return;

      const canvas = canvasRef.current;
      if (!canvas) {
        retryRef.current = setTimeout(connect, 500);
        return;
      }

      const ws = new WebSocket(`${WS_BASE}/ws/stream/${cameraId}`);
      wsRef.current = ws;
      ws.binaryType = "arraybuffer";

      ws.onmessage = async (event: MessageEvent<ArrayBuffer>) => {
        if (deadRef.current) return;
        try {
          const blob = new Blob([event.data], { type: "image/jpeg" });
          const bmp  = await createImageBitmap(blob);
          if (deadRef.current) { bmp.close(); return; }
          const c = canvasRef.current;
          if (!c) { bmp.close(); return; }
          c.width  = bmp.width;
          c.height = bmp.height;
          c.getContext("2d")?.drawImage(bmp, 0, 0);
          bmp.close();
        } catch {}
      };

      ws.onerror = () => ws.close();

      ws.onclose = () => {
        if (!deadRef.current) {
          retryRef.current = setTimeout(connect, 2000);
        }
      };
    }

    connect();

    return () => {
      deadRef.current = true;
      if (retryRef.current) clearTimeout(retryRef.current);
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [cameraId]);
}