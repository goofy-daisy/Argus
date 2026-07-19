export interface Camera {
  id: number;
  name: string;
  type: "rgb" | "thermal";
  location: string;
}

export interface Track {
  id: number;
  camera_id: number;
  frame_start: number;
  frame_end: number | null;
  label: string | null;
  anomaly_score: number | null;
  bbox_history: number[][];
}

export interface Alert {
  id: number;
  track_id: number;
  type: string;
  confidence: number;
  severity: "low" | "medium" | "high" | null;
  acknowledged: boolean;
  timestamp: string;
}

export interface LiveAlert {
  type: "alert";
  camera_id: number;
  track_id: number;
  threat_score: number;
  severity: "low" | "medium" | "high";
  action: string;
  anomaly_score: number;
  in_zone: boolean;
  zone_name: string;
  timestamp: number;
}

export interface Zone {
  id: string;
  camera_id: number;
  name: string;
  polygon: number[][];
  alert_on_enter: boolean;
  alert_on_dwell: boolean;
  dwell_threshold_seconds: number;
  active: boolean;
  created_at: string;
}

export interface HeatmapData {
  camera_id: number;
  grid: number[][];
  grid_rows: number;
  grid_cols: number;
  timestamp: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}
