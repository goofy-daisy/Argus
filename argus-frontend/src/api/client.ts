import axios from "axios";

const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const api = axios.create({ baseURL: BASE });

// Inject JWT on every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("argus_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// Redirect to login on 401
api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem("argus_token");
      window.location.href = "/login";
    }
    return Promise.reject(err);
  }
);

export const WS_BASE = BASE.replace(/^http/, "ws");
