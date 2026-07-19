import React from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  Camera,
  BellRing,
  BarChart2,
  Map,
  LogOut,
  Shield,
} from "lucide-react";

const NAV = [
  { to: "/", icon: LayoutDashboard, label: "Dashboard" },
  { to: "/cameras", icon: Camera, label: "Cameras" },
  { to: "/alerts", icon: BellRing, label: "Alerts" },
  { to: "/analytics", icon: BarChart2, label: "Analytics" },
  { to: "/zones", icon: Map, label: "Zones" },
];

export default function Sidebar() {
  const navigate = useNavigate();

  const handleLogout = () => {
    localStorage.removeItem("argus_token");
    navigate("/login");
  };

  return (
    <motion.aside
      initial={{ x: -80, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className="fixed inset-y-0 left-0 w-56 flex flex-col z-20"
      style={{ background: "var(--surface)", borderRight: "1px solid var(--border)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-5 py-5 border-b" style={{ borderColor: "var(--border)" }}>
        <Shield size={22} className="text-blue-500" />
        <span className="font-semibold text-base tracking-wide" style={{ color: "var(--text)" }}>
          Argus
        </span>
        <span className="ml-auto text-xs px-1.5 py-0.5 rounded" style={{ background: "var(--border)", color: "var(--muted)" }}>
          v0.9
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 space-y-0.5 px-2">
        {NAV.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                isActive
                  ? "bg-blue-600 text-white shadow-lg shadow-blue-900/40"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Logout */}
      <div className="px-2 py-4 border-t" style={{ borderColor: "var(--border)" }}>
        <button
          onClick={handleLogout}
          className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg text-sm font-medium text-slate-400 hover:bg-red-900/20 hover:text-red-400 transition-all duration-150"
        >
          <LogOut size={16} />
          Logout
        </button>
      </div>
    </motion.aside>
  );
}
