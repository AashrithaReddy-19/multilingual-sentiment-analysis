import { useState } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import Batch from "./pages/Batch";
import Comparison from "./pages/Comparison";
import Audit from "./pages/Audit";
import Errors from "./pages/Errors";
import Context from "./pages/Context";

const NAV = [
  { to: "/", label: "Dashboard", icon: "▦" },
  { to: "/predict", label: "Single prediction", icon: "✎" },
  { to: "/batch", label: "Batch prediction", icon: "⇪" },
  { to: "/comparison", label: "Model comparison", icon: "◫" },
  { to: "/audit", label: "Data audit", icon: "☰" },
  { to: "/errors", label: "Error analysis", icon: "⚑" },
  { to: "/context", label: "Context status", icon: "↺" },
];

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Main" className="flex flex-col gap-1">
      {NAV.map((n) => (
        <NavLink key={n.to} to={n.to} end={n.to === "/"} onClick={onNavigate}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
              isActive ? "bg-indigo-50 text-indigo-700" : "text-slate-700 hover:bg-slate-100"}`}>
          <span aria-hidden className="w-4 text-center">{n.icon}</span>
          {n.label}
        </NavLink>
      ))}
    </nav>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-indigo-600 text-sm font-bold text-white" aria-hidden>తె</div>
      <div className="leading-tight">
        <p className="text-sm font-semibold text-slate-900">Sentiment Lab</p>
        <p className="text-xs text-slate-500">Telugu · English · Code-mixed</p>
      </div>
    </div>
  );
}

export default function App() {
  const [open, setOpen] = useState(false);
  const location = useLocation();
  return (
    <div className="min-h-screen lg:flex">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:rounded focus:bg-white focus:px-3 focus:py-2">
        Skip to content
      </a>
      <aside className="hidden w-64 shrink-0 border-r border-slate-200 bg-white p-4 lg:block">
        <div className="sticky top-4 space-y-6">
          <Brand />
          <Nav />
        </div>
      </aside>
      <div className="sticky top-0 z-30 flex items-center justify-between border-b border-slate-200 bg-white/90 px-4 py-3 backdrop-blur lg:hidden">
        <Brand />
        <button onClick={() => setOpen(!open)} aria-expanded={open} aria-controls="mobile-nav"
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-700">
          {open ? "Close" : "Menu"}
        </button>
      </div>
      {open && (
        <div id="mobile-nav" className="border-b border-slate-200 bg-white p-3 lg:hidden">
          <Nav onNavigate={() => setOpen(false)} />
        </div>
      )}
      <main id="main" key={location.pathname} className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/batch" element={<Batch />} />
          <Route path="/comparison" element={<Comparison />} />
          <Route path="/audit" element={<Audit />} />
          <Route path="/errors" element={<Errors />} />
          <Route path="/context" element={<Context />} />
          <Route path="*" element={<p className="text-slate-600">Page not found.</p>} />
        </Routes>
      </main>
    </div>
  );
}
