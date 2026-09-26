import { Activity, Boxes, ChartNoAxesCombined, ChevronsLeft, ChevronsRight, CircleGauge, Layers3, Map, Mountain, Radar } from "lucide-react";
import type { LucideIcon } from "lucide-react";

export type View = "overview" | "live" | "semantic" | "objects" | "terrain" | "elevation" | "performance" | "benchmarks" | "settings";

const sections: Array<{ label: string; items: Array<{ id: View; label: string; icon: LucideIcon }> }> = [
  { label: "Overview", items: [{ id: "overview", label: "Overview", icon: CircleGauge }] },
  { label: "Perception", items: [
    { id: "objects", label: "Object Detection", icon: Boxes },
    { id: "terrain", label: "Terrain Analysis", icon: Mountain },
    { id: "elevation", label: "Elevation Map", icon: Map },
  ] },
  { label: "Analysis", items: [
    { id: "performance", label: "Performance", icon: Activity },
    { id: "benchmarks", label: "Foveated vs Uniform", icon: ChartNoAxesCombined },
  ] },
];

interface AppSidebarProps {
  view: View;
  collapsed: boolean;
  connected: boolean;
  sequence: string;
  onSelect: (view: View) => void;
  onToggle: () => void;
}

export function AppSidebar({ view, collapsed, connected, sequence, onSelect, onToggle }: AppSidebarProps) {
  return <aside className="sidebar">
    <div className="sidebar-brand">
      <button className="brand" onClick={() => onSelect("overview")} title="FoveaMap overview" aria-label="FoveaMap overview">
        <span className="brand-mark"><Radar size={19} strokeWidth={1.75} /></span>
        {!collapsed && <span className="brand-copy"><strong>FoveaMap</strong><small>PERCEPTION SYSTEM</small></span>}
      </button>
      <button className="sidebar-toggle" onClick={onToggle} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
        {collapsed ? <ChevronsRight size={15} /> : <ChevronsLeft size={15} />}
      </button>
    </div>
    <nav className="sidebar-nav" aria-label="Workspace navigation">
      {sections.map((section) => <div className="nav-section" key={section.label}>
        {!collapsed && <div className="nav-caption">{section.label}</div>}
        {section.items.map(({ id, label, icon: Icon }) => <button
          key={id}
          type="button"
          className={`nav-item ${view === id ? "active" : ""}`}
          aria-current={view === id ? "page" : undefined}
          aria-label={label}
          title={label}
          onClick={() => onSelect(id)}
        ><Icon size={16} strokeWidth={1.75} />{!collapsed && <span>{label}</span>}</button>)}
      </div>)}
    </nav>
    <div className="sidebar-bottom">
      <div className="sidebar-system-icon"><Layers3 size={16} strokeWidth={1.75} /></div>
      {!collapsed && <div className="sidebar-system-copy"><strong>Sequence {sequence}</strong><span><i className={connected ? "connected" : ""} />{connected ? "Connected" : "Reconnecting"}</span></div>}
    </div>
  </aside>;
}
