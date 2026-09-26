# FoveaMap Frontend Redesign and v0 Integration

## Project context

**Problem statement:** Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception.

Autonomous navigation needs rich 3D LiDAR perception, but millions of points can cause computational bottlenecks and memory latency. Ordinary 2D occupancy grids discard the height information needed to detect curbs, potholes and overhanging obstacles. FoveaMap transforms raw 3D LiDAR point clouds into a variable resolution 2.5D elevation grid with semantic layers: high detail nearby for safety and progressively coarser detail at a distance.

The system must show:

1. **Terrain analysis:** drivable versus non-drivable surfaces.
2. **Object detection:** static obstacles such as walls and poles, and dynamic objects such as pedestrians and vehicles.
3. **Adaptive spatial representation:** non-uniform grid cells that grow with distance, while avoiding alignment errors and data loss during 3D-to-2.5D projection.
4. **Real-time visualization and performance:** semantic elevation mapping, memory reduction against a uniform high-resolution representation, low latency/high FPS and classification accuracy at varying distances.

The proposed pipeline includes a point-cloud semantic segmentation network, such as PointNet++ or sparse convolution; a variable resolution grid engine; and a dashboard that exposes results and measurements. The existing point-cloud viewer is the strongest part of the current UI. Keep its underlying renderer and LiDAR logic when integrating the redesigned shell.

> **Data note:** All numbers below, including FPS, latency, memory, cell counts, detection counts, percentages and benchmark results, are illustrative UI examples from the design plan. Replace them with measured values and make the baseline and measurement method explicit before presenting them as experimental evidence. The suggested ring sizes are design inputs, not measured outcomes.

## 1. Change the overall structure

Current structure: **top navbar → small stats bar → huge LiDAR map → tabs**.

Target structure:

```text
┌───────────────┬─────────────────────────────────────────────────┐
│               │ FoveaMap                       LIVE ●  Sequence │
│   SIDEBAR     ├─────────────────────────────────────────────────┤
│               │  FPS       Latency      Memory      Points     │
│ Dashboard     │  5.3       189ms        12.9MB      124K       │
│ Live Map      ├─────────────────────────────────────────────────┤
│ Objects       │                                                 │
│ Terrain       │               LiDAR MAP                         │
│ Performance   │                                                 │
│ Benchmark     │                                                 │
│ Settings      │                                                 │
│               ├───────────────────────────┬─────────────────────┤
│               │ Resolution / Rings        │ Detected Objects    │
│               │ R0 5cm  0–10m            │ Cars       7        │
│               │ R1 10cm 10–30m           │ Pedestrian 3        │
│               │ R2 25cm 30–60m           │ Pole       11       │
└───────────────┴───────────────────────────┴─────────────────────┘
```

This gives the robotics/perception dashboard a clear hierarchy and makes the project's adaptive mapping contribution visible alongside the map.

## 2. Sidebar

Use a narrow, collapsible sidebar with a small **FoveaMap** logo at the top.

Primary navigation:

- Overview
- Live Perception
- Semantic Map
- Object Detection
- Terrain Analysis
- Elevation Map
- Performance
- Benchmarks

Near the bottom:

- Dataset / Sequence
- Settings
- About model

Suggested Lucide icons: `LayoutDashboard`, `Radar`, `ScanLine`, `Car`, `Mountain`, `Layers3`, `Gauge`, `ChartNoAxesCombined`, `Settings`.

The selected v0 sidebar template supplies much of the application shell.

## 3. Redo the header

The current header includes many similarly prominent items: `SEQUENCE 08`, `ORACLE`, `PRESET`, `PERFORMANCE DASHBOARD`, `DISPLAY FPS` and `Connected`.

A clearer hierarchy:

```text
Live Perception
Sequence 08
KITTI / SemanticKITTI

                         ● LIVE     Ground Truth ▼     Fovea Adaptive ▼
```

Then a concise description:

```text
Adaptive 2.5D Mapping
4 resolution rings • 5 cm → 50 cm • 100 m range
```

Keep **LIVE / Connected** green and avoid excessive pills.

## 4. Add four primary metric cards

Position these immediately above the point-cloud viewer:

| Card | Example value | Supporting comparison/context |
| --- | ---: | --- |
| Pipeline FPS | 5.3 FPS | ↑ 14% |
| Pipeline latency | 189 ms | Inference / pipeline timing, explicitly labeled |
| Map memory | 12.9 MB | ↓ 87.6% versus a uniform 5 cm grid |
| Active cells | 48,293 | / 4.2M uniform cells |

For memory, show both **12.9 MB** and **87.6% lower than uniform 5 cm grid**. This exposes the project's main contribution immediately. Connect the comparison to the actual measured baseline before claiming it in a demo.

## 5. Keep the LiDAR visualization as the hero

Give the point-cloud map roughly **65–70% of the main viewport**. Example:

```text
LIVE SEMANTIC MAP                     Top View ▼    ◉ Center Vehicle

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                   POINT CLOUD MAP                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Place compact viewer controls in the upper-right of the viewer: zoom in `+`, zoom out `−`, Fit, Center and Layers. Preserve the existing WebGL/Canvas visualization in this region rather than replacing the real map with a v0 placeholder.

## 6. Replace the current tabs with map controls

The current large tabs are **Terrain Analysis**, **Object Detection**, **Traversability**, **Semantic**, **Elevation** and **Confidence**. Convert map overlays to a **Layers** control:

```text
Layers
☑ Semantic
☑ Resolution rings
☑ Objects
☑ Vehicle
☐ Elevation
☐ Confidence
☐ Traversability
```

Add a smaller map mode selector: **Semantic | Elevation | Confidence**. Terrain Analysis and Object Detection can also be dedicated perception views or panels; they need not compete as large map tabs.

## 7. Explain adaptive resolution rings

The rings communicate the central innovation, so provide a readable legend:

| Ring | Range | Cell resolution | Role |
| --- | --- | --- | --- |
| R0 | 0–10 m | 5 cm / cell | Near-field safety zone |
| R1 | 10–30 m | 10 cm / cell | Intermediate detail |
| R2 | 30–60 m | 25 cm / cell | Reduced detail |
| R3 | 60–100 m | 50 cm / cell | Far-field coverage |

Show a colored indicator for each ring. On hover or selection, show its range, resolution, grid cells, point count, object count and role. Example:

```text
R0 — Safety Zone
Range          0–10 m
Resolution     5 cm
Grid cells     160,000
Points         38,451
Objects        4

Highest-resolution region used for immediate collision awareness.
```

The ring shapes and legend should align with the actual grid engine's geometry; sample cell counts must be derived from the implemented grid.

## 8. Add an object detection panel

Put this beside the map, with a total and counts by class:

```text
OBJECTS                         24 detected
Vehicles                         8
Pedestrians                      3
Poles                            9
Walls                            4
```

Include a selectable list with distance, classification confidence and motion state:

```text
Vehicle #07        12.4 m       Confidence 97.3%       Dynamic
Pedestrian #03      8.7 m       Confidence 94.8%       Dynamic
Pole #11           21.8 m       Confidence 91.2%       Static
```

Clicking an object highlights its location in the map. This makes the Object Detection requirement apparent in the demo.

## 9. Add terrain classification stats

A small panel can report:

| Terrain class | Example share |
| --- | ---: |
| Drivable | 73.4% |
| Non-drivable | 18.2% |
| Obstacle | 6.1% |
| Unknown | 2.3% |

A horizontal segmented bar gives an at-a-glance view of the composition. Keep the displayed percentages synchronized with the semantic segmentation output and clearly define whether the denominator is points, cells or area.

## 10. Create a dedicated Performance page

This screen gives evidence for the problem statement's latency, throughput, memory and accuracy requirements.

Top metrics: example **FPS 5.3**, **latency 189 ms**, **GPU memory 1.8 GB**, **map memory 12.9 MB**.

Charts:

1. Pipeline FPS over time.
2. Pipeline latency over time.
3. Memory usage over time.
4. Pipeline-stage latency breakdown.

Example stage breakdown:

| Stage | Example latency |
| --- | ---: |
| Point loading | 18 ms |
| Segmentation | 71 ms |
| Projection | 29 ms |
| Grid aggregation | 34 ms |
| Rendering | 22 ms |

Include classification accuracy by class and distance when real evaluation results are available. Label whether the displayed latency is inference only or end to end; stage timings and totals should come from the same measured runs.

## 11. Add a Foveated vs Uniform comparison page

Call this page **Foveated vs Uniform**. It is a strong demo of why the representation matters.

| Metric | Uniform 5 cm example | FoveaMap example |
| --- | ---: | ---: |
| Map cells | 16M | 1.9M |
| Memory | 128 MB | 12.9 MB |
| Processing latency | 430 ms | 189 ms |
| FPS | 2.3 | 5.3 |
| Near-field resolution | 5 cm | 5 cm |
| Max range | 100 m | 100 m |

Prominent callout:

```text
89.9%
MEMORY REDUCTION

while retaining 5 cm near-field precision
```

These numbers are an example for layout only. Some sample metrics in the original plan use different baselines or ratios. Recompute the cells, memory saving, FPS and percentages from one consistent dataset, map extent, data representation and hardware configuration before publication.

## 12. Color system

Use a dark perception-console theme:

| UI element | Suggested color |
| --- | --- |
| Background | `#090D12` |
| Panel | `#11171F` |
| Card | `#151C25` |
| Border | `#26313D` |
| Primary | Cyan / blue |
| Healthy | Green |
| Warning | Amber |
| Danger | Red |

Semantic colors:

| Class | Color |
| --- | --- |
| Drivable | Cyan |
| Vehicle | Blue |
| Pedestrian | Magenta |
| Vegetation | Green |
| Structure | Orange |
| Unknown | Gray |

Keep most of the interface neutral so the colored point cloud carries the visual emphasis. Avoid excessive neon, gradients and glass effects. A dark dashboard example mentioned in the earlier plan is [this v0 example](https://v0.app/abhishek-s-projects-06581bd8/chat/dark-mode-dashboard-jJYCzd961bU).

## 13. Typography

Use **Inter** or **Geist** for the interface and **Geist Mono** for numeric values, such as:

```text
Pipeline FPS
5.3

Latency
189.8 ms
```

This makes metrics legible with a technical feel without turning the whole UI into a terminal.

## 14. Choose and integrate a v0 template

**Selected template:** [Dashboard Sidebar Layout](https://v0.app/templates/dashboard-sidebar-layout-3kfXz3NHa1F). It provides a collapsible sidebar, navigation structure, shadcn/ui components and a clean desktop content region without business-specific dashboard elements. Convert the shell to the dark theme described above.

**Alternative:** [Dashboard Application Shell](https://v0.app/templates/dashboard-application-shell-4nUQAugbwJa), if a more minimal starting point is desired. The preferred choice remains **Dashboard Sidebar Layout**.

### How to use the template with the existing project

1. Open and duplicate **Dashboard Sidebar Layout** in v0.
2. Send the prompt in the next section to generate the FoveaMap dashboard shell and components.
3. Review the generated views, layout and responsive behavior.
4. Bring the generated shell/components into the project's frontend using the project's actual framework and package versions. If the existing frontend is not Next.js, adapt the generated components rather than replacing the application setup blindly.
5. Replace the generated map placeholder with the **existing WebGL/Canvas LiDAR visualization**. Keep its data pipeline, renderer, interaction logic and coordinate conventions.
6. Connect the cards, selectors, resolution rings and object/terrain panels to the current data source and backend/WebSocket as appropriate.
7. Make the Performance and Benchmark views consume real measured data and show metadata for the sequence, model, hardware and uniform-grid baseline.

v0 supplies the UI shell and components; the existing LiDAR viewer supplies the actual map.

## 15. Prompt to paste into v0

> Transform this dashboard into a premium autonomous vehicle LiDAR perception dashboard named **FoveaMap**.
>
> FoveaMap is an Adaptive Variable Resolution 2.5D LiDAR Mapping system. It converts raw 3D point clouds into semantic elevation grids using four distance-dependent resolution rings.
>
> Use a dark engineering/robotics visual style inspired by autonomous driving perception systems and professional observability dashboards. Avoid excessive gradients and glassmorphism. Use dark charcoal panels, subtle borders, blue/cyan primary accents, green status indicators, amber warnings and semantic colours.
>
> Build a collapsible left sidebar with: Overview, Live Perception, Semantic Map, Object Detection, Terrain Analysis, Elevation, Performance, Benchmarks and Settings.
>
> Create a top header showing: FoveaMap logo, Sequence 08, dataset selector, Ground Truth/Prediction selector, Fovea Adaptive preset selector and a green LIVE connection indicator.
>
> On the Overview/Live Perception screen create four KPI cards: Pipeline FPS, Pipeline Latency, Map Memory and Active Grid Cells. Include comparison values against a uniform 5 cm grid.
>
> The primary component should be a very large LiDAR semantic map viewer occupying approximately 65–70% of the central dashboard. Leave a clearly defined component region where an existing WebGL/Canvas LiDAR visualization can be inserted.
>
> Include floating viewer controls for zoom, reset view, center vehicle and layer selection.
>
> Visualize four adaptive resolution zones:
> R0: 0–10 m, 5 cm/cell
> R1: 10–30 m, 10 cm/cell
> R2: 30–60 m, 25 cm/cell
> R3: 60–100 m, 50 cm/cell.
>
> Add an Adaptive Resolution panel showing these rings with colour indicators and statistics.
>
> Add an Object Detection panel showing total detections and counts for vehicles, pedestrians, poles and walls, with confidence scores and static/dynamic badges.
>
> Add a Terrain panel showing percentages for Drivable, Non-drivable, Obstacle and Unknown terrain.
>
> Create a Performance screen with FPS-over-time graph, latency graph, memory usage graph and pipeline-stage latency breakdown.
>
> Create a Benchmarks screen comparing FoveaMap against a uniform 5 cm grid using Map Cells, Map Memory, Processing Latency, FPS, Near-field Resolution and Maximum Range.
>
> Highlight memory reduction and latency improvement prominently.
>
> Use Next.js, TypeScript, Tailwind, shadcn/ui, Lucide icons and Recharts. Components must be modular and ready for real backend/WebSocket data later.
>
> Keep the interface information-dense but clean and appropriate for a university engineering / autonomous robotics project.

This prompt aims to produce most of the visual redesign in one generation. Treat any generated map or sample statistics as placeholders until the existing renderer and real metrics are connected.

## Suggested development order

1. Duplicate **Dashboard Sidebar Layout**.
2. Send the full v0 prompt above.
3. Generate and review the FoveaMap dashboard.
4. Insert the existing LiDAR visualization into the central map component.
5. Connect existing values such as FPS, latency, memory, sequence and preset.
6. Connect semantic/object counts.
7. Build Performance graphs.
8. Add Uniform vs Foveated benchmark mode.
9. Add animations, ring/object hover tooltips and final polish.

The current frontend already demonstrates the map. The redesigned frontend should also demonstrate high precision nearby, progressively cheaper representation farther away, semantic understanding and measurable performance savings.
