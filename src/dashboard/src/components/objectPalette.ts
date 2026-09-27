// Six display groups aligned with foveamap.pipeline.display. The navigation
// superclasses and underlying adaptive grid are not changed by this palette.
export const OBJECT_GROUPS = [
  { label: "Unclassified",           color: "#7C8FA8", rgba: [124, 143, 168, 0.80] },
  { label: "Drivable terrain",       color: "#00DC96", rgba: [  0, 220, 150, 0.97] },
  { label: "Static vehicles",        color: "#2196F3", rgba: [ 33, 150, 243, 0.98] },
  { label: "Moving vehicles",        color: "#FF3366", rgba: [255,  51, 102, 0.98] },
  { label: "Other static objects",   color: "#FF6EC7", rgba: [255, 110, 199, 0.96] },
  { label: "Other classified points",color: "#FFD600", rgba: [255, 214,   0, 0.96] },
] as const;
