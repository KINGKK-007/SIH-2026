// Six display groups aligned with foveamap.pipeline.display. The navigation
// superclasses and underlying adaptive grid are not changed by this palette.
export const OBJECT_GROUPS = [
  { label: "Unclassified", color: "#9BA6B2", rgba: [155, 166, 178, 0.86] },
  { label: "Drivable terrain", color: "#3DD6A1", rgba: [61, 214, 161, 0.96] },
  { label: "Static vehicles", color: "#447CC8", rgba: [68, 124, 200, 0.98] },
  { label: "Moving vehicles", color: "#C34D58", rgba: [195, 77, 88, 0.98] },
  { label: "Other static objects", color: "#E478CF", rgba: [228, 120, 207, 0.96] },
  { label: "Other classified points", color: "#F0BD63", rgba: [240, 189, 99, 0.95] },
] as const;
