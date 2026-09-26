import type { ObjectBox } from "../types";

const VEHICLE_CLASSES = new Set(["vehicle", "car", "bicycle", "bicyclist", "bus", "motorcycle", "motorcyclist", "on-rails", "truck", "other-vehicle"]);

// Static  → electric lime-green  (#50FF82)  matches point-cloud cls=3 color
// Moving  → hot neon magenta     (#FF32A0)  matches point-cloud cls=4 color
export const VEHICLE_COLORS = {
  static: { line: "#35E1F3", fill: "rgba(53, 225, 243, 0.18)" },
  moving: { line: "#FF7954", fill: "rgba(255, 121, 84, 0.22)" },
} as const;

export function isVehicleObject(obj: ObjectBox): boolean {
  return VEHICLE_CLASSES.has(obj.cls_name.replace(/^moving-/, "").toLowerCase());
}

export function isPersonObject(obj: ObjectBox): boolean {
  return ["person", "pedestrian"].includes(obj.cls_name.replace(/^moving-/, "").toLowerCase());
}

export function isDrawableObject(obj: ObjectBox): boolean {
  return obj.size[0] > 0 && obj.size[1] > 0 && obj.size[0] <= 15 && obj.size[1] <= 15
    && obj.center.every(Number.isFinite) && Number.isFinite(obj.yaw);
}
