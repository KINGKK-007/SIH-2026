import type { ObjectBox } from "../types";
import { OBJECT_GROUPS } from "./objectPalette";

const VEHICLE_CLASSES = new Set(["vehicle", "car", "bicycle", "bicyclist", "bus", "motorcycle", "motorcyclist", "on-rails", "truck", "other-vehicle"]);

export const VEHICLE_COLORS = {
  static: { line: OBJECT_GROUPS[2].color, fill: "rgba(33, 150, 243, 0.20)" },
  moving: { line: OBJECT_GROUPS[3].color, fill: "rgba(255, 51, 102, 0.24)" },
} as const;

export function objectDisplayGroup(obj: ObjectBox): number {
  return isVehicleObject(obj) ? (obj.moving ? 3 : 2) : obj.moving ? 5 : 4;
}

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
