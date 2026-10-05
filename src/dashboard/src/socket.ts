import { io, Socket } from "socket.io-client";

// VITE_SERVER_URL points at the deployed backend (e.g. Render) when frontend and
// backend are on different origins. Falls back to same-origin / localhost:8000 for
// local dev where Vite and uvicorn run side by side.
export const SERVER_URL =
  import.meta.env.VITE_SERVER_URL ??
  (import.meta.env.DEV
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin);

export const socket: Socket = io(`${SERVER_URL}/fovea`, {
  path: "/socket.io",
  transports: ["websocket", "polling"],
  reconnection: true,
  reconnectionAttempts: 10,
  reconnectionDelay: 1000,
  auth: typeof DecompressionStream !== "undefined" ? { frame_codec: "gzip-json-v1" } : undefined,
});
