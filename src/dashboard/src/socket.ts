import { io, Socket } from "socket.io-client";

// In dev mode with Vite on port 5173, connect to backend on port 8000.
// In production or when served by backend, connect to same host.
const SERVER_URL =
  window.location.port === "5173"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

export const socket: Socket = io(`${SERVER_URL}/fovea`, {
  path: "/socket.io",
  transports: ["websocket", "polling"],
  reconnection: true,
  reconnectionAttempts: 10,
  reconnectionDelay: 1000,
});
