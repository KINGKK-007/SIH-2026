import { io, Socket } from "socket.io-client";

// In Vite dev mode, connect to backend on port 8000 even if Vite picks another port.
// In production or when served by backend, connect to same host.
export const SERVER_URL =
  import.meta.env.DEV
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : window.location.origin;

export const socket: Socket = io(`${SERVER_URL}/fovea`, {
  path: "/socket.io",
  transports: ["websocket", "polling"],
  reconnection: true,
  reconnectionAttempts: 10,
  reconnectionDelay: 1000,
});
