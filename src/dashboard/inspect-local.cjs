const WebSocket = require("ws");
const fs = require("fs");
const { io } = require("socket.io-client");

(async () => {
  const tabs = await (await fetch("http://127.0.0.1:9224/json/list")).json();
  const tab = tabs.find((entry) => entry.url.includes("127.0.0.1:8000"))
    ?? tabs.find((entry) => entry.url === "about:blank");
  if (!tab) throw new Error("Local FoveaMap browser tab not found");
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { ws.once("open", resolve); ws.once("error", reject); });
  let nextId = 0;
  const pending = new Map();
  ws.on("message", (buffer) => {
    const message = JSON.parse(buffer);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    }
  });
  const send = (method, params = {}) => new Promise((resolve) => {
    const id = ++nextId;
    pending.set(id, resolve);
    ws.send(JSON.stringify({ id, method, params }));
  });
  if (process.argv.includes("--blank")) {
    await send("Page.navigate", { url: "about:blank" });
    ws.close();
    return;
  }
  const mobile = process.argv.includes("--mobile");
  await send("Emulation.setDeviceMetricsOverride", { width: mobile ? 390 : 1920, height: mobile ? 844 : 1080, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: "http://127.0.0.1:8000/#objects" });
  await new Promise((resolve) => setTimeout(resolve, 5000));
  const seekArg = process.argv.find((arg) => arg.startsWith("--seek="));
  if (seekArg) {
    const socket = io("http://127.0.0.1:8000/fovea", { transports: ["websocket"] });
    await new Promise((resolve) => socket.on("connect", resolve));
    socket.emit("seek_frame", { frame_idx: Number(seekArg.slice(7)) });
    await new Promise((resolve) => setTimeout(resolve, 1500));
    socket.disconnect();
  }
  const evaluation = await send("Runtime.evaluate", {
    expression: "({title:document.title,frame:document.querySelector('.frame-status-bar')?.textContent,canvas:[...document.querySelectorAll('canvas')].map(c=>[c.width,c.height]),arrows:document.querySelectorAll('.velocity-arrow').length,overflow:document.documentElement.scrollWidth>innerWidth,status:document.body.innerText.slice(0,400)})",
    returnByValue: true,
  });
  console.log(evaluation.result.result.value);
  if (process.argv.includes("--benchmark")) {
    await send("Runtime.evaluate", { expression: `window._fmBench={changes:[],longTasks:[]};new MutationObserver(()=>window._fmBench.changes.push({frame:document.querySelector('.frame-status-bar')?.textContent,time:performance.now()})).observe(document.querySelector('.frame-status-bar'),{childList:true,subtree:true,characterData:true});new PerformanceObserver(list=>window._fmBench.longTasks.push(...list.getEntries().map(e=>Math.round(e.duration)))).observe({entryTypes:['longtask']});` });
    const socket = io("http://127.0.0.1:8000/fovea", { transports: ["websocket"] });
    await new Promise((resolve) => socket.on("connect", resolve));
    const received = [];
    socket.on("frame_update", (frame) => received.push({idx: frame.frame_idx, time: Date.now(), ms: Math.round(Object.values(frame.timings_ms).reduce((a,b)=>a+b,0))}));
    socket.emit("play");
    await new Promise((resolve) => setTimeout(resolve, 7000));
    socket.emit("pause");
    socket.disconnect();
    const bench = await send("Runtime.evaluate", { expression: "window._fmBench", returnByValue: true });
    console.log({ received, browser: bench.result.result.value });
  }
  const screenshot = await send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(mobile ? "../../results/cache/ui/current-objects-mobile.png" : "../../results/cache/ui/current-objects.png", Buffer.from(screenshot.result.data, "base64"));
  ws.close();
})().catch((error) => { console.error(error); process.exitCode = 1; });
