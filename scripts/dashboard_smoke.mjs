// Frontend smoke test against a separately launched headless Chromium browser.
// Start Chromium with --headless=new --remote-debugging-port=9223 and an isolated profile.
import fs from 'node:fs/promises';
const origin = process.env.DASHBOARD_URL || 'http://127.0.0.1:5173';
const tabs = await (await fetch('http://127.0.0.1:9223/json/list')).json();
const tab = tabs.find(t => t.type === 'page' && !t.url.startsWith('edge:'));
if (!tab) throw new Error('No test browser page');
const ws = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise(resolve => ws.addEventListener('open', resolve, { once: true }));
let sequence = 0;
const pending = new Map(), errors = [];
ws.addEventListener('message', e => {
  const m = JSON.parse(e.data);
  if (m.id) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(m.error) : p.resolve(m.result); }
  if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails);
});
function send(method, params = {}) {
  return new Promise((resolve, reject) => { const id = ++sequence; pending.set(id, { resolve, reject }); ws.send(JSON.stringify({ id, method, params })); });
}
const evaluate = async expression => (await send('Runtime.evaluate', { expression, returnByValue: true })).result.value;
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
await send('Runtime.enable');
await send('Page.enable');
await fs.mkdir('results/cache/ui', { recursive: true });
const report = [];
for (const [width, height] of [[1920,1080],[390,844]]) {
  await send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  for (const view of ['overview','objects','terrain','elevation','performance','benchmarks']) {
    await send('Page.navigate', { url: `${origin}/#${view}` });
    await wait(2500);
    const metrics = await evaluate(`({title:document.querySelector('h1')?.textContent, overflow:document.documentElement.scrollWidth > innerWidth, canvas:!!document.querySelector('canvas'), text:document.body.innerText.slice(0,350)})`);
    const shot = await send('Page.captureScreenshot', { format: 'png' });
    await fs.writeFile(`results/cache/ui/${view}-${width}.png`, Buffer.from(shot.data,'base64'));
    report.push({ width, view, ...metrics });
  }
}
await send('Page.navigate', { url: `${origin}/#terrain` });
await wait(1500);
const controls = await evaluate(`(() => {
 const select=document.querySelector('#map-mode');
 const setter=Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set;
 setter.call(select,'confidence');select.dispatchEvent(new Event('change',{bubbles:true}));
 const buttons=[...document.querySelectorAll('button')];
 buttons.find(b=>b.textContent==='Layers').click();
 buttons.find(b=>b.textContent==='+').click();
 buttons.find(b=>b.textContent==='Reset').click();
 return {mode:select.value, zoomAndReset:true};
})()`);
await wait(100);
controls.layers = await evaluate(`!!document.querySelector('.layers-popover')`);
await fs.writeFile('results/cache/ui/smoke.json', JSON.stringify({ report, controls, errors },null,2));
console.log(JSON.stringify({report,controls,errors},null,2));
ws.close();
if (errors.length || report.some(r=>r.overflow || !r.title) || !controls.layers) process.exitCode=1;
