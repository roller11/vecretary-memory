import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const {script, scenario} = JSON.parse(readFileSync(0, 'utf8'));
const calls = [];
let heartbeatCount = 0;
let sessionCount = 0;
let domReadyHandler = null;
let preferenceChangeHandler = null;
const preferenceValues = new Map(scenario === 'read_only' ? [['instinct:preferReadOnly', '1']] : []);
const readOnlyPreference = {
  checked: false,
  addEventListener(event, handler) {
    if (event === 'change') preferenceChangeHandler = handler;
  },
};
const context = vm.createContext({
  URL, Headers, Response,
  setInterval() { return 1; },
  location: {href: 'http://127.0.0.1:9876/', origin: 'http://127.0.0.1:9876'},
  document: {
    addEventListener(event, handler) { if (event === 'DOMContentLoaded') domReadyHandler = handler; },
    querySelectorAll() { return []; },
    getElementById(id) { return id === 'instinct-prefer-readonly' ? readOnlyPreference : null; },
  },
  localStorage: {
    getItem(key) { return preferenceValues.get(key) ?? null; },
    setItem(key, value) { preferenceValues.set(key, String(value)); },
    removeItem(key) { preferenceValues.delete(key); },
  },
  async fetch(path, options = {}) {
    const headers = new Headers(options.headers);
    calls.push({path, options, capability: headers.get('X-Instinct-Capability')});
    if (path === '/api/session') {
      sessionCount += 1;
      assert.equal(options.cache, 'no-store');
      assert.equal(options.credentials, 'same-origin');
      assert.equal(options.redirect, 'error');
      assert.equal(headers.has('X-Instinct-Capability'), false);
      if (scenario === 'concurrent') await new Promise(resolve => setTimeout(resolve, 10));
      return Response.json({capability: scenario === 'invalid_capability' ? 123 : 'synthetic-renewed'},
        {status: scenario === 'bootstrap_denied' ? 403 : 200});
    }
    if (path === '/api/mode') return Response.json({read_only: false});
    assert.equal(new URL(path, context.location.href).origin, context.location.origin);
    if (path === '/api/heartbeat') {
      heartbeatCount += 1;
      if (scenario === 'success' || headers.get('X-Instinct-Capability') === 'synthetic-renewed' && scenario !== 'retry_denied') {
        return Response.json({ok: true});
      }
    }
    return Response.json({error: scenario === 'origin_denied' ? 'invalid dashboard Origin' : 'dashboard capability required'}, {status: 403});
  },
});
vm.runInContext(script, context);
vm.runInContext('instinctReadOnly = false', context);
const invoke = expression => vm.runInContext(expression, context);

if (scenario === 'cross_origin') {
  await assert.rejects(invoke("dashboardFetch('https://attacker.example/api/heartbeat', {method:'POST'})"), /Cross-origin/);
  assert.equal(calls.length, 0);
} else if (scenario === 'read_only') {
  await domReadyHandler();
  assert.equal(readOnlyPreference.checked, true);
  await assert.rejects(invoke("dashboardFetch('/api/add', {method:'POST'})"), /Read-only/);
  assert.equal(calls.some(call => call.path === '/api/add'), false);
} else if (scenario === 'preference_toggle') {
  await domReadyHandler();
  assert.equal(readOnlyPreference.checked, false);
  readOnlyPreference.checked = true;
  preferenceChangeHandler();
  assert.equal(preferenceValues.get('instinct:preferReadOnly'), '1');
  assert.equal(invoke('instinctCanWrite()'), false);
  readOnlyPreference.checked = false;
  preferenceChangeHandler();
  assert.equal(preferenceValues.has('instinct:preferReadOnly'), false);
  assert.equal(invoke('instinctCanWrite()'), true);
} else if (scenario === 'edit_denied' || scenario === 'get_denied') {
  const response = await invoke(scenario === 'edit_denied' ? "dashboardFetch('/api/add', {method:'POST', body:'{}'})" : "dashboardFetch('/api/heartbeat')");
  assert.equal(response.status, 403);
  assert.equal(calls.length, 1);
  assert.equal(sessionCount, 0);
} else if (scenario === 'bootstrap_denied' || scenario === 'invalid_capability') {
  await assert.rejects(invoke("dashboardFetch('/api/heartbeat', {method:'POST'})"), /reconnect failed/);
  assert.equal(heartbeatCount, 1);
  assert.equal(sessionCount, 1);
} else if (scenario === 'concurrent') {
  const responses = await invoke("Promise.all([dashboardFetch('/api/heartbeat', {method:'POST'}), dashboardFetch('/api/heartbeat', {method:'POST'})])");
  assert(responses.every(response => response.status === 200));
  assert.equal(sessionCount, 1);
  assert.equal(heartbeatCount, 4);
} else {
  const response = await invoke("dashboardFetch('/api/heartbeat', {method:'POST'})");
  assert.equal(response.status, ['origin_denied', 'retry_denied'].includes(scenario) ? 403 : 200);
  assert.equal(sessionCount, ['origin_denied', 'success'].includes(scenario) ? 0 : 1);
  assert.equal(heartbeatCount, ['origin_denied', 'success'].includes(scenario) ? 1 : 2);
  if (scenario === 'recovery') {
    const fresh = await invoke("dashboardFetch('/api/heartbeat', {method:'POST'})");
    assert.equal(fresh.status, 200);
    assert.equal(sessionCount, 1);
    assert.equal(calls.at(-1).capability, 'synthetic-renewed');
  }
}
