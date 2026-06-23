const fs = require('fs');
const path = require('path');

if (process.argv.includes('--help')) {
  console.log('Usage: electron electron/main.js --data-dir <path>');
  process.exit(0);
}

const { app, BrowserWindow, Menu, ipcMain, screen } = require('electron');

const POLL_MS = 700;
const PET_WIDTH = 190;
const PET_HEIGHT = 190;
const BUBBLE_WIDTH = 320;
const BUBBLE_HEIGHT = 152;
const HUD_WIDTH = 320;
const HUD_HEIGHT = 390;
const EDGE_MARGIN = 8;
const MINI_VISIBLE_WIDTH = 58;

let dataDir = '';
let petDir = '';
let petWindow = null;
let bubbleWindow = null;
let hudWindow = null;
let latestState = null;
let lastSignature = '';
let bubbleTimer = null;
let pollTimer = null;
let smokeCaptureDir = '';
let smokeCaptureTimer = null;

function parseArgs(argv) {
  const args = argv.slice(2);
  const index = args.indexOf('--data-dir');
  if (index >= 0 && args[index + 1]) {
    dataDir = path.resolve(args[index + 1]);
    petDir = path.join(dataDir, 'pet');
  }
  const smokeIndex = args.indexOf('--smoke-capture-dir');
  if (smokeIndex >= 0 && args[smokeIndex + 1]) {
    smokeCaptureDir = path.resolve(args[smokeIndex + 1]);
  }
  if (!dataDir) {
    console.error('error: --data-dir is required');
    process.exit(2);
  }
}

function readJson(file, fallback = {}) {
  try {
    const payload = JSON.parse(fs.readFileSync(file, 'utf8'));
    return payload && typeof payload === 'object' && !Array.isArray(payload) ? payload : fallback;
  } catch (_error) {
    return fallback;
  }
}

function writeJson(file, payload) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  const temporary = `${file}.${process.pid}.${Date.now()}.tmp`;
  try {
    fs.writeFileSync(temporary, JSON.stringify(payload) + '\n', 'utf8');
    fs.renameSync(temporary, file);
  } catch (error) {
    try {
      fs.unlinkSync(temporary);
    } catch (_unlinkError) {
      // Best effort cleanup; keep the original state file intact.
    }
    throw error;
  }
}

function appendAction(action, payload) {
  fs.mkdirSync(petDir, { recursive: true });
  const record = {
    action,
    created_at: new Date().toISOString(),
    payload: payload || {}
  };
  fs.appendFileSync(path.join(petDir, 'actions.jsonl'), JSON.stringify(record) + '\n', 'utf8');
}

function petStatePath() {
  return path.join(petDir, 'pet_state.json');
}

function petProfilePath() {
  return path.join(petDir, 'pet_profile.json');
}

function petLearningStatePath() {
  return path.join(petDir, 'pet_learning_state.json');
}

function uiStatePath() {
  return path.join(petDir, 'ui_state.json');
}

function loadHostState() {
  return readJson(petStatePath(), {});
}

function loadProfile() {
  return readJson(petProfilePath(), {});
}

function saveProfile(extra = {}) {
  const current = loadProfile();
  writeJson(petProfilePath(), {
    ...current,
    ...extra
  });
}

function saveHostState(extra = {}) {
  if (!petWindow) return;
  const [x, y] = petWindow.getPosition();
  saveHostStateRecord(extra, { x, y });
}

function saveHostStateRecord(extra = {}, position = null) {
  const current = loadHostState();
  const resolvedPosition = position || {};
  writeJson(petStatePath(), {
    ...current,
    ...extra,
    ...(Number.isFinite(resolvedPosition.x) ? { x: resolvedPosition.x } : {}),
    ...(Number.isFinite(resolvedPosition.y) ? { y: resolvedPosition.y } : {}),
    width: PET_WIDTH,
    height: PET_HEIGHT
  });
}

function windowOptions(width, height, extra = {}) {
  return {
    width,
    height,
    show: false,
    frame: false,
    transparent: true,
    backgroundColor: '#00000000',
    resizable: false,
    hasShadow: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    focusable: false,
    ...extra,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      ...(extra.webPreferences || {})
    }
  };
}

function createWindows() {
  const host = loadHostState();
  const display = screen.getPrimaryDisplay().workArea;
  const x = clamp(Number.isInteger(host.x) ? host.x : display.x + display.width - PET_WIDTH - 34, display.x, display.x + display.width - PET_WIDTH);
  const y = clamp(Number.isInteger(host.y) ? host.y : display.y + Math.round(display.height * 0.22), display.y, display.y + display.height - PET_HEIGHT);

  petWindow = new BrowserWindow(windowOptions(PET_WIDTH, PET_HEIGHT));
  petWindow.setPosition(x, y);
  applyFloatingLevel(petWindow);
  petWindow.loadFile(path.join(__dirname, '..', 'renderer', 'pet.html'));
  petWindow.webContents.on('did-finish-load', () => sendPetState());
  petWindow.once('ready-to-show', () => petWindow.showInactive());
  petWindow.on('move', () => {
    if (bubbleWindow && bubbleWindow.isVisible()) positionBubble();
    if (hudWindow && hudWindow.isVisible()) positionHud();
  });
  petWindow.on('closed', () => {
    petWindow = null;
    if (bubbleWindow) bubbleWindow.close();
    if (hudWindow) hudWindow.close();
  });

  bubbleWindow = new BrowserWindow({
    ...windowOptions(BUBBLE_WIDTH, BUBBLE_HEIGHT),
    focusable: false
  });
  applyFloatingLevel(bubbleWindow);
  bubbleWindow.setIgnoreMouseEvents(true, { forward: true });
  bubbleWindow.loadFile(path.join(__dirname, '..', 'renderer', 'bubble.html'));
  bubbleWindow.webContents.on('did-finish-load', () => {
    if (latestState) {
      const bubble = latestState.bubble || {};
      bubbleWindow.webContents.send('pet-bubble', bubble);
    }
  });
  bubbleWindow.on('closed', () => {
    bubbleWindow = null;
  });

  hudWindow = new BrowserWindow(windowOptions(HUD_WIDTH, HUD_HEIGHT, { focusable: true }));
  applyFloatingLevel(hudWindow);
  hudWindow.loadFile(path.join(__dirname, '..', 'renderer', 'hud.html'));
  hudWindow.webContents.on('did-finish-load', () => sendHudState());
  hudWindow.on('blur', () => hideHud());
  hudWindow.on('closed', () => {
    hudWindow = null;
  });

  if (process.platform === 'darwin' && app.dock) {
    app.dock.hide();
  }
}

function applyFloatingLevel(win) {
  win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  try {
    win.setAlwaysOnTop(true, 'screen-saver');
  } catch (_error) {
    win.setAlwaysOnTop(true, 'floating');
  }
}

function clamp(value, min, max) {
  if (max < min) return min;
  return Math.min(Math.max(value, min), max);
}

function stateSignature(state) {
  const bubble = state.bubble || {};
  const profile = state.profile || {};
  const display = state.display || {};
  const local = state.local || {};
  return JSON.stringify({
    event_id: state.event_id,
    kind: state.kind,
    state: state.state,
    severity: state.severity,
    workspace: state.workspace,
    session_id: state.session_id,
    title: state.title,
    summary: state.summary,
    refs: Array.isArray(state.refs) ? state.refs : [],
    actions: Array.isArray(state.actions) ? state.actions : [],
    bubble: {
      text: bubble.text,
      show: bubble.show,
      hold: bubble.hold,
      reason: bubble.reason,
      priority: bubble.priority,
      duration_ms: bubble.duration_ms
    },
    display: {
      reason: display.reason,
      priority: display.priority,
      hold: display.hold
    },
    profile: {
      pet_id: profile.pet_id,
      species: profile.species,
      style: profile.style,
      learning_mode: profile.learning_mode,
      muted_until: profile.muted_until
    },
    muted: state.muted,
    local: {
      muted: local.muted,
      permanentMuted: local.permanentMuted,
      timedMuted: local.timedMuted,
      mini: local.mini
    }
  });
}

function pollState() {
  const state = readJson(uiStatePath(), null);
  if (!state) return;
  const merged = mergeLocalState(state);
  const signature = stateSignature(merged);
  if (signature === lastSignature) return;
  lastSignature = signature;
  latestState = merged;
  if (petWindow) {
    sendPetState();
  }
  maybeShowBubble(merged);
  if (hudWindow && hudWindow.isVisible()) {
    sendHudState();
  }
  maybeRunSmokeCapture();
}

function sendPetState() {
  if (petWindow && latestState) {
    petWindow.webContents.send('pet-state', latestState);
  }
}

function sendHudState() {
  if (hudWindow) {
    hudWindow.webContents.send('pet-hud', latestState || {});
  }
}

function mergeLocalState(state) {
  const host = loadHostState();
  const backendProfile = (state && state.profile) || {};
  const localProfile = loadProfile();
  const profile = {
    ...backendProfile,
    ...localProfile,
    muted_until: localProfile.muted_until || backendProfile.muted_until
  };
  const timedMuted = mutedUntilActive(profile.muted_until);
  return {
    ...state,
    profile,
    muted: Boolean(host.muted || state.muted || timedMuted),
    mini: Boolean(host.mini || host.collapsed),
    local: {
      muted: Boolean(host.muted || timedMuted),
      permanentMuted: Boolean(host.muted),
      timedMuted,
      mini: Boolean(host.mini || host.collapsed)
    }
  };
}

function mutedUntilActive(value) {
  const stamp = String(value || '').trim();
  if (!stamp) return false;
  const until = Date.parse(stamp);
  return Number.isFinite(until) && until > Date.now();
}

function maybeShowBubble(state) {
  const bubble = state.bubble || {};
  if (!bubble.show || !bubble.text || state.muted) {
    hideBubble();
    return;
  }
  showBubble(state);
}

function positionBubble() {
  if (!petWindow || !bubbleWindow) return;
  const [x, y] = petWindow.getPosition();
  const display = screen.getDisplayNearestPoint({ x, y }).workArea;
  let bx = x - BUBBLE_WIDTH + 42;
  let by = y + 6;
  if (bx < display.x + 8) {
    bx = x + PET_WIDTH - 36;
  }
  by = clamp(by, display.y + 8, display.y + display.height - BUBBLE_HEIGHT - 8);
  bubbleWindow.setPosition(Math.round(bx), Math.round(by), false);
}

function showBubble(state) {
  if (!bubbleWindow) return;
  positionBubble();
  bubbleWindow.webContents.send('pet-bubble', state.bubble || {});
  bubbleWindow.showInactive();
  maybeRunSmokeCapture();
  if (bubbleTimer) clearTimeout(bubbleTimer);
  const bubble = state.bubble || {};
  if (!bubble.hold) {
    const duration = Number.isInteger(bubble.duration_ms) && bubble.duration_ms > 0 ? bubble.duration_ms : 7500;
    bubbleTimer = setTimeout(hideBubble, duration);
  }
}

function maybeRunSmokeCapture() {
  if (!smokeCaptureDir || smokeCaptureTimer || !latestState || !petWindow || !bubbleWindow || !hudWindow) {
    return;
  }
  smokeCaptureTimer = setTimeout(runSmokeCapture, 900);
}

async function runSmokeCapture() {
  try {
    fs.mkdirSync(smokeCaptureDir, { recursive: true });
    showBubble(latestState);
    showHud();
    await captureWindow(petWindow, path.join(smokeCaptureDir, 'pet.png'));
    await captureWindow(bubbleWindow, path.join(smokeCaptureDir, 'bubble.png'));
    await captureWindow(hudWindow, path.join(smokeCaptureDir, 'hud.png'));
    writeJson(path.join(smokeCaptureDir, 'state.json'), {
      pet: petWindow.getBounds(),
      bubble: bubbleWindow.getBounds(),
      hud: hudWindow.getBounds(),
      state: latestState.state,
      species: latestState.profile && latestState.profile.species,
      bubble_text: latestState.bubble && latestState.bubble.text
    });
    app.quit();
  } catch (error) {
    console.error(`smoke capture failed: ${error && error.message ? error.message : error}`);
    app.exit(3);
  }
}

async function captureWindow(win, file) {
  if (!win) return;
  const image = await win.webContents.capturePage();
  fs.writeFileSync(file, image.toPNG());
}

function hideBubble() {
  if (bubbleTimer) {
    clearTimeout(bubbleTimer);
    bubbleTimer = null;
  }
  if (bubbleWindow && bubbleWindow.isVisible()) bubbleWindow.hide();
}

function positionHud() {
  if (!petWindow || !hudWindow) return;
  const [x, y] = petWindow.getPosition();
  const display = screen.getDisplayNearestPoint({ x, y }).workArea;
  let hx = x - HUD_WIDTH + 34;
  let hy = y + 16;
  if (hx < display.x + 8) {
    hx = x + PET_WIDTH - 28;
  }
  hx = clamp(hx, display.x + 8, display.x + display.width - HUD_WIDTH - 8);
  hy = clamp(hy, display.y + 8, display.y + display.height - HUD_HEIGHT - 8);
  hudWindow.setPosition(Math.round(hx), Math.round(hy), false);
}

function showHud() {
  if (!hudWindow) return;
  positionHud();
  hudWindow.webContents.send('pet-hud', latestState || {});
  hudWindow.show();
}

function hideHud() {
  if (hudWindow && hudWindow.isVisible()) hudWindow.hide();
}

function toggleHud() {
  if (!hudWindow || !hudWindow.isVisible()) {
    showHud();
  } else {
    hideHud();
  }
}

function showMenu() {
  const host = loadHostState();
  const muted = Boolean(host.muted || mutedUntilActive(loadProfile().muted_until));
  const mini = Boolean(host.mini || host.collapsed);
  const menu = Menu.buildFromTemplate([
    {
      label: 'Open Current Work',
      click: () => openCurrentWork()
    },
    {
      label: muted ? 'Exit Quiet Mode' : 'Quiet 1h',
      click: () => toggleQuietOneHour()
    },
    {
      label: mini ? 'Exit Mini Mode' : 'Mini Mode',
      click: () => setMini(!mini)
    },
    {
      label: bubbleWindow && bubbleWindow.isVisible() ? 'Hide Bubble' : 'Show Status',
      click: () => {
        if (bubbleWindow && bubbleWindow.isVisible()) {
          hideBubble();
        } else if (latestState) {
          showBubble({
            bubble: {
              text: statusText(latestState),
              show: true,
              hold: false,
              duration_ms: 4500
            }
          });
        }
      }
    },
    { type: 'separator' },
    {
      label: 'Close',
      click: () => app.quit()
    }
  ]);
  menu.popup({ window: petWindow || undefined });
}

function setMini(enabled) {
  if (!petWindow) return;
  const display = screen.getDisplayNearestPoint(pointForWindow(petWindow)).workArea;
  const [, currentY] = petWindow.getPosition();
  const y = clamp(currentY, display.y, display.y + display.height - PET_HEIGHT);
  if (enabled) {
    petWindow.setSize(PET_WIDTH, PET_HEIGHT, false);
    const [currentX] = petWindow.getPosition();
    const leftDistance = Math.abs(currentX - display.x);
    const rightDistance = Math.abs(display.x + display.width - (currentX + PET_WIDTH));
    const x = leftDistance < rightDistance
      ? display.x - PET_WIDTH + MINI_VISIBLE_WIDTH
      : display.x + display.width - MINI_VISIBLE_WIDTH;
    petWindow.setPosition(x, y, false);
  } else {
    const [currentX] = petWindow.getPosition();
    const leftMini = currentX < display.x;
    const x = leftMini ? display.x + 24 : display.x + display.width - PET_WIDTH - 34;
    petWindow.setPosition(x, y, false);
  }
  saveHostState({ mini: enabled, collapsed: enabled });
  lastSignature = '';
  pollState();
  if (bubbleWindow && bubbleWindow.isVisible()) positionBubble();
  if (hudWindow && hudWindow.isVisible()) positionHud();
}

function toggleMuted() {
  const host = loadHostState();
  const muted = Boolean(host.muted || mutedUntilActive(loadProfile().muted_until));
  if (muted) {
    saveHostState({ muted: false });
    saveProfile({ muted_until: '' });
  } else {
    setTimedMuteHours(1);
  }
  hideBubble();
  lastSignature = '';
  pollState();
}

function toggleQuietOneHour() {
  toggleMuted();
}

function setTimedMuteHours(hours) {
  const durationMs = Math.max(1, Number(hours) || 1) * 60 * 60 * 1000;
  saveProfile({ muted_until: new Date(Date.now() + durationMs).toISOString() });
  saveHostState({ muted: false });
}

function recordFeedback(value) {
  const clean = String(value || '').trim();
  if (!['useful', 'less_often'].includes(clean)) return;
  const state = latestState || {};
  const bubble = state.bubble || {};
  const display = state.display || {};
  const reason = String(bubble.reason || display.reason || state.kind || '').trim();
  const learningState = readJson(petLearningStatePath(), {});
  const existingFeedback = Array.isArray(learningState.feedback)
    ? learningState.feedback
    : [];
  const record = {
    value: clean,
    created_at: new Date().toISOString(),
    event_id: String(state.event_id || ''),
    kind: String(state.kind || ''),
    reason,
    title: String(state.title || ''),
    summary: String(state.summary || '').slice(0, 240)
  };
  const nextState = {
    ...learningState,
    feedback: existingFeedback.concat(record).slice(-100)
  };
  if (clean === 'less_often' && reason) {
    const existingSuppression =
      learningState.suppressed_until && typeof learningState.suppressed_until === 'object'
        ? learningState.suppressed_until
        : {};
    nextState.suppressed_until = {
      ...existingSuppression,
      [reason]: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
    };
  }
  writeJson(petLearningStatePath(), nextState);
  showBubble({
    bubble: {
      text: clean === 'useful' ? '已记录：这个提醒有用。' : '已记录：同类提醒会减少。',
      show: true,
      hold: false,
      duration_ms: 3200
    }
  });
}

function pointForWindow(win) {
  const [x, y] = win.getPosition();
  return { x, y };
}

function openCurrentWork() {
  const state = latestState || {};
  const actions = Array.isArray(state.actions) ? state.actions : [];
  const hasAction = actions.some((action) => action && action.id === 'open_current_work');
  if (!hasAction && !state.session_id && !state.workspace) {
    showBubble({
      bubble: {
        text: '当前没有可打开的工作。',
        show: true,
        hold: false,
        duration_ms: 3200
      }
    });
    return;
  }
  appendAction('open_current_work', {
    session_id: state.session_id || '',
    workspace: state.workspace || '',
    title: state.title || ''
  });
  showBubble({
    bubble: {
      text: '已记录打开当前工作的请求。',
      show: true,
      hold: false,
      duration_ms: 3500
    }
  });
}

function statusText(state) {
  const summary = String((state && state.summary) || '').trim();
  const title = String((state && state.title) || '').trim();
  if (summary) return summary;
  if (title) return title;
  return 'Deepmate is ready.';
}

function constrainPetWindow() {
  if (!petWindow) return;
  const host = loadHostState();
  if (host.mini || host.collapsed) return;
  const display = screen.getDisplayNearestPoint(pointForWindow(petWindow)).workArea;
  const [x, y] = petWindow.getPosition();
  const nextX = clamp(x, display.x + EDGE_MARGIN, display.x + display.width - PET_WIDTH - EDGE_MARGIN);
  const nextY = clamp(y, display.y + EDGE_MARGIN, display.y + display.height - PET_HEIGHT - EDGE_MARGIN);
  if (nextX !== x || nextY !== y) {
    petWindow.setPosition(nextX, nextY, false);
  }
}

ipcMain.on('pet-drag-by', (_event, delta) => {
  if (!petWindow || !delta) return;
  const [x, y] = petWindow.getPosition();
  petWindow.setPosition(x + Math.round(delta.dx || 0), y + Math.round(delta.dy || 0), false);
  if (bubbleWindow && bubbleWindow.isVisible()) positionBubble();
  if (hudWindow && hudWindow.isVisible()) positionHud();
});

ipcMain.on('pet-finish-drag', () => {
  constrainPetWindow();
  saveHostState({ mini: false, collapsed: false });
});
ipcMain.on('pet-toggle-hud', () => toggleHud());
ipcMain.on('pet-close-hud', () => hideHud());
ipcMain.on('pet-hide-bubble', () => hideBubble());
ipcMain.on('pet-action-open-current-work', () => openCurrentWork());
ipcMain.on('pet-toggle-muted', () => toggleMuted());
ipcMain.on('pet-toggle-mini', () => {
  const host = loadHostState();
  setMini(!Boolean(host.mini || host.collapsed));
});
ipcMain.on('pet-feedback', (_event, value) => recordFeedback(value));
ipcMain.on('pet-quit', () => app.quit());
ipcMain.on('pet-show-menu', () => showMenu());
ipcMain.on('pet-reaction', (_event, reaction) => {
  if (petWindow) petWindow.webContents.send('pet-state', { ...(latestState || {}), reaction });
});
ipcMain.on('pet-mouse-region', (_event, active) => {
  if (!petWindow) return;
  petWindow.setIgnoreMouseEvents(!active, { forward: true });
});
ipcMain.on('pet-renderer-ready', (_event, surface) => {
  const name = String(surface || '');
  if (name === 'pet') {
    sendPetState();
  } else if (name === 'bubble' && latestState && bubbleWindow) {
    bubbleWindow.webContents.send('pet-bubble', latestState.bubble || {});
  } else if (name === 'hud') {
    sendHudState();
  }
});

parseArgs(process.argv);
app.whenReady().then(() => {
  createWindows();
  pollState();
  pollTimer = setInterval(pollState, POLL_MS);
  maybeRunSmokeCapture();
});

app.on('before-quit', () => {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (smokeCaptureTimer) {
    clearTimeout(smokeCaptureTimer);
    smokeCaptureTimer = null;
  }
  saveHostState();
});

app.on('window-all-closed', () => {
  app.quit();
});
