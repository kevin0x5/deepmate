const pet = document.getElementById('pet');
const sprite = document.getElementById('sprite');
const ctx = sprite.getContext('2d');

const BASE_FRAMES = {
  dog: [
    '....................',
    '...BBB........BBB...',
    '..BCCCBB....BBCCCB..',
    '.BCCCCCCBBBBCCCCCCB.',
    '.BCCCCCCCCCCCCCCCCB.',
    '.BCCWWCCCCCCCCWWCCB.',
    '.BCCCCCCCPCCCCCCCCB.',
    '..BCCCCCCCCCCCCCCB..',
    '...BCCCCCCCCCCCCB...',
    '..BBCCCCCGGCCCCBB...',
    '.B..BCCCCCCCCB..B...',
    '....BCCCCCCCCB......',
    '....BB......BB......',
    '...B..B....B..B.....',
    '....................',
    '....................'
  ],
  cat: [
    '....................',
    '..BB..........BB....',
    '.BSSB........BSSB...',
    '.BCCCBBBBBBBBCCCB...',
    '.BCCCCCCCCCCCCCCB...',
    '.BCCWWCCCCCCWWCCB...',
    '.BCCCCCCPCCCCCCCB...',
    'B..BCCCCCCCCCCB..B..',
    '...BCCCCCCCCCCB.....',
    '..BBCCCCGGCCCCBB....',
    '....BCCCCCCCCB......',
    '....BCCCCCCCCB......',
    '....BB......BB......',
    '...B..B....B..B.....',
    '....................',
    '....................'
  ],
  squirrel: [
    '....................',
    '...........BBBB.....',
    '..........BCCCCB....',
    '.....BBBBBCCCCCB....',
    '....BCCCCCCCCCCB....',
    '...BCCCWWCCCCWWB....',
    '...BCCCCCCPCCCB.....',
    '....BCCCCCCCCB......',
    '...BBCCCCGGCCB......',
    '..B..BCCCCCCCB......',
    '.....BCCCCCCCB......',
    '.....BB....BBB......',
    '....B..B..B..B......',
    '....................',
    '....................',
    '....................'
  ],
  penguin: [
    '....................',
    '......BBBBBBBB......',
    '.....BSSSSSSSSB.....',
    '....BSSCCCCCCSSB....',
    '....BSCWWCCWWCSB....',
    '....BSCCCCCCCCSB....',
    '....BSCCCPCCCCSB....',
    '.....BSCCCCCCSB.....',
    '......BCCCCCCB......',
    '.....BBCCCCCCBB.....',
    '....B..BCCCCB..B....',
    '.......BG..GB.......',
    '......G......G......',
    '....................',
    '....................',
    '....................'
  ]
};

const PALETTES = {
  dog: {
    B: '#3d2b1f',
    C: '#c98245',
    W: '#fff4df',
    P: '#f58ca8',
    G: '#ffd166',
    Y: '#f4c430',
    R: '#e85d75',
    L: '#7cc7ff',
    S: '#fff4df'
  },
  cat: {
    B: '#3f4657',
    C: '#8fa3bf',
    W: '#f5f7fb',
    P: '#b7a6d9',
    G: '#f3f0ff',
    Y: '#d8c76f',
    R: '#d7728a',
    L: '#91b8ff',
    S: '#d7deea'
  },
  squirrel: {
    B: '#4a2d1f',
    C: '#9a5b2f',
    W: '#ffe6bd',
    P: '#dd8f45',
    G: '#8ab17d',
    Y: '#f2c14e',
    R: '#dd6b6b',
    L: '#77c7c2',
    S: '#6d8f57'
  },
  penguin: {
    B: '#1d2633',
    C: '#f7fbff',
    W: '#98d8ef',
    P: '#ffb4a2',
    G: '#f4c430',
    Y: '#f4c430',
    R: '#e76f8a',
    L: '#80d8ff',
    S: '#bdefff'
  }
};

const FRAME_MS = 420;
const PIXEL_SIZE = 7;
const DRAW_OFFSET_X = 10;
const DRAW_OFFSET_Y = 8;
const DRAG_THRESHOLD_PX = 2;
const STATES = ['idle', 'thinking', 'working', 'waiting', 'reporting', 'celebrate', 'blocked', 'resting', 'offline'];
const SPECIES = ['dog', 'cat', 'squirrel', 'penguin'];
const REACTIONS = ['poke'];
const FRAME_CACHE = buildFrameCache();

let lastPoint = null;
let dragOrigin = null;
let dragging = false;
let lastClickAt = 0;
let mouseRegionActive = false;
let currentVisual = 'idle';
let currentSpecies = 'dog';
let frameTick = 0;

function cleanState(value) {
  const state = String(value || 'idle').toLowerCase();
  return STATES.includes(state) ? state : 'idle';
}

function cleanSpecies(profile) {
  const species = String((profile && profile.species) || 'dog').toLowerCase();
  return SPECIES.includes(species) ? species : 'dog';
}

function applyState(state) {
  currentVisual = cleanState(state.state);
  currentSpecies = cleanSpecies(state.profile || {});
  pet.className = `pet state-${currentVisual} species-${currentSpecies}`;
  renderSprite();
  const reaction = cleanReaction(state.reaction);
  if (reaction) {
    pet.classList.add(`reaction-${reaction}`);
    window.setTimeout(() => pet.classList.remove(`reaction-${reaction}`), 700);
  }
}

function cleanReaction(value) {
  const reaction = String(value || '').toLowerCase();
  return REACTIONS.includes(reaction) ? reaction : '';
}

function renderSprite() {
  const frames = FRAME_CACHE[currentSpecies] || FRAME_CACHE.dog;
  const variants = frames[currentVisual] || frames.idle;
  const frame = variants[frameTick % variants.length];
  const palette = PALETTES[currentSpecies] || PALETTES.dog;
  ctx.clearRect(0, 0, sprite.width, sprite.height);
  ctx.imageSmoothingEnabled = false;
  frame.forEach((row, y) => {
    Array.from(row).forEach((code, x) => {
      const color = palette[code];
      if (!color) return;
      ctx.fillStyle = color;
      ctx.fillRect(
        DRAW_OFFSET_X + x * PIXEL_SIZE,
        DRAW_OFFSET_Y + y * PIXEL_SIZE,
        PIXEL_SIZE,
        PIXEL_SIZE
      );
    });
  });
}

function buildFrameCache() {
  return Object.fromEntries(
    Object.entries(BASE_FRAMES).map(([species, base]) => [species, stateFrames(base)])
  );
}

function stateFrames(base) {
  return {
    idle: [base, shift(base, 0, 1)],
    thinking: [base, ears(base, 'G'), shift(ears(base, 'G'), 1, 0)],
    working: [base, shift(base, 1, 0), shift(base, -1, 0)],
    waiting: [ears(base, 'Y'), shift(ears(base, 'Y'), 0, -1)],
    reporting: [ears(base, 'L'), shift(ears(base, 'L'), 0, -1)],
    celebrate: [sparkle(base, 'G'), shift(sparkle(base, 'G'), 0, -1)],
    blocked: [ears(base, 'R'), shift(ears(base, 'R'), 0, 1)],
    resting: [shift(base, 0, 1), shift(base, 0, 2)],
    offline: [shift(base, 0, 2)]
  };
}

function shift(frame, dx, dy) {
  let rows = frame.slice();
  const width = rows[0].length;
  if (dy > 0) {
    rows = Array(dy).fill('.'.repeat(width)).concat(rows.slice(0, -dy));
  } else if (dy < 0) {
    rows = rows.slice(-dy).concat(Array(-dy).fill('.'.repeat(width)));
  }
  return rows.map((row) => {
    if (dx > 0) return '.'.repeat(dx) + row.slice(0, -dx);
    if (dx < 0) return row.slice(-dx) + '.'.repeat(-dx);
    return row;
  });
}

function ears(frame, code) {
  const rows = frame.slice();
  if (rows.length < 2) return frame;
  rows[1] = replaceAt(rows[1], 2, code);
  rows[1] = replaceAt(rows[1], Math.max(0, rows[1].length - 3), code);
  return rows;
}

function sparkle(frame, code) {
  const rows = ears(frame, code);
  if (rows.length < 4) return rows;
  rows[0] = replaceAt(rows[0], 6, code);
  rows[2] = replaceAt(rows[2], Math.max(0, rows[2].length - 7), code);
  return rows;
}

function replaceAt(row, index, code) {
  if (index < 0 || index >= row.length) return row;
  return row.slice(0, index) + code + row.slice(index + 1);
}

window.setInterval(() => {
  frameTick += 1;
  renderSprite();
}, FRAME_MS);

window.deepmatePet.onState(applyState);
window.deepmatePet.ready('pet');
renderSprite();

function isInteractivePoint(event) {
  if (dragging) return true;
  const rect = pet.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  if (x >= 24 && x <= 166 && y >= 20 && y <= 160) return true;
  if (x >= 118 && x <= 178 && y >= 14 && y <= 62) return true;
  return false;
}

function setMouseRegion(active) {
  if (mouseRegionActive === active) return;
  mouseRegionActive = active;
  window.deepmatePet.setMouseRegion(active);
}

window.addEventListener('mousemove', (event) => {
  setMouseRegion(isInteractivePoint(event));
});

pet.addEventListener('pointerleave', () => {
  if (!dragging) {
    setMouseRegion(false);
  }
});

pet.addEventListener('pointerdown', (event) => {
  if (event.button === 2) {
    window.deepmatePet.showMenu();
    return;
  }
  setMouseRegion(true);
  lastPoint = { x: event.screenX, y: event.screenY };
  dragOrigin = lastPoint;
  dragging = false;
  pet.setPointerCapture(event.pointerId);
});

pet.addEventListener('pointermove', (event) => {
  if (!lastPoint) return;
  const dx = event.screenX - lastPoint.x;
  const dy = event.screenY - lastPoint.y;
  const totalDx = dragOrigin ? event.screenX - dragOrigin.x : dx;
  const totalDy = dragOrigin ? event.screenY - dragOrigin.y : dy;
  if (Math.hypot(totalDx, totalDy) > DRAG_THRESHOLD_PX) dragging = true;
  if (dragging) {
    window.deepmatePet.dragBy({ dx, dy });
    lastPoint = { x: event.screenX, y: event.screenY };
  }
});

pet.addEventListener('pointerup', (event) => {
  if (lastPoint) {
    pet.releasePointerCapture(event.pointerId);
  }
  lastPoint = null;
  dragOrigin = null;
  if (dragging) {
    dragging = false;
    window.deepmatePet.finishDrag();
    return;
  }
  const now = Date.now();
  if (now - lastClickAt < 360) {
    window.deepmatePet.setReaction('poke');
    window.deepmatePet.openCurrentWork();
    lastClickAt = 0;
    return;
  }
  lastClickAt = now;
  window.deepmatePet.toggleHud();
});

pet.addEventListener('contextmenu', (event) => {
  event.preventDefault();
  window.deepmatePet.showMenu();
});

window.deepmatePet.setMouseRegion(false);
