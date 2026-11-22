const pet = document.getElementById('pet');
const sprite = document.getElementById('sprite');
const ctx = sprite.getContext('2d');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const STATES = ['idle', 'thinking', 'working', 'waiting', 'reporting', 'celebrate', 'blocked', 'resting', 'offline'];
const SPECIES = ['dog', 'cat', 'squirrel', 'penguin'];
const DRAG_THRESHOLD_PX = 2;

const frames = new Map();
let currentVisual = 'idle';
let currentSpecies = 'dog';
let blinkTimer = null;
let reactionTimer = null;
let lastPoint = null;
let dragOrigin = null;
let dragging = false;
let clickTimer = null;
let mouseRegionActive = false;

function paintRows(context, rows, palette, left = 0, top = 0) {
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x += 1) {
      const color = palette[row[x]];
      if (color) {
        context.fillStyle = color;
        context.fillRect(left + x, top + y, 1, 1);
      }
    }
  });
}

function buildFrames(portrait) {
  const open = document.createElement('canvas');
  open.width = portrait.rows[0].length;
  open.height = portrait.rows.length;
  paintRows(open.getContext('2d'), portrait.rows, portrait.palette);
  const closed = open.cloneNode();
  const eyes = closed.getContext('2d');
  eyes.drawImage(open, 0, 0);
  for (const patch of portrait.closedEyes) {
    paintRows(eyes, patch.rows, portrait.palette, patch.x, patch.y);
  }
  return { open, closed };
}

function drawFace(closed) {
  const face = frames.get(currentSpecies);
  if (!face) return;
  ctx.clearRect(0, 0, sprite.width, sprite.height);
  ctx.drawImage(closed ? face.closed : face.open, 0, 0);
}

function scheduleBlink() {
  window.clearTimeout(blinkTimer);
  const resting = currentVisual === 'resting' || currentVisual === 'offline';
  drawFace(resting);
  if (resting || document.hidden || reducedMotion.matches || !frames.size) return;
  blinkTimer = window.setTimeout(() => {
    drawFace(true);
    blinkTimer = window.setTimeout(scheduleBlink, 150);
  }, 4400);
}

function applyState(state) {
  const requestedVisual = String(state.state || '').toLowerCase();
  const requestedSpecies = String(state.profile?.species || '').toLowerCase();
  const visual = STATES.includes(requestedVisual) ? requestedVisual : 'idle';
  const species = SPECIES.includes(requestedSpecies) ? requestedSpecies : 'dog';
  if (currentVisual !== visual || currentSpecies !== species) {
    currentVisual = visual;
    currentSpecies = species;
    pet.className = `pet state-${visual} species-${species}`;
    scheduleBlink();
  }
  if (String(state.reaction || '').toLowerCase() === 'poke') {
    window.clearTimeout(reactionTimer);
    pet.classList.add('reaction-poke');
    reactionTimer = window.setTimeout(() => pet.classList.remove('reaction-poke'), 700);
  }
}

window.deepmatePet.onPortraits((portraits) => {
  for (const species of SPECIES) {
    frames.set(species, buildFrames(portraits[species]));
  }
  scheduleBlink();
});
document.addEventListener('visibilitychange', scheduleBlink);
reducedMotion.addEventListener('change', scheduleBlink);
window.deepmatePet.onState(applyState);
window.deepmatePet.ready('pet');

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
  if (event.button !== 0) return;
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
  if (event.button !== 0 || !lastPoint) return;
  pet.releasePointerCapture(event.pointerId);
  lastPoint = null;
  dragOrigin = null;
  if (dragging) {
    dragging = false;
    window.deepmatePet.finishDrag();
    return;
  }
  if (clickTimer !== null) {
    window.clearTimeout(clickTimer);
    clickTimer = null;
    window.deepmatePet.setReaction('poke');
    window.deepmatePet.openCurrentWork();
    return;
  }
  clickTimer = window.setTimeout(() => {
    clickTimer = null;
    window.deepmatePet.toggleHud();
  }, 320);
});

pet.addEventListener('pointercancel', () => {
  lastPoint = null;
  dragOrigin = null;
  if (dragging) window.deepmatePet.finishDrag();
  dragging = false;
  setMouseRegion(false);
});

pet.addEventListener('contextmenu', (event) => {
  event.preventDefault();
  window.deepmatePet.showMenu();
});

window.deepmatePet.setMouseRegion(false);
