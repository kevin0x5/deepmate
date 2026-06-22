const kind = document.getElementById('kind');
const title = document.getElementById('title');
const summary = document.getElementById('summary');
const project = document.getElementById('project');
const status = document.getElementById('status');
const timeline = document.getElementById('timeline');
const feedback = document.getElementById('feedback');
const feedbackUseful = document.getElementById('feedback-useful');
const feedbackLess = document.getElementById('feedback-less');
const openCurrent = document.getElementById('open-current');
const mute = document.getElementById('mute');
const mini = document.getElementById('mini');
const hideBubble = document.getElementById('hide-bubble');
const quit = document.getElementById('quit');

function statusLabel(state) {
  const raw = String((state && state.kind) || '').replace('task.', '').replace('current_work.', '');
  const visual = String((state && state.state) || 'idle');
  return raw || visual || 'Deepmate';
}

window.deepmatePet.onHud((state) => {
  kind.textContent = statusLabel(state);
  title.textContent = String((state && state.title) || 'Current work');
  renderInlineMarkdown(summary, String((state && state.summary) || 'Deepmate is ready.'));
  project.textContent = projectName(state);
  status.textContent = statusText(state);
  renderTimeline(state);
  feedback.hidden = !shouldShowFeedback(state);
  const local = (state && state.local) || {};
  mute.textContent = local.muted ? 'Exit quiet mode' : 'Quiet 1h';
  mini.textContent = local.mini ? 'Exit mini mode' : 'Mini mode';
});

window.deepmatePet.ready('hud');

function projectName(state) {
  const workspace = String((state && state.workspace) || '').trim();
  if (!workspace) return '-';
  const parts = workspace.split(/[\\/]+/).filter(Boolean);
  return parts[parts.length - 1] || workspace;
}

function statusText(state) {
  const visual = String((state && state.state) || 'idle').trim();
  const severity = String((state && state.severity) || '').trim();
  return severity && severity !== 'info' ? `${visual} / ${severity}` : visual;
}

function renderTimeline(state) {
  timeline.replaceChildren();
  timelineItems(state).forEach((item) => {
    const row = document.createElement('li');
    const label = document.createElement('span');
    const body = document.createElement('b');
    label.textContent = item.label;
    renderInlineMarkdown(body, item.body);
    row.append(label, body);
    timeline.append(row);
  });
}

function timelineItems(state) {
  const items = [];
  const kindText = statusLabel(state);
  if (kindText) {
    items.push({ label: 'Now', body: kindText });
  }
  const refs = Array.isArray(state && state.refs) ? state.refs : [];
  const ref = refs.find((item) => String(item || '').trim());
  if (ref) {
    items.push({ label: 'Ref', body: String(ref).replace(/^path=/, '') });
  }
  if (!items.length) {
    items.push({ label: 'Now', body: 'Deepmate is ready.' });
  }
  return items.slice(0, 3);
}

function renderInlineMarkdown(root, text) {
  root.replaceChildren();
  let index = 0;
  const pattern = /(\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/g;
  for (const match of text.matchAll(pattern)) {
    if (match.index > index) {
      root.append(document.createTextNode(text.slice(index, match.index)));
    }
    if (match[2]) {
      const strong = document.createElement('strong');
      strong.textContent = match[2];
      root.append(strong);
    } else if (match[3]) {
      const code = document.createElement('code');
      code.textContent = match[3];
      root.append(code);
    } else if (match[4]) {
      const span = document.createElement('span');
      span.className = 'hud-link';
      span.textContent = match[4];
      root.append(span);
    }
    index = match.index + match[0].length;
  }
  if (index < text.length) {
    root.append(document.createTextNode(text.slice(index)));
  }
}

function shouldShowFeedback(state) {
  const kindText = String((state && state.kind) || '');
  const bubble = (state && state.bubble) || {};
  const display = (state && state.display) || {};
  const reason = String(bubble.reason || display.reason || '');
  if (!(bubble.show && bubble.text)) return false;
  return (
    kindText.startsWith('learning.')
    || kindText.startsWith('care.')
    || reason === 'learning_suggestion'
    || reason === 'proactive_care'
  );
}

function sendFeedback(value) {
  window.deepmatePet.sendFeedback(value);
  window.deepmatePet.closeHud();
}

openCurrent.addEventListener('click', () => {
  window.deepmatePet.openCurrentWork();
  window.deepmatePet.closeHud();
});

mute.addEventListener('click', () => {
  window.deepmatePet.toggleMuted();
});

mini.addEventListener('click', () => {
  window.deepmatePet.toggleMini();
  window.deepmatePet.closeHud();
});

hideBubble.addEventListener('click', () => {
  window.deepmatePet.hideBubble();
  window.deepmatePet.closeHud();
});

feedbackUseful.addEventListener('click', () => {
  sendFeedback('useful');
});

feedbackLess.addEventListener('click', () => {
  sendFeedback('less_often');
});

quit.addEventListener('click', () => {
  window.deepmatePet.quit();
});
