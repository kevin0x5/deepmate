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

const eventLabels = {
  'task.started': '任务开始', 'task.progress': '任务进度', 'task.waiting': '等待处理',
  'task.completed': '任务完成', 'task.failed': '任务受阻', 'task.achievement': '任务进展'
};
const stateLabels = {
  idle: '空闲', thinking: '思考中', working: '进行中', waiting: '等待中',
  reporting: '汇报中', celebrate: '已完成', blocked: '受阻', resting: '休息中', offline: '离线'
};

window.deepmatePet.onHud((state) => {
  kind.textContent = eventLabels[state.kind] || '当前动态';
  title.textContent = String((state && state.title) || '当前工作');
  renderInlineMarkdown(summary, String((state && state.summary) || '等待任务。'));
  project.textContent = projectName(state);
  status.textContent = stateLabels[state.state] || stateLabels.idle;
  openCurrent.disabled = !state.canOpenWork;
  hideBubble.disabled = !state.bubbleVisible;
  renderTimeline(state);
  feedback.hidden = !shouldShowFeedback(state);
  const local = (state && state.local) || {};
  mute.textContent = local.muted ? '结束静音' : '静音一小时';
  mute.setAttribute('aria-pressed', String(Boolean(local.muted)));
  mini.textContent = local.mini ? '退出贴边模式' : '贴边收起';
  mini.setAttribute('aria-pressed', String(Boolean(local.mini)));
});

document.getElementById('close-panel').addEventListener('click', () => window.deepmatePet.closeHud());
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') window.deepmatePet.closeHud();
});
window.deepmatePet.ready('hud');

function projectName(state) {
  const workspace = String((state && state.workspace) || '').trim();
  if (!workspace) return '-';
  const parts = workspace.split(/[\\/]+/).filter(Boolean);
  return parts[parts.length - 1] || workspace;
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
  const refs = Array.isArray(state.refs) ? state.refs : [];
  return refs.filter((ref) => typeof ref === 'string' && ref.trim())
    .slice(0, 3).map((ref) => ({ label: '关联', body: ref.replace(/^path=/, '') }));
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
