const bubble = document.getElementById('bubble');

window.deepmatePet.onBubble((payload) => {
  const text = String((payload && payload.text) || '');
  renderInlineMarkdown(bubble, text);
  bubble.hidden = !text.trim();
});

window.deepmatePet.ready('bubble');

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
      span.className = 'bubble-link';
      span.textContent = match[4];
      root.append(span);
    }
    index = match.index + match[0].length;
  }
  if (index < text.length) {
    root.append(document.createTextNode(text.slice(index)));
  }
}
