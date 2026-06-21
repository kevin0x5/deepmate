const bubble = document.getElementById('bubble');

window.deepmatePet.onBubble((payload) => {
  const text = String((payload && payload.text) || '');
  bubble.textContent = text;
  bubble.hidden = !text.trim();
});

window.deepmatePet.ready('bubble');
