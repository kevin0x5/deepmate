const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('deepmatePet', {
  onState(callback) {
    ipcRenderer.on('pet-state', (_event, state) => callback(state));
  },
  onBubble(callback) {
    ipcRenderer.on('pet-bubble', (_event, payload) => callback(payload));
  },
  onHud(callback) {
    ipcRenderer.on('pet-hud', (_event, payload) => callback(payload));
  },
  ready(surface) {
    ipcRenderer.send('pet-renderer-ready', surface);
  },
  dragBy(delta) {
    ipcRenderer.send('pet-drag-by', delta);
  },
  finishDrag() {
    ipcRenderer.send('pet-finish-drag');
  },
  toggleHud() {
    ipcRenderer.send('pet-toggle-hud');
  },
  closeHud() {
    ipcRenderer.send('pet-close-hud');
  },
  hideBubble() {
    ipcRenderer.send('pet-hide-bubble');
  },
  openCurrentWork() {
    ipcRenderer.send('pet-action-open-current-work');
  },
  toggleMuted() {
    ipcRenderer.send('pet-toggle-muted');
  },
  toggleMini() {
    ipcRenderer.send('pet-toggle-mini');
  },
  sendFeedback(value) {
    ipcRenderer.send('pet-feedback', value);
  },
  quit() {
    ipcRenderer.send('pet-quit');
  },
  showMenu() {
    ipcRenderer.send('pet-show-menu');
  },
  setMouseRegion(active) {
    ipcRenderer.send('pet-mouse-region', Boolean(active));
  },
  setReaction(reaction) {
    ipcRenderer.send('pet-reaction', reaction);
  }
});
