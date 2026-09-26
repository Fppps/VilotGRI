/**
 * VilotNI Unified Controller
 * Features:
 * - Adaptive Time Thought Tag: Renders sub-second runs cleanly (e.g. 4ms, 38ms) and full seconds as X.XXs.
 * - Zero Artificial Stalls: Real-time execution with immediate DOM text commit.
 * - Multi-Engine Routing: VilotNI 1, VilotNI 1.5, and the production VilotNI 2 6M RSL engine.
 * - Real-Time Dock & Carousel State: Dynamically tracks RL (Low), RSL (Medium), and Deep RSL (High).
 * - Anytime Inference Integration: Directly supports fast-exit rollouts from RLEngine.
 * - Exact-Message Reward Routing: Feedback trains the prompt/response that was actually rated.
 * - WebGPU/CPU Reward-Trace Parity: Both paths isolate policy traces and exact response head rows.
 */

window.activeModel = 'vilotni_1_5'; // vilotni_1 | vilotni_1_5 | vilotni_2
window.vocabInstance = null;
window.backendInstance = null;
window.networkInstance = null;
window.agentInstance = null;

// Single readiness gate for the ENTIRE VilotNI 1.5 stack.
window.vilotni15CoreState = "initializing";
window.vilotni15CoreError = null;

// V22 memory readiness stays internal so the UI remains clean.
window.vilotni15MemoryReady = false;
window.vilotni15MemoryError = null;

let resolveVilotni15CoreReady;
window.vilotni15CoreReady = new Promise(resolve => {
  resolveVilotni15CoreReady = resolve;
});

function settleVilotni15Core(ok, error = null) {
  if (window.vilotni15CoreState !== "initializing") return;

  window.vilotni15CoreState = ok ? "ready" : "error";
  window.vilotni15CoreError = error || null;

  resolveVilotni15CoreReady({
    ok,
    error: error || null
  });

  const input = document.getElementById('user-input');
  if (input && typeof window.updateSubmitBtnState === 'function') {
    window.updateSubmitBtnState(input.value);
  }
}

// VilotNI 2 runtime readiness is independent from the 1.5 stack. A V1.5
// vocabulary/WebGPU failure must never make the V2 worker inaccessible.
window.vilotni2Server = null;
window.vilotni2CoreState = "initializing";
window.vilotni2CoreError = null;
window.lastVilotni2Result = null;

let resolveVilotni2CoreReady;
window.vilotni2CoreReady = new Promise(resolve => {
  resolveVilotni2CoreReady = resolve;
});

function settleVilotni2Core(ok, error = null, status = null) {
  if (window.vilotni2CoreState !== "initializing") return;
  window.vilotni2CoreState = ok ? "ready" : "error";
  window.vilotni2CoreError = error || null;
  resolveVilotni2CoreReady({ ok, error: error || null, status });
  const input = document.getElementById('user-input');
  if (input && typeof window.updateSubmitBtnState === 'function') {
    window.updateSubmitBtnState(input.value);
  }
}

async function initializeVilotni2Core() {
  try {
    if (typeof VilotNIServer !== 'function') {
      throw new Error('VilotNI 2 Server.js did not load.');
    }
    window.vilotni2Server = new VilotNIServer({
      stateDim: 6000000,
      activeTickMs: 40,
      warmTickMs: 180,
      idleTickMs: 850,
      warmAfterMs: 1400,
      idleAfterMs: 12000,
      activeHomeostasisMs: 240,
      warmHomeostasisMs: 520,
      idleHomeostasisMs: 900,
      minResponseConfidence: 80,
      latencyTargetMs: 40
    });
    const status = await window.vilotni2Server.start();
    settleVilotni2Core(true, null, status);
    return status;
  } catch (error) {
    console.error('VilotNI 2 engine failed to initialize:', error);
    settleVilotni2Core(false, error, null);
    throw error;
  }
}

window.getVilotni2Status = async function(sync = true) {
  if (!window.vilotni2Server) return null;
  return window.vilotni2Server.getStatus(Boolean(sync));
};

window.getVilotni2RSLStatus = async function() {
  if (!window.vilotni2Server) return null;
  return window.vilotni2Server.getRSLStatus();
};

window.getVilotni2DecoderStatus = async function() {
  if (!window.vilotni2Server) return null;
  return window.vilotni2Server.getDecoderStatus();
};

window.inspectVilotni2Word = async function(word) {
  if (!window.vilotni2Server) return null;
  return window.vilotni2Server.inspectWord(word);
};

window.inspectVilotni2Knowledge = async function(query, limit = 8) {
  if (!window.vilotni2Server) return [];
  return window.vilotni2Server.inspectKnowledge(query, limit);
};

window.getVilotni2Trace = async function(limit = 64) {
  if (!window.vilotni2Server) return [];
  return window.vilotni2Server.getTrace(limit);
};

window.resetVilotni2State = async function() {
  if (!window.vilotni2Server) return null;
  window.lastVilotni2Result = null;
  return window.vilotni2Server.resetState();
};

window.benchmarkVilotni2 = async function(iterations = 2000) {
  if (!window.vilotni2Server) return null;
  return {
    local: await window.vilotni2Server.benchmarkLocal(iterations),
    roundTrip: await window.vilotni2Server.benchmarkRoundTrip(50)
  };
};

// Optional confidence instrumentation for engine diagnostics.
window.showConfidenceDiagnostics = false;
window.lastVilotni15Diagnostics = null;

window.getVilotniAutoregressiveStatus = function() {
  const agent = window.agentInstance;
  const cfg = agent?.reasoningConfig || {};
  const d = window.lastVilotni15Diagnostics || {};
  return {
    mode: agent?.mode || null,
    deep: Boolean(agent?.isDeepLearning),
    enabled: Boolean(cfg.autoregressiveLatentEnabled),
    trueNeuralSelfConditioning: Boolean(cfg.rslTrueNeuralSelfConditioningEnabled),
    updates: d.autoregressiveLatentUpdates || 0,
    stability: d.autoregressiveLatentStability ?? null,
    livePool: d.peakLiveCandidatePoolSize || d.candidatePoolSize || 0,
    neuralForwards: d.neuralForwards ?? null,
    frozenLogitFloor: cfg.autoregressiveFrozenLogitFloor ?? null,
    frozenLogitDecay: cfg.autoregressiveFrozenLogitDecay ?? null,
    liveProjectionWeight: cfg.autoregressiveLiveProjectionWeight ?? null,
    liveDeltaWeight: cfg.autoregressiveLiveDeltaWeight ?? null,
    liveNeuralScoreWeight: cfg.rslLiveNeuralScoreWeight ?? null,
    liveNeuralCandidateLimit: cfg.rslLiveNeuralCandidateLimit ?? null,
    decoderVersion: d.rslVersion ?? null,
    persistenceKey: d.rslPersistenceKey ?? null,
    legacyImportEnabled: d.rslLegacyImportEnabled ?? null,
    neuralPositiveWeight: cfg.rslNeuralPositiveWeight ?? null,
    transitionScale: cfg.rslLegacyTransitionScale ?? null,
    roleScale: cfg.rslLegacyRoleScale ?? null,
    lookaheadScale: cfg.rslLegacyLookaheadScale ?? null,
    minimumContinuationLength: d.minimumContinuationLength ?? null,
    continuationSoftMinFraction: cfg.rslContinuationSoftMinFraction ?? null,
    lexicalRootFamilyEnabled: Boolean(cfg.lexicalRootFamilyEnabled),
    postPredicateSoftLimit: cfg.rslPostPredicateSoftLimit ?? null
  };
};

window.getVilotniDecoderTrace = function() {
  const d = window.lastVilotni15Diagnostics || {};
  return Array.isArray(d.decoderTrace)
    ? d.decoderTrace
    : [];
};

window.printVilotniDecoderTrace = function() {
  const trace = window.getVilotniDecoderTrace();
  const rows = trace.map(step => {
    const s = step.selected || {};
    return {
      step: step.index,
      word: step.word,
      total: Number.isFinite(s.total) ? Number(s.total.toFixed(3)) : null,
      neural: Number.isFinite(s.neural) ? Number(s.neural.toFixed(3)) : null,
      neuralEvidence: Number.isFinite(s.neuralEvidence) ? Number(s.neuralEvidence.toFixed(3)) : null,
      legacyStructural: Number.isFinite(s.legacyStructural) ? Number(s.legacyStructural.toFixed(3)) : null,
      semantic: Number.isFinite(s.routedSemantic) ? Number(s.routedSemantic.toFixed(3)) : null,
      latent: Number.isFinite(s.latent) ? Number(s.latent.toFixed(3)) : null,
      planner: Number.isFinite(s.planner) ? Number(s.planner.toFixed(3)) : null,
      other: Number.isFinite(s.other) ? Number(s.other.toFixed(3)) : null,
      margin: Number.isFinite(step.margin) ? Number(step.margin.toFixed(3)) : null
    };
  });
  console.table(rows);
  return trace;
};

window.resetVilotniV28DecoderLearning = function() {
  if (typeof localStorage !== 'undefined') {
    localStorage.removeItem('vilotni15_rsl_continuation_v28');
  }
  location.reload();
};


window.inspectVilotniLexicalRoot = function(wordOrToken) {
  const agent = window.agentInstance;
  if (!agent) return null;
  let token = Number.isInteger(wordOrToken) ? wordOrToken : null;
  if (token === null && typeof wordOrToken === 'string') {
    token = agent.getVocabTokenId?.(wordOrToken);
  }
  if (!Number.isInteger(token)) {
    return { token: null, word: String(wordOrToken ?? ''), roots: [] };
  }
  return {
    token,
    word: agent.tokenWord?.(token) || String(wordOrToken ?? ''),
    roots: agent.lexicalRootCandidates?.(token) || []
  };
};

window.printVilotniDecoderAlternatives = function(stepIndex = 0) {
  const trace = window.getVilotniDecoderTrace();
  const step = trace[stepIndex];
  if (!step) return [];
  const rows = (step.topAlternatives || []).map(item => ({
    word: item.word,
    total: Number.isFinite(item.total) ? Number(item.total.toFixed(3)) : null,
    neural: Number.isFinite(item.neural) ? Number(item.neural.toFixed(3)) : null,
    neuralEvidence: Number.isFinite(item.neuralEvidence) ? Number(item.neuralEvidence.toFixed(3)) : null,
    legacyStructural: Number.isFinite(item.legacyStructural) ? Number(item.legacyStructural.toFixed(3)) : null,
    semantic: Number.isFinite(item.routedSemantic) ? Number(item.routedSemantic.toFixed(3)) : null,
    latent: Number.isFinite(item.latent) ? Number(item.latent.toFixed(3)) : null,
    planner: Number.isFinite(item.planner) ? Number(item.planner.toFixed(3)) : null,
    other: Number.isFinite(item.other) ? Number(item.other.toFixed(3)) : null
  }));
  console.table(rows);
  return step.topAlternatives || [];
};

let isProcessing = false;
let tempAvatarUrl = null;
let lastPromptTokens = [];

// Feedback must train the message the user actually rated, not simply the
// latest generation. Each visible VilotNI 1.5 message gets its own immutable
// prompt/response/trajectory snapshot.
const messageTrainingState = new Map();

function formatElapsed(ms) {
  if (ms < 1) return "<1ms";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function assertVilotMemoryModules() {
  const missing = [];

  if (typeof VilotMemoryStore !== 'function') missing.push('memory_store.js');
  if (typeof VilotMemoryRetrieval !== 'function') missing.push('memory_retrieval.js');
  if (typeof VilotTypePhraseParser !== 'function') missing.push('phrase_parser.js');

  if (missing.length) {
    throw new Error(
      `VilotNI runtime modules are missing: ${missing.join(', ')}. ` +
      'Load memory_store.js, memory_retrieval.js, and phrase_parser.js before rl_agent.js.'
    );
  }

  return true;
}

async function createAgent(network) {
  if (typeof RLEngine !== 'function') {
    throw new Error(
      'RLEngine is not defined. VilotNI 1.5 cannot start its RSL/RL stack.'
    );
  }

  assertVilotMemoryModules();

  const agent = new RLEngine(network);

  if (
    agent.memoryReady &&
    typeof agent.memoryReady.then === 'function'
  ) {
    const memoryReady = await agent.memoryReady;

    if (
      agent.reasoningConfig?.memoryEnabled !== false &&
      memoryReady !== true
    ) {
      throw new Error(
        'VilotNI memory system did not become ready.'
      );
    }
  }

  if (
    agent.reasoningConfig?.memoryEnabled !== false &&
    (!agent.memoryStore || !agent.memoryRetrieval)
  ) {
    throw new Error(
      'VilotNI V22 memory is enabled, but its store/retrieval system was not initialized.'
    );
  }

  window.vilotni15MemoryReady = Boolean(
    agent.memoryStore &&
    agent.memoryRetrieval
  );

  window.vilotni15MemoryError = null;

  return agent;
}

function resetVilotniConversationState() {
  lastPromptTokens = [];

  const agent = window.agentInstance;

  if (!agent) {
    return;
  }

  // "New Conversation" clears only short-term conversational state.
  // Episodic, semantic, and procedural long-term memory remain intact.
  if (agent.memoryStore?.working) {
    agent.memoryStore.working.length = 0;
  }

  agent.previousUserAnchorTokens = [];
  agent.previousUserLastContentToken = null;
  agent.pendingUserStateSource = null;
  agent.pendingNextUserStatePrediction = null;
  agent.pendingPredictedTurnTokens = [];

  if (agent.recentLexicalUsage?.clear) {
    agent.recentLexicalUsage.clear();
  }

  agent.lastDiscardedTrajectory = [];
  agent.trajectory = [];
}

let userProfile = {
  name: 'User',
  bgColor: 'var(--brand-primary)',
  textColor: '#ffffff',
  avatarUrl: null
};

const CHAT_STORAGE_KEY = 'vilotni_saved_chats_v1';
const APP_SETTINGS_KEY = 'vilotni_app_settings_v1';
let savedChats = [];
let activeChatId = null;
let suppressChatPersistence = false;
let pendingDeleteChatId = null;

const DEFAULT_APP_SETTINGS = Object.freeze({
  chats: {
    retainContextAfterDelete: false,
    skipDeleteWarning: false
  },
  defaultModel: 'vilotni_1_5',
  appearance: {
    theme: 'dark',
    useModelAccent: true,
    accentPrimary: '#6366f1',
    accentSecondary: '#8b5cf6',
    palettes: {
      dark: {
        bgBase: '#0f0f10',
        bgSurface: '#171719',
        bgSidebar: '#141416',
        bgElevated: '#202023',
        textPrimary: '#f5f5f6'
      },
      light: {
        bgBase: '#ffffff',
        bgSurface: '#f7f7f8',
        bgSidebar: '#f5f5f6',
        bgElevated: '#ececef',
        textPrimary: '#18181b'
      }
    }
  }
});

function cloneDefaultSettings() {
  return JSON.parse(JSON.stringify(DEFAULT_APP_SETTINGS));
}

let appSettings = cloneDefaultSettings();

function mergeAppSettings(raw) {
  const merged = cloneDefaultSettings();
  if (!raw || typeof raw !== 'object') return merged;

  if (raw.chats && typeof raw.chats === 'object') {
    if (typeof raw.chats.retainContextAfterDelete === 'boolean') merged.chats.retainContextAfterDelete = raw.chats.retainContextAfterDelete;
    if (typeof raw.chats.skipDeleteWarning === 'boolean') merged.chats.skipDeleteWarning = raw.chats.skipDeleteWarning;
  }
  if (['vilotni_1', 'vilotni_1_5', 'vilotni_2'].includes(raw.defaultModel)) merged.defaultModel = raw.defaultModel;

  if (raw.appearance && typeof raw.appearance === 'object') {
    if (raw.appearance.theme === 'light' || raw.appearance.theme === 'dark') merged.appearance.theme = raw.appearance.theme;
    if (typeof raw.appearance.useModelAccent === 'boolean') merged.appearance.useModelAccent = raw.appearance.useModelAccent;
    if (/^#[0-9a-f]{6}$/i.test(raw.appearance.accentPrimary || '')) merged.appearance.accentPrimary = raw.appearance.accentPrimary;
    if (/^#[0-9a-f]{6}$/i.test(raw.appearance.accentSecondary || '')) merged.appearance.accentSecondary = raw.appearance.accentSecondary;

    for (const mode of ['dark', 'light']) {
      const src = raw.appearance.palettes?.[mode];
      if (!src || typeof src !== 'object') continue;
      for (const key of ['bgBase', 'bgSurface', 'bgSidebar', 'bgElevated', 'textPrimary']) {
        if (/^#[0-9a-f]{6}$/i.test(src[key] || '')) merged.appearance.palettes[mode][key] = src[key];
      }
    }
  }
  return merged;
}

function loadAppSettings() {
  try {
    const raw = JSON.parse(localStorage.getItem(APP_SETTINGS_KEY) || 'null');
    appSettings = mergeAppSettings(raw);

    // Migrate the older standalone theme preference once.
    if (!raw) {
      const legacyTheme = localStorage.getItem('vilotni_theme');
      if (legacyTheme === 'light' || legacyTheme === 'dark') appSettings.appearance.theme = legacyTheme;
    }
  } catch (_) {
    appSettings = cloneDefaultSettings();
  }
  window.vilotniAppSettings = appSettings;
  return appSettings;
}

function persistAppSettings() {
  try {
    localStorage.setItem(APP_SETTINGS_KEY, JSON.stringify(appSettings));
    localStorage.setItem('vilotni_theme', appSettings.appearance.theme);
  } catch (error) {
    console.warn('Could not save app settings:', error);
  }
  window.vilotniAppSettings = appSettings;
}

function applyAppAppearanceSettings() {
  const appearance = appSettings.appearance;
  const mode = appearance.theme === 'light' ? 'light' : 'dark';
  const palette = appearance.palettes[mode];
  document.body.classList.toggle('light-mode', mode === 'light');

  const vars = {
    '--bg-base': palette.bgBase,
    '--bg-surface': palette.bgSurface,
    '--bg-sidebar': palette.bgSidebar,
    '--bg-elevated': palette.bgElevated,
    '--bg-glass': palette.bgBase,
    '--text-primary': palette.textPrimary
  };
  for (const [name, value] of Object.entries(vars)) document.body.style.setProperty(name, value);

  if (!appearance.useModelAccent) {
    document.documentElement.style.setProperty('--brand-primary', appearance.accentPrimary);
    document.documentElement.style.setProperty('--brand-secondary', appearance.accentSecondary);
    document.documentElement.style.setProperty('--vilotni-gradient', `linear-gradient(135deg, ${appearance.accentPrimary}, ${appearance.accentSecondary})`);
  }
}

window.applyAppAppearanceSettings = applyAppAppearanceSettings;

function makeChatId() {
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function cleanChatTitle(text) {
  const compact = String(text || '').replace(/\s+/g, ' ').trim();
  if (!compact) return 'New conversation';
  return compact.length > 44 ? `${compact.slice(0, 43).trimEnd()}…` : compact;
}

function loadSavedChatsFromStorage() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CHAT_STORAGE_KEY) || '[]');
    savedChats = Array.isArray(parsed)
      ? parsed.filter(chat => chat && typeof chat.id === 'string' && Array.isArray(chat.messages))
      : [];
  } catch (_) {
    savedChats = [];
  }
}

function persistSavedChats() {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(savedChats));
  } catch (error) {
    console.warn('Could not save chat history:', error);
  }
}

function getActiveSavedChat() {
  return activeChatId ? savedChats.find(chat => chat.id === activeChatId) || null : null;
}

function ensureActiveSavedChat(firstUserText = '') {
  let chat = getActiveSavedChat();
  if (chat) return chat;

  chat = {
    id: makeChatId(),
    title: cleanChatTitle(firstUserText),
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: []
  };
  savedChats.unshift(chat);
  activeChatId = chat.id;
  persistSavedChats();
  renderSavedChatList();
  return chat;
}

function recordChatMessage(message) {
  if (suppressChatPersistence || !message) return;
  const chat = ensureActiveSavedChat(message.role === 'user' ? message.text : '');
  chat.messages.push({
    ...message,
    at: Date.now()
  });
  if (message.role === 'user' && chat.messages.filter(item => item.role === 'user').length === 1) {
    chat.title = cleanChatTitle(message.text);
  }
  chat.updatedAt = Date.now();
  savedChats.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  persistSavedChats();
  renderSavedChatList();
}

function renderSavedChatList() {
  const list = document.getElementById('chat-list');
  const count = document.getElementById('chat-count');
  if (!list) return;

  list.innerHTML = '';
  if (count) count.textContent = String(savedChats.length);

  if (!activeChatId) {
    const draft = document.createElement('button');
    draft.type = 'button';
    draft.className = 'chat-session-item is-active';
    draft.innerHTML = '<span class="chat-session-dot"></span><span class="chat-session-title">New conversation</span>';
    draft.addEventListener('click', () => window.startNewChat());
    list.appendChild(draft);
  }

  for (const chat of savedChats) {
    const row = document.createElement('div');
    row.className = `chat-session-row${chat.id === activeChatId ? ' is-active' : ''}`;
    row.dataset.chatId = chat.id;

    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'chat-session-item';
    item.title = chat.title || 'Conversation';

    const dot = document.createElement('span');
    dot.className = 'chat-session-dot';
    const title = document.createElement('span');
    title.className = 'chat-session-title';
    title.textContent = chat.title || 'Conversation';
    item.append(dot, title);
    item.addEventListener('click', () => window.openSavedChat(chat.id));

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.className = 'chat-delete-btn';
    deleteBtn.title = `Delete ${chat.title || 'conversation'}`;
    deleteBtn.setAttribute('aria-label', deleteBtn.title);
    deleteBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M3 6h18M8 6V4h8v2m-9 0 1 14h8l1-14M10 10v6m4-6v6"/></svg>';
    deleteBtn.addEventListener('click', event => window.requestDeleteChat(chat.id, event));

    row.append(item, deleteBtn);
    list.appendChild(row);
  }
}

function showBlankConversation({ preserveContext = false } = {}) {
  activeChatId = null;
  renderSavedChatList();

  const chatHistory = document.getElementById('chat-history');
  const emptyState = document.getElementById('empty-state');
  const userInputEl = document.getElementById('user-input');

  if (chatHistory) {
    chatHistory.innerHTML = '';
    chatHistory.classList.add('hidden');
  }
  messageTrainingState.clear();

  if (!preserveContext) {
    resetVilotniConversationState();
    if (window.vilotni2Server?.clearDialogueContext) {
      window.vilotni2Server.clearDialogueContext().catch(error => console.warn('VilotNI 2 dialogue reset failed:', error));
    }
  }

  window.lastVilotni2Result = null;
  if (emptyState) emptyState.style.display = 'flex';
  if (userInputEl) {
    userInputEl.value = '';
    userInputEl.style.height = 'auto';
    window.updateSubmitBtnState('');
    userInputEl.focus();
  }
}

window.requestDeleteChat = function(chatId, event) {
  if (event) {
    event.preventDefault();
    event.stopPropagation();
  }
  const chat = savedChats.find(item => item.id === chatId);
  if (!chat) return;

  if (appSettings.chats.skipDeleteWarning) {
    window.deleteSavedChat(chatId);
    return;
  }

  pendingDeleteChatId = chatId;
  const modal = document.getElementById('delete-chat-modal');
  const title = document.getElementById('delete-chat-name');
  const skip = document.getElementById('delete-warning-skip-checkbox');
  if (title) title.textContent = chat.title || 'this conversation';
  if (skip) skip.checked = false;
  if (modal) {
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    requestAnimationFrame(() => modal.classList.add('is-open'));
  }
};

window.closeDeleteChatModal = function() {
  const modal = document.getElementById('delete-chat-modal');
  pendingDeleteChatId = null;
  if (!modal) return;
  modal.classList.remove('is-open');
  setTimeout(() => {
    modal.classList.add('hidden');
    modal.classList.remove('flex');
  }, 150);
};

window.confirmDeleteChat = function() {
  if (!pendingDeleteChatId) return;
  const chatId = pendingDeleteChatId;
  const skip = document.getElementById('delete-warning-skip-checkbox');
  if (skip?.checked) {
    appSettings.chats.skipDeleteWarning = true;
    persistAppSettings();
  }
  window.closeDeleteChatModal();
  window.deleteSavedChat(chatId);
};

window.deleteSavedChat = function(chatId) {
  const chatIndex = savedChats.findIndex(item => item.id === chatId);
  if (chatIndex < 0) return;
  const wasActive = activeChatId === chatId;
  savedChats.splice(chatIndex, 1);
  persistSavedChats();

  if (wasActive) {
    showBlankConversation({ preserveContext: appSettings.chats.retainContextAfterDelete });
  } else {
    renderSavedChatList();
  }
};

async function restoreShortTermContext(messages) {
  const transcript = Array.isArray(messages) ? messages : [];
  resetVilotniConversationState();

  // Rebuild VilotNI 1.5 working memory without touching episodic, semantic,
  // procedural, or learned network state.
  const store = window.agentInstance?.memoryStore;
  const vocab = window.vocabInstance;
  if (store?.working) store.working.length = 0;
  if (store?.addWorkingTurn && vocab?.tokenize) {
    let pendingUser = null;
    for (const message of transcript) {
      if (message.role === 'user') {
        pendingUser = message;
      } else if (message.role === 'assistant' && pendingUser) {
        const userTokens = vocab.tokenize(String(pendingUser.text || ''));
        const assistantTokens = vocab.tokenize(String(message.text || ''));
        store.addWorkingTurn({
          userTokens,
          assistantTokens,
          frame: 'conversation',
          queryTarget: 'content',
          answerAct: 'respond',
          resolved: true,
          timestamp: Number(message.at) || Date.now()
        });
        pendingUser = null;
      }
    }
  }

  // VilotNI 2 keeps dialogue context separately from learned RSL/correlation
  // state. Replacing only this buffer lets a restored chat continue naturally
  // without resetting the model's persistent learning.
  try {
    const core = await window.vilotni2CoreReady;
    if (core?.ok && window.vilotni2Server?.setDialogueContext) {
      const turns = transcript
        .filter(message => message.role === 'user' || message.role === 'assistant')
        .map(message => ({
          role: message.role,
          text: String(message.text || ''),
          at: Number(message.at) || Date.now(),
          confidence: Number.isFinite(Number(message.confidence)) ? Number(message.confidence) : null
        }));
      await window.vilotni2Server.setDialogueContext(turns);
    }
  } catch (error) {
    console.warn('Could not restore VilotNI 2 dialogue context:', error);
  }
}

window.openSavedChat = async function(chatId) {
  const chat = savedChats.find(item => item.id === chatId);
  if (!chat) return;

  activeChatId = chat.id;
  chat.updatedAt = Date.now();
  persistSavedChats();
  renderSavedChatList();

  const chatHistory = document.getElementById('chat-history');
  const emptyState = document.getElementById('empty-state');
  if (chatHistory) {
    chatHistory.innerHTML = '';
    chatHistory.classList.remove('hidden');
  }
  if (emptyState) emptyState.style.display = 'none';

  messageTrainingState.clear();
  suppressChatPersistence = true;
  try {
    for (const message of chat.messages) {
      if (message.role === 'user') {
        appendUserMessage(message.text, false);
      } else if (message.role === 'assistant') {
        await appendAIMessage(
          message.text,
          message.confidence ?? null,
          message.codeBlock ?? null,
          message.elapsedTime ?? null,
          null,
          message.confidenceDiagnostics ?? null,
          message.modelId || null,
          false
        );
      }
    }
  } finally {
    suppressChatPersistence = false;
  }

  await restoreShortTermContext(chat.messages);
  if (window.matchMedia('(max-width: 768px)').matches) window.closeMobileSidebar();
};

window.initSavedChats = function() {
  loadSavedChatsFromStorage();
  activeChatId = null;
  renderSavedChatList();
};

// ========================================================
// 1. NEURAL NODE CANVAS BACKGROUND
// ========================================================

const CanvasSystem = {
  canvas: null,
  ctx: null,
  particles: [],
  animationId: null,
  brandColors: ['#a855f7', '#ec4899', '#f97316', '#06b6d4'],
  mouse: { x: null, y: null, radius: 120 },

  init() {
    this.canvas = document.getElementById('bg-canvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    
    this.resize();
    window.addEventListener('resize', () => this.resize());
    
    window.addEventListener('mousemove', (e) => {
      this.mouse.x = e.clientX;
      this.mouse.y = e.clientY;
    });

    window.addEventListener('mouseout', () => {
      this.mouse.x = null;
      this.mouse.y = null;
    });

    this.createParticles();
    this.start();
  },

  resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
    this.createParticles();
  },

  createParticles() {
    this.particles = [];
    const particleCount = (this.canvas.width * this.canvas.height) / 20000;
    for (let i = 0; i < particleCount; i++) {
      const size = (Math.random() * 2.5) + 1;
      const x = Math.random() * (this.canvas.width - size * 2) + size * 2;
      const y = Math.random() * (this.canvas.height - size * 2) + size * 2;
      const dirX = (Math.random() * 0.3) - 0.15;
      const dirY = (Math.random() * 0.3) - 0.15;
      const color = this.brandColors[Math.floor(Math.random() * this.brandColors.length)];
      this.particles.push({ x, y, dirX, dirY, size, color });
    }
  },

  draw() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    const isLight = document.body.classList.contains('light-mode');
    const lineColor = isLight ? 'rgba(0, 0, 0, 0.05)' : 'rgba(255, 255, 255, 0.05)';
    
    for (let i = 0; i < this.particles.length; i++) {
      let p = this.particles[i];
      this.ctx.beginPath();
      this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2, false);
      this.ctx.fillStyle = p.color;
      this.ctx.globalAlpha = 0.5;
      this.ctx.fill();
      this.ctx.globalAlpha = 1.0;

      p.x += p.dirX;
      p.y += p.dirY;

      if (p.x > this.canvas.width || p.x < 0) p.dirX = -p.dirX;
      if (p.y > this.canvas.height || p.y < 0) p.dirY = -p.dirY;

      if (this.mouse.x != null) {
        let dx = this.mouse.x - p.x;
        let dy = this.mouse.y - p.y;
        let distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < this.mouse.radius) {
          if (this.mouse.x < p.x && p.x < this.canvas.width - p.size * 10) p.x += 0.5;
          if (this.mouse.x > p.x && p.x > p.size * 10) p.x -= 0.5;
          if (this.mouse.y < p.y && p.y < this.canvas.height - p.size * 10) p.y += 0.5;
          if (this.mouse.y > p.y && p.y > p.size * 10) p.y -= 0.5;
        }
      }

      for (let j = i; j < this.particles.length; j++) {
        let p2 = this.particles[j];
        let dx = p.x - p2.x;
        let dy = p.y - p2.y;
        let distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < 120) {
          this.ctx.beginPath();
          this.ctx.strokeStyle = lineColor;
          this.ctx.lineWidth = 1;
          this.ctx.moveTo(p.x, p.y);
          this.ctx.lineTo(p2.x, p2.y);
          this.ctx.stroke();
        }
      }
    }
  },

  animate() {
    this.draw();
    this.animationId = requestAnimationFrame(() => this.animate());
  },

  start() {
    if (!this.animationId) this.animate();
  }
};

// ========================================================
// 2. MODEL SWITCHER & CAROUSEL CONTROLS
// ========================================================

window.toggleModelPopup = function(e) {
  if (e) e.stopPropagation();
  const stack = document.getElementById('popup-stack-container');
  if (!stack) return;
  stack.classList.toggle('hidden');
};

window.getVilotni2Palette = function() {
  return { a: '#8300FB', b: '#FE0070', soft: 'rgba(254, 0, 112, 0.20)', name: 'VilotNI 2' };
};

window.applyVilotni2UI = function() {
  if (window.activeModel !== 'vilotni_2') return;
  const palette = window.getVilotni2Palette();
  const triggerName = document.getElementById('trigger-model-name');
  const triggerStatus = document.getElementById('trigger-model-status');
  const headerLabel = document.getElementById('header-model-label');
  const dockIndicator = document.getElementById('dock-model-indicator');
  const sessionLabel = document.getElementById('active-session-label');
  const sidebarTitle = document.getElementById('sidebar-app-title');
  const heroTitle = document.getElementById('empty-hero-title');
  const heroDesc = document.getElementById('empty-hero-desc');
  const suggestionsGrid = document.getElementById('suggestions-grid');
  const modelCheck = document.getElementById('model-check-2');

  document.documentElement.style.setProperty('--brand-primary', palette.a);
  document.documentElement.style.setProperty('--brand-secondary', palette.b);
  document.documentElement.style.setProperty('--vilotni-gradient', `linear-gradient(135deg, ${palette.a}, ${palette.b})`);
  if (modelCheck) modelCheck.style.color = palette.a;
  if (triggerName) triggerName.textContent = 'VilotNI 2';
  if (triggerStatus) { triggerStatus.textContent = ''; triggerStatus.classList.add('hidden'); }
  if (headerLabel) headerLabel.textContent = 'VilotNI 2';
  if (dockIndicator) dockIndicator.textContent = 'VilotNI 2';
  if (sessionLabel) sessionLabel.textContent = 'New conversation';
  if (sidebarTitle) sidebarTitle.textContent = 'VilotNI';
  if (heroTitle) {
    heroTitle.textContent = 'How can I help?';
    heroTitle.style.backgroundImage = 'none';
    heroTitle.style.webkitBackgroundClip = 'border-box';
    heroTitle.style.backgroundClip = 'border-box';
    heroTitle.style.color = 'var(--text-primary)';
  }
  if (heroDesc) heroDesc.textContent = 'Start with a direct question. Follow-up context is used after the conversation begins.';
  if (suggestionsGrid) suggestionsGrid.innerHTML = `
    <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" data-prompt="What's an AI?" onclick="window.fillInput(this.dataset.prompt)">
      <span>AI basics</span>
      <span>Ask what artificial intelligence is.</span>
    </button>
    <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('What is a neural network?')">
      <span>Neural networks</span>
      <span>Learn what a neural network is.</span>
    </button>
    <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('What is machine learning?')">
      <span>Machine learning</span>
      <span>Ask what machine learning means.</span>
    </button>
    <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('What does a CPU do?')">
      <span>Computer hardware</span>
      <span>Learn what a CPU does.</span>
    </button>`;
};

window.switchActiveModel = function(modelName) {
  window.activeModel = modelName;
  document.body.classList.toggle('vilotni2-active', modelName === 'vilotni_2');
  const modelTrigger = document.getElementById('model-popup-trigger');
  if (modelTrigger) modelTrigger.title = modelName === 'vilotni_2' ? 'Select model' : 'Select model and learning mode';

  if (modelName !== 'vilotni_2') {
    document.documentElement.style.setProperty('--brand-primary', '#6366f1');
    document.documentElement.style.setProperty('--brand-secondary', '#8b5cf6');
    document.documentElement.style.setProperty('--vilotni-gradient', 'linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899)');
    const v2Check = document.getElementById('model-check-2');
    if (v2Check) v2Check.style.color = '';
  }

  const ids = ['vilotni_1', 'vilotni_1_5', 'vilotni_2'];
  const itemByModel = {
    vilotni_1: document.getElementById('model-item-1'),
    vilotni_1_5: document.getElementById('model-item-1-5'),
    vilotni_2: document.getElementById('model-item-2')
  };
  const checkByModel = {
    vilotni_1: document.getElementById('model-check-1'),
    vilotni_1_5: document.getElementById('model-check-1-5'),
    vilotni_2: document.getElementById('model-check-2')
  };

  for (const id of ids) {
    const active = id === modelName;
    const item = itemByModel[id];
    const check = checkByModel[id];
    if (item) {
      item.className = active
        ? "w-full px-2.5 py-2 rounded-xl transition-all flex items-center justify-between cursor-pointer border border-[var(--border-highlight)] bg-[var(--bg-elevated)] text-[var(--text-primary)]"
        : "w-full px-2.5 py-2 rounded-xl transition-all flex items-center justify-between cursor-pointer border border-transparent hover:bg-[var(--bg-elevated)]/50 text-[var(--text-secondary)]";
    }
    if (check) check.classList.toggle('hidden', !active);
  }

  const learningCard = document.getElementById('learning-mode-card');
  const triggerName = document.getElementById('trigger-model-name');
  const triggerStatus = document.getElementById('trigger-model-status');
  const headerLabel = document.getElementById('header-model-label');
  const dockIndicator = document.getElementById('dock-model-indicator');
  const sessionLabel = document.getElementById('active-session-label');
  const sidebarTitle = document.getElementById('sidebar-app-title');
  const heroTitle = document.getElementById('empty-hero-title');
  const heroDesc = document.getElementById('empty-hero-desc');
  const suggestionsGrid = document.getElementById('suggestions-grid');

  const showLearningCard = modelName === 'vilotni_1_5';
  if (learningCard) {
    learningCard.hidden = !showLearningCard;
    learningCard.setAttribute('aria-hidden', showLearningCard ? 'false' : 'true');
    if (showLearningCard) {
      learningCard.style.maxHeight = '300px';
      learningCard.style.paddingTop = '';
      learningCard.style.paddingBottom = '';
      learningCard.style.marginBottom = '';
      learningCard.style.overflow = 'visible';
      learningCard.style.pointerEvents = 'auto';
      void learningCard.offsetWidth;
      learningCard.style.opacity = '1';
      learningCard.style.transform = 'translateY(0) scale(1)';
    } else {
      learningCard.style.opacity = '0';
      learningCard.style.transform = 'translateY(10px) scale(0.95)';
      learningCard.style.maxHeight = '0px';
      learningCard.style.paddingTop = '0px';
      learningCard.style.paddingBottom = '0px';
      learningCard.style.marginBottom = '0px';
      learningCard.style.overflow = 'hidden';
      learningCard.style.pointerEvents = 'none';
    }
  }

  if (modelName === 'vilotni_1') {
    if (triggerName) triggerName.textContent = 'VilotNI 1';
    if (triggerStatus) triggerStatus.classList.add('hidden');
    if (headerLabel) headerLabel.textContent = 'VilotNI 1';
    if (dockIndicator) dockIndicator.textContent = 'VilotNI 1';
    if (sessionLabel) sessionLabel.textContent = 'New conversation';
    if (sidebarTitle) sidebarTitle.textContent = 'VilotNI';
    if (heroTitle) heroTitle.textContent = 'How can I help?';
    if (heroDesc) heroDesc.textContent = 'Ask a question, explore an idea, or start a new conversation.';
    if (suggestionsGrid) suggestionsGrid.innerHTML = `
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('Explain heuristic evaluation')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] group-hover:text-[var(--brand-secondary)] transition-colors">Test Cognitive Engine</span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">Ask the original cognitive engine about its heuristic logic.</span>
      </button>`;
  } else if (modelName === 'vilotni_2') {
    window.applyVilotni2UI();
  } else {
    if (triggerName) triggerName.textContent = 'VilotNI 1.5';
    if (triggerStatus) triggerStatus.classList.remove('hidden');
    if (headerLabel) headerLabel.textContent = 'VilotNI 1.5';
    if (dockIndicator) dockIndicator.textContent = 'VilotNI 1.5';
    if (sessionLabel) sessionLabel.textContent = 'New conversation';
    if (sidebarTitle) sidebarTitle.textContent = 'VilotNI';
    if (heroTitle) heroTitle.textContent = 'How can I help?';
    if (heroDesc) heroDesc.textContent = 'Ask a question, explore an idea, or continue a conversation with VilotNI 1.5.';
    if (suggestionsGrid) suggestionsGrid.innerHTML = `
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('Does a Neural Inspector calculate compute speed?')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] group-hover:text-[var(--brand-primary)] transition-colors">Compute Performance</span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">Inquire about processing power and accelerator speed.</span>
      </button>
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('How will AI model chips accelerate?')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] group-hover:text-[var(--brand-secondary)] transition-colors">Modern Processors</span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">Discuss future hardware models and specialized chips.</span>
      </button>`;
  }

  window.refreshDockIndicators();
  if (modelName === 'vilotni_1_5') window.refreshActiveModeStyling();
  const input = document.getElementById('user-input');
  if (input) window.updateSubmitBtnState(input.value);
  window.applyAppAppearanceSettings?.();
};

window.refreshDockIndicators = function() {
  const dockModel = document.getElementById('dock-model-indicator');
  const separator = document.getElementById('dock-separator');
  const learningIndicator = document.getElementById('dock-learning-indicator');
  const depthIndicator = document.getElementById('dock-depth-indicator');
  if (!dockModel || !learningIndicator || !depthIndicator) return;

  const isModel1 = window.activeModel === 'vilotni_1';
  const isModel2 = window.activeModel === 'vilotni_2';
  const agent = window.agentInstance;
  const isRsl = agent?.mode === 'rsl';
  const isDeep = isRsl && Boolean(agent?.isDeepLearning);

  learningIndicator.style.color = '';
  learningIndicator.className = 'text-[10px] font-mono font-medium text-[var(--text-tertiary)]';
  depthIndicator.className = 'text-[10px] font-mono text-[var(--text-tertiary)]';

  if (isModel1) {
    dockModel.textContent = 'VilotNI 1';
    if (separator) separator.textContent = ' • ';
    learningIndicator.textContent = 'Cognitive';
    depthIndicator.textContent = '';
  } else if (isModel2) {
    // VilotNI 2 has one unified runtime. Do not expose a model/learning type,
    // neuron count, phase label, or latency target in the consumer UI.
    dockModel.textContent = 'VilotNI 2';
    if (separator) separator.textContent = ' • ';
    learningIndicator.textContent = window.vilotni2CoreState === 'ready' ? 'Ready' : 'Starting';
    depthIndicator.textContent = '';
  } else {
    dockModel.textContent = 'VilotNI 1.5';
    if (separator) separator.textContent = ' • ';
    learningIndicator.textContent = isRsl ? (isDeep ? 'Deep RSL' : 'RSL') : 'RL';
    depthIndicator.textContent = isRsl ? (isDeep ? ' • Adaptive Search' : ' • Fast Search') : ' • 256 tokens';
  }
};

window.stepLearningMode = function(targetMode) {
  const agent = window.agentInstance;
  if (!agent) return;
  agent.setMode(targetMode);

  const carouselContent = document.getElementById('carousel-mode-content');
  if (carouselContent) {
    const animClass = targetMode === 'user' ? 'animate-slide-right' : 'animate-slide-left';
    carouselContent.classList.remove('animate-slide-right', 'animate-slide-left');
    void carouselContent.offsetWidth;
    carouselContent.classList.add(animClass);
  }

  window.refreshActiveModeStyling();
};

window.refreshActiveModeStyling = function() {
  const agent = window.agentInstance;
  if (!agent) return;

  const mode = agent.mode;
  const isDeep = Boolean(agent.isDeepLearning);

  const caretLeft = document.getElementById('caret-left-btn');
  const caretRight = document.getElementById('caret-right-btn');
  const carouselIcon = document.getElementById('carousel-mode-icon');
  const carouselTitle = document.getElementById('carousel-mode-title');
  const carouselDesc = document.getElementById('carousel-mode-desc');
  const deepContainer = document.getElementById('deep-learning-container');
  const triggerStatus = document.getElementById('trigger-model-status');
  const statusPill = document.getElementById('learning-status-pill');
  const depthThumb = document.getElementById('depth-slider-thumb');
  const depthSubtext = document.getElementById('depth-subtext');
  const deepSvg = document.getElementById('deep-learning-svg');
  const deepCircle = document.getElementById('deep-learning-circle');
  const rayLines = document.querySelectorAll('#deep-learning-svg .ray-line');

  window.refreshDockIndicators();

  if (mode === 'rsl') {
    if (caretLeft) caretLeft.classList.add('invisible');
    if (caretRight) caretRight.classList.remove('invisible');

    if (carouselTitle) carouselTitle.textContent = "RSL (Reinforcement Self-Learning)";
    if (carouselDesc) carouselDesc.textContent = "Anytime compute & dynamic output length.";
    if (deepContainer) deepContainer.classList.remove('hidden');

    if (isDeep) {
      if (triggerStatus) {
        triggerStatus.textContent = "High";
        triggerStatus.className = "text-[11px] font-mono font-bold bg-gradient-to-r from-amber-400 via-orange-400 to-red-500 bg-clip-text text-transparent leading-none";
      }
      if (statusPill) {
        statusPill.textContent = "RSL Deep";
        statusPill.className = "text-[10px] font-mono font-bold bg-gradient-to-r from-amber-400 via-orange-400 to-red-500 bg-clip-text text-transparent";
      }
      if (carouselIcon) carouselIcon.className = "ph ph-lightning text-sm text-orange-400";

      if (depthThumb) {
        depthThumb.style.transform = "translateX(100%)";
        depthThumb.textContent = "Adaptive Pareto Search";
        depthThumb.className = "w-1/2 h-full rounded-lg transition-all duration-300 flex items-center justify-center font-display font-bold text-[10px] text-white shadow-sm z-10 bg-gradient-to-r from-amber-500 to-red-500 translate-x-full";
      }
      if (depthSubtext) depthSubtext.textContent = "1 neural forward + live latent self-conditioning";

      rayLines.forEach(l => l.style.display = 'inline');
      if (deepCircle) deepCircle.classList.remove('deep-circle-anim');
      if (deepSvg) deepSvg.className = "w-4 h-4 text-orange-400";

    } else {
      if (triggerStatus) {
        triggerStatus.textContent = "Medium";
        triggerStatus.className = "text-[11px] font-mono font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent leading-none";
      }
      if (statusPill) {
        statusPill.textContent = "RSL";
        statusPill.className = "text-[10px] font-mono font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent";
      }
      if (carouselIcon) carouselIcon.className = "ph ph-lightning text-sm text-purple-400";

      if (depthThumb) {
        depthThumb.style.transform = "translateX(0%)";
        depthThumb.textContent = "Fast Search";
        depthThumb.className = "w-1/2 h-full rounded-lg transition-all duration-300 flex items-center justify-center font-display font-bold text-[10px] text-zinc-300 shadow-sm z-10 bg-[var(--bg-elevated)] border border-[var(--border-subtle)] translate-x-0";
      }
      if (depthSubtext) depthSubtext.textContent = "Evaluates fast candidate rollouts";

      rayLines.forEach(l => l.style.display = 'none');
      if (deepCircle) deepCircle.classList.add('deep-circle-anim');
      if (deepSvg) deepSvg.className = "w-4 h-4 text-purple-400";
    }

  } else {
    if (caretLeft) caretLeft.classList.remove('invisible');
    if (caretRight) caretRight.classList.add('invisible');

    if (carouselTitle) carouselTitle.textContent = "RL (Reinforcement Learning)";
    if (carouselDesc) carouselDesc.textContent = "Direct 1-pass execution.";
    if (carouselIcon) carouselIcon.className = "ph ph-lightning-slash text-sm text-cyan-400";

    if (triggerStatus) {
      triggerStatus.textContent = "Low";
      triggerStatus.className = "text-[11px] font-mono font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent leading-none";
    }
    if (statusPill) {
      statusPill.textContent = "RL";
      statusPill.className = "text-[10px] font-mono font-bold bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-transparent";
    }

    if (deepContainer) deepContainer.classList.add('hidden');
  }
};

window.toggleDeepLearning = function(e) {
  if (e) e.stopPropagation();
  const agent = window.agentInstance;
  if (!agent || agent.mode !== 'rsl') return;

  agent.isDeepLearning = !agent.isDeepLearning;
  window.refreshActiveModeStyling();
};

document.addEventListener('click', (e) => {
  const stack = document.getElementById('popup-stack-container');
  const trigger = document.getElementById('model-popup-trigger');
  if (stack && !stack.classList.contains('hidden')) {
    if (!stack.contains(e.target) && !trigger.contains(e.target)) {
      stack.classList.add('hidden');
    }
  }
});

// ========================================================
// 3. DIALOGUE & EXECUTION LOOP (ADAPTIVE TIME FORMATTER)
// ========================================================

function scrollToBottom() {
  const chatScrollArea = document.getElementById('chat-scroll-area');
  if (chatScrollArea) {
    chatScrollArea.scrollTo({ top: chatScrollArea.scrollHeight, behavior: 'smooth' });
  }
}

function appendUserMessage(text, persistMessage = true) {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'flex w-full justify-end animate-fade-in';
  const bubble = document.createElement('div');
  bubble.className = 'msg-user max-w-[85%] rounded-2xl px-5 py-3.5 text-[15px] leading-relaxed shadow-sm';
  bubble.innerText = text;
  wrapper.appendChild(bubble);
  chatHistory.appendChild(wrapper);
  if (persistMessage) {
    recordChatMessage({ role: 'user', text: String(text || '') });
  }
  scrollToBottom();
}

function showThinkingLoader(initialStatus = "Generating...") {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return { id: '', textId: '' };

  const id = 'loader-' + Date.now();
  const textId = `${id}-text`;

  const wrapper = document.createElement('div');
  wrapper.id = id;
  wrapper.className = 'flex w-full justify-start mb-6 animate-fade-in';
  wrapper.innerHTML = `
    <div class="flex flex-col space-y-2 py-1">
      <span id="${textId}" class="text-[13px] font-medium text-[var(--text-secondary)] select-none">
        ${initialStatus}
      </span>
      <div class="h-2 w-32 rounded-full bg-[var(--border-subtle)]/70 overflow-hidden relative shadow-inner">
        <div class="w-full h-full rounded-full bg-gradient-to-r from-transparent via-[var(--brand-primary)]/50 to-transparent animate-pulse"></div>
      </div>
    </div>
  `;

  chatHistory.appendChild(wrapper);
  scrollToBottom();

  return { id, textId };
}

function removeLoader(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.remove();
}

function formatConfidenceDiagnostics(d) {
  if (!d || !window['showConfidenceDiagnostics']) return "";

  const pct = value => {
    const n = Number(value);
    return Number.isFinite(n)
      ? `${Math.round(Math.max(0, Math.min(1, n)) * 100)}`
      : "0";
  };

  const mode =
    d.mode === "deep"
      ? `High ${d.passesUsed || 1}/${d.preferredPasses || 1}p`
      : `Low ${d.passesUsed || 1}p`;

  const compact =
    `Rel ${pct(d.relevance)} | ` +
    `Lang ${pct(d.language)} | ` +
    `Sem ${pct(d.semantic)} | ` +
    `Ans ${pct(d.answer)} | ` +
    `Coh ${pct(d.coherence)} | ` +
    `Seq ${pct(d.sequence)}`;

  const tooltip =
    `Coverage ${pct(d.promptCoverage)} | ` +
    `Precision ${pct(d.contentPrecision)} | ` +
    `Route ${pct(d.dualRouteCoherence)} | ` +
    `Syntax ${pct(d.syntax)} | ` +
    `Order ${pct(d.clauseOrder)} | ` +
    `Fluency ${pct(d.fluency)} | ` +
    `Consistency ${pct(d.consistency)} | ` +
    `Neural ${pct(d.neural)} | ` +
    `Reasoning ${pct(d.reasoning)} | ` +
    `Integrity ${pct(d.integrity)} | ` +
    `Raw ${pct(d.raw)} | ` +
    `Gate ${pct(d.highQualityGate)} | ` +
    `Final ${pct(d.final)} | ` +
    `Anchors ${d.anchorBudget || 0}/${d.uniquePromptContent || 0} | ` +
    `Budget ${d.generationBudget || 0} | ` +
    `AR Latent ${d.autoregressiveLatentUpdates || 0} updates | ` +
    `AR Stability ${pct(d.autoregressiveLatentStability)} | ` +
    `Live Pool ${d.peakLiveCandidatePoolSize || d.candidatePoolSize || 0}`;

  return `
    <div
      class="mt-1 mb-2 text-[9px] font-mono text-[var(--text-tertiary)] leading-relaxed select-text"
      title="${tooltip.replace(/"/g, '&quot;')}"
    >
      <span class="opacity-70">${mode}</span>
      <span class="mx-1 opacity-40">•</span>
      <span>${compact}</span>
    </div>
  `;
}

window.toggleInferenceDetails = function(msgId, button) {
  const panel = document.getElementById(`inference-details-${msgId}`);
  if (!panel) return;
  const open = panel.classList.toggle('is-open');
  panel.setAttribute('aria-hidden', open ? 'false' : 'true');
  if (button) button.setAttribute('aria-expanded', open ? 'true' : 'false');
};
// Compatibility for older rendered messages while moving every model to the shared response UI.
window.toggleVilotni2InferenceDetails = window.toggleInferenceDetails;

async function appendAIMessage(
  generatedText,
  confidencePct = null,
  codeBlock = null,
  elapsedTime = null,
  trainingState = null,
  confidenceDiagnostics = null,
  modelIdOverride = null,
  persistMessage = true
) {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return;

  const responseModel = modelIdOverride || window.activeModel;
  const isModel1 = responseModel === 'vilotni_1';
  const isModel2 = responseModel === 'vilotni_2';
  const isModel15 = responseModel === 'vilotni_1_5';
  const isRsl = !window.agentInstance || window.agentInstance.mode === 'rsl';
  const isDeep = window.agentInstance && window.agentInstance.isDeepLearning;

  const avatarGradient = isModel1
    ? "from-blue-600 to-indigo-700"
    : isModel2
      ? ""
      : (!isRsl
          ? "from-cyan-500 to-blue-600"
          : (isDeep ? "from-amber-500 to-red-500" : "from-purple-500 to-pink-500"));

  const badgeClasses = isModel1
    ? "border-indigo-500/30 bg-indigo-500/10 text-indigo-300"
    : isModel2
      ? "vilotni2-confidence-badge text-[var(--text-primary)]"
      : (!isRsl
          ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
          : (isDeep ? "border-orange-500/30 bg-orange-500/10 text-orange-300" : "border-purple-500/30 bg-purple-500/10 text-purple-300"));

  const modelLabel = isModel1 ? "VilotNI 1" : (isModel2 ? "VilotNI 2" : "VilotNI 1.5");
  const modelLabelStyle = '';

  const msgId = 'msg-' + Date.now();

  if (isModel15 && trainingState) {
    messageTrainingState.set(msgId, {
      promptTokens: Array.from(trainingState.promptTokens || []),
      responseTokens: Array.from(trainingState.responseTokens || []),
      trajectory: Array.isArray(trainingState.trajectory)
        ? trainingState.trajectory.map(step => ({ ...step }))
        : []
    });
  }

  const wrapper = document.createElement('div');
  wrapper.id = `msg-wrap-${msgId}`;
  wrapper.className = 'flex w-full justify-start animate-fade-in mb-8 group/msg';

  const safeElapsed = elapsedTime || '—';
  const safeConfidence = confidencePct !== null ? `${confidencePct}%` : '—';
  const inferenceMode = isModel15
    ? (!isRsl ? 'User RL' : (isDeep ? 'Deep RL' : 'RSL'))
    : null;
  const detailTarget = isModel2
    ? `<div><span>Latency target</span><strong>&lt;40 ms</strong></div>`
    : '';
  const detailMode = inferenceMode
    ? `<div><span>Mode</span><strong>${inferenceMode}</strong></div>`
    : '';
  const model15Diagnostics = isModel15 && confidenceDiagnostics
    ? `<div class="inference-diagnostics">${formatConfidenceDiagnostics(confidenceDiagnostics)}</div>`
    : '';

  wrapper.innerHTML = `
    <div class="model-response-shell w-full">
      <div class="model-response-layout">
        <div class="model-response-rail" aria-hidden="true">
          <span class="model-brand-icon" aria-hidden="true"></span>
        </div>

        <div class="model-response-main">
          <button
            type="button"
            class="inference-trigger"
            onclick="window.toggleInferenceDetails('${msgId}', this)"
            aria-expanded="false"
            aria-controls="inference-details-${msgId}"
            title="Show inference details"
          >Inference</button>

          <div id="inference-details-${msgId}" class="inference-details" aria-hidden="true">
            <div><span>Model</span><strong>${modelLabel}</strong></div>
            <div><span>Latency</span><strong>${safeElapsed}</strong></div>
            <div><span>Confidence</span><strong>${safeConfidence}</strong></div>
            ${detailTarget}
            ${detailMode}
            ${model15Diagnostics}
          </div>

          <div id="text-${msgId}" class="model-response-text leading-[1.72] text-[15px] space-y-4 text-[var(--text-primary)] font-normal"></div>
          ${codeBlock ? `
          <div class="mt-3 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded-xl p-4 overflow-x-auto text-xs font-mono text-zinc-300">
            <pre><code>${codeBlock.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
          </div>` : ''}

          <div id="action-${msgId}" class="model-response-actions mt-4 flex items-center space-x-2 opacity-0 group-hover/msg:opacity-100 transition-opacity duration-300">
            ${isModel15 ? `
            <button type="button" onclick="window.handleRegenerate('${msgId}')" class="flex items-center justify-center space-x-1.5 px-3 py-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] text-[11px] font-bold uppercase tracking-widest text-[var(--text-tertiary)] hover:text-indigo-400 hover:border-indigo-400 transition-all shadow-sm group cursor-pointer" title="Regenerate">
              <i class="ph ph-arrows-clockwise text-xs group-hover:rotate-180 transition-transform duration-500"></i>
              <span>Regenerate</span>
            </button>
            ${window.agentInstance && window.agentInstance.mode === 'user' ? `
            <button type="button" onclick="window.applyFeedback(1.0, '${msgId}', this)" class="p-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] text-[var(--text-tertiary)] hover:text-emerald-400 hover:border-emerald-500/40 transition-all shadow-sm cursor-pointer" title="Teacher: Good response">
              <i class="ph ph-thumbs-up text-xs"></i>
            </button>
            <button type="button" onclick="window.applyFeedback(-1.0, '${msgId}', this)" class="p-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] text-[var(--text-tertiary)] hover:text-rose-400 hover:border-rose-500/40 transition-all shadow-sm cursor-pointer" title="Teacher: Bad response">
              <i class="ph ph-thumbs-down text-xs"></i>
            </button>` : ''}
            ` : ''}
            <button type="button" onclick="window.copyMessageText('${msgId}', this)" class="flex items-center justify-center space-x-1.5 px-3 py-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] text-[11px] font-bold uppercase tracking-widest text-[var(--text-tertiary)] hover:text-indigo-400 hover:border-indigo-400 transition-all shadow-sm cursor-pointer">
              <i class="ph ph-copy text-xs"></i>
              <span>Copy</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  `;
  chatHistory.appendChild(wrapper);

  const textContainer = document.getElementById(`text-${msgId}`);
  if (textContainer) {
    textContainer.innerText = generatedText || "";
  }
  if (persistMessage) {
    recordChatMessage({
      role: 'assistant',
      text: String(generatedText || ''),
      modelId: responseModel,
      confidence: confidencePct,
      elapsedTime,
      codeBlock,
      confidenceDiagnostics
    });
  }
  scrollToBottom();
}

function formatVilotni2RSLResult(result) {
  const hot = Number.isFinite(result?.hotPathMs) ? result.hotPathMs.toFixed(3) : '?';
  const round = Number.isFinite(result?.roundTripMs) ? result.roundTripMs.toFixed(3) : '?';
  const novelty = Number.isFinite(result?.novelty) ? (result.novelty * 100).toFixed(1) : '?';
  const momentum = Number.isFinite(result?.momentum) ? result.momentum.toFixed(3) : '?';
  const energy = result?.energy;
  const energyLine = energy
    ? `State energy: mean |H| ${energy.meanAbs.toFixed(5)}, peak ${energy.peak.toFixed(5)}.`
    : 'State energy was not synchronized back from the GPU on this pass.';

  return [
    `RSL 2 accepted ${result?.lexicalSteps || 0} lexical steps into persistent H_t.`,
    `State revision ${result?.stateRevision ?? '?'} • novelty ${novelty}% • momentum ${momentum}.`,
    `Worker hot path ${hot} ms • message roundtrip ${round} ms.`,
    energyLine,
    '',
    result?.note || 'Processor.js, Decoder.js, Words.json, Lexicon.js, WordState.js, PhraseState.js, WordMatrix.js, the resident candidate matrix, FastLane, Correlation.js, curiosity exploration, and learned POS/context RSL are attached to the persistent VilotNI 2 worker.'
  ].join('\n');
}

window.handleSend = async function() {
  const input = document.getElementById('user-input');
  const chatHistory = document.getElementById('chat-history');
  const emptyState = document.getElementById('empty-state');
  if (!input) return;

  const text = input.value.trim();
  if (!text || isProcessing) return;

  isProcessing = true;
  if (emptyState) emptyState.style.display = 'none';
  if (chatHistory) chatHistory.classList.remove('hidden');

  appendUserMessage(text);
  input.value = '';
  input.style.height = 'auto';
  window.updateSubmitBtnState('');

  let activeLoader = null;

  try {
    if (window.activeModel === 'vilotni_1') {
      activeLoader = showThinkingLoader("Synthesizing...");
      const t0 = performance.now();

      if (typeof VilotNI === 'undefined') {
        throw new Error("VilotNI 1 engine script not loaded.");
      }
      const v1Result = VilotNI.process(text);
      const elapsed = formatElapsed(performance.now() - t0);

      removeLoader(activeLoader.id);
      await appendAIMessage(v1Result.text, null, v1Result.code || null, elapsed);
    } else if (window.activeModel === 'vilotni_2') {
      if (window.vilotni2CoreState === 'initializing') {
        activeLoader = showThinkingLoader('Starting the 6M-neuron engine...');
      }

      const core = await window.vilotni2CoreReady;
      if (!core?.ok || !window.vilotni2Server) {
        throw new Error(
          core?.error?.message ||
          window.vilotni2CoreError?.message ||
          'VilotNI 2 engine failed to initialize.'
        );
      }

      if (!activeLoader) activeLoader = showThinkingLoader('Running VilotNI 2 fast response lane...');
      const t0 = performance.now();
      const responseProgressHandler = event => {
        const detail = event?.detail || {};
        const label = activeLoader?.textId ? document.getElementById(activeLoader.textId) : null;
        if (!label) return;
        const confidence = Number(detail.bestConfidence);
        const elapsedMs = Number(detail.elapsedMs);
        if (detail.stage === 'context-resolved') {
          const turns = Number(detail.contextTurnsUsed) || 1;
          label.textContent = `Using ${turns} previous conversation turn${turns === 1 ? '' : 's'} as context...`;
        } else if (detail.stage === 'context-required') {
          label.textContent = 'Waiting for conversation context...';
        } else {
          label.textContent = Number.isFinite(confidence)
            ? `Searching candidate batch ${detail.batch || 1} • best ${confidence.toFixed(1)}% • ${detail.candidates || 0} candidates${Number.isFinite(elapsedMs) ? ` • ${elapsedMs.toFixed(1)} ms` : ''}`
            : 'Running VilotNI 2 fast response lane...';
        }
      };
      window.addEventListener('vilotni2responseprogress', responseProgressHandler);
      let result;
      try {
        result = await window.vilotni2Server.generate(text, {
          syncState: false
        });
      } finally {
        window.removeEventListener('vilotni2responseprogress', responseProgressHandler);
      }
      window.lastVilotni2Result = result;
      const elapsed = formatElapsed(performance.now() - t0);

      removeLoader(activeLoader.id);
      await appendAIMessage(
        result.text || formatVilotni2RSLResult(result),
        Number.isFinite(result.confidence) ? result.confidence : null,
        null,
        elapsed,
        null,
        null
      );
      window.refreshDockIndicators();
    } else {
      if (window.vilotni15CoreState === "initializing") {
        activeLoader = showThinkingLoader("Initializing neural core...");
      }

      const core = await window.vilotni15CoreReady;

      if (!core?.ok) {
        throw new Error(
          core?.error?.message ||
          window.vilotni15CoreError?.message ||
          "VilotNI 1.5 neural core failed to initialize."
        );
      }

      const vocab = window.vocabInstance;
      const agent = window.agentInstance;

      if (!vocab || !vocab.loaded || !agent) {
        throw new Error(
          "VilotNI 1.5 core reported ready without a usable vocabulary/agent."
        );
      }

      // Defensive hot-reload guard.
      if (!vocab.loaded && vocab.ready) {
        await vocab.ready;
      }

      if (!activeLoader) {
        activeLoader = showThinkingLoader("Generating...");
      }

      lastPromptTokens = vocab.tokenize(text);

      const t0 = performance.now();
      const result = await agent.generate(lastPromptTokens);
      window.lastVilotni15Diagnostics = result.confidenceDiagnostics || null;
      const elapsed = formatElapsed(performance.now() - t0);

      const responseText = vocab.detokenize(result.tokens);

      const trainingState = {
        promptTokens: Array.from(lastPromptTokens),
        responseTokens: Array.from(result.tokens || []),
        trajectory: Array.isArray(agent.trajectory)
          ? agent.trajectory.map(step => ({ ...step }))
          : []
      };

      removeLoader(activeLoader.id);
      await appendAIMessage(
        responseText,
        result.confidence,
        null,
        elapsed,
        trainingState,
        result.confidenceDiagnostics || null
      );
    }
  } catch (err) {
    console.error("Inference Error:", err);
    if (activeLoader) removeLoader(activeLoader.id);
    await appendAIMessage(`[Offline: ${err.message}]`, "0.0", null, "<1ms");
  }

  isProcessing = false;
  if (input) window.updateSubmitBtnState(input.value);
};

window.applyFeedback = async function(rewardValue, msgId, btnElement) {
  const agent = window.agentInstance;
  const network = window.networkInstance;

  if (!agent) return;

  const state = messageTrainingState.get(msgId) || null;

  try {
    // Capture the exact prompt+response trace being rated. This happens only
    // when feedback is clicked, so normal generation latency is unaffected.
    if (
      state &&
      network &&
      typeof network.captureRewardTrace === 'function'
    ) {
      const rewardSequence = [
        ...state.promptTokens,
        ...state.responseTokens
      ];

      await network.captureRewardTrace(
        rewardSequence,
        state.responseTokens
      );
    }

    // RSL word-pattern learning uses the historical response trajectory.
    agent.train(
      rewardValue,
      state?.trajectory || null
    );

    // Apply head/deep policy reward only after the exact trace is captured.
    if (
      state &&
      network &&
      typeof network.applyPolicyReward === 'function'
    ) {
      const updateDeep =
        typeof agent.shouldUpdateDeepSynapses === 'function'
          ? agent.shouldUpdateDeepSynapses()
          : Boolean(agent.isDeepLearning);

      network.applyPolicyReward(
        rewardValue,
        0.008,
        updateDeep
      );
    }

    if (btnElement) {
      btnElement.classList.add(
        rewardValue > 0
          ? 'text-emerald-400'
          : 'text-rose-400'
      );

      btnElement.style.borderColor =
        rewardValue > 0
          ? 'rgba(52, 211, 153, 0.4)'
          : 'rgba(244, 63, 94, 0.4)';
    }
  } catch (err) {
    console.warn("Feedback learning failed:", err);
  }
};

window.copyMessageText = function(msgId, btn) {
  const textEl = document.getElementById(`text-${msgId}`);
  if (!textEl) return;
  navigator.clipboard.writeText(textEl.innerText);
  const orig = btn.innerHTML;
  btn.innerHTML = '<span class="text-emerald-400 font-bold">COPIED</span>';
  setTimeout(() => { btn.innerHTML = orig; }, 1800);
};

window.handleRegenerate = async function(msgId) {
  if (isProcessing || !window.agentInstance || lastPromptTokens.length === 0) return;
  await window.applyFeedback(-0.5, msgId, null);

  const parentMsg = document.getElementById(`msg-wrap-${msgId}`);
  if (parentMsg) parentMsg.remove();
  messageTrainingState.delete(msgId);

  isProcessing = true;
  const regenLoader = showThinkingLoader("Generating...");

  try {
    const vocab = window.vocabInstance;
    const agent = window.agentInstance;

    const t0 = performance.now();
    const result = await agent.generate(lastPromptTokens);
    window.lastVilotni15Diagnostics = result.confidenceDiagnostics || null;
    const elapsed = formatElapsed(performance.now() - t0);

    const responseText = vocab.detokenize(result.tokens);

    const trainingState = {
      promptTokens: Array.from(lastPromptTokens),
      responseTokens: Array.from(result.tokens || []),
      trajectory: Array.isArray(agent.trajectory)
        ? agent.trajectory.map(step => ({ ...step }))
        : []
    };

    removeLoader(regenLoader.id);
    await appendAIMessage(
      responseText,
      result.confidence,
      null,
      elapsed,
      trainingState,
      result.confidenceDiagnostics || null
    );
  } catch (err) {
    removeLoader(regenLoader.id);
    console.error(err);
  }

  isProcessing = false;
};

// ========================================================
// 4. UI SHELL & LIFECYCLE
// ========================================================

window.syncSidebarUI = function() {
  const sidebar = document.getElementById('sidebar');
  const toggleBtn = document.getElementById('sidebar-toggle-btn');
  const closeBtn = document.getElementById('sidebar-close-btn');
  const overlay = document.getElementById('sidebar-overlay');
  if (!sidebar) return;

  const mobile = window.matchMedia('(max-width: 768px)').matches;
  const collapsed = sidebar.classList.contains('collapsed');
  const open = sidebar.classList.contains('open');

  if (toggleBtn) {
    toggleBtn.style.display = (!mobile && collapsed) ? 'block' : 'none';
    toggleBtn.setAttribute('aria-expanded', String(!collapsed));
  }
  if (closeBtn) closeBtn.setAttribute('aria-expanded', String(!collapsed));
  if (overlay) {
    if (mobile && open) overlay.classList.remove('hidden');
    else overlay.classList.add('hidden');
  }
};

window.toggleDesktopSidebar = function(e) {
  if (e) e.stopPropagation();
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  if (window.matchMedia('(max-width: 768px)').matches) {
    window.toggleMobileSidebar(e);
    return;
  }

  sidebar.classList.toggle('collapsed');
  sidebar.classList.remove('open');
  window.syncSidebarUI();
};

window.toggleMobileSidebar = function(e) {
  if (e) e.stopPropagation();
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;

  sidebar.classList.remove('collapsed');
  sidebar.classList.toggle('open');
  window.syncSidebarUI();
};

window.closeMobileSidebar = function() {
  const sidebar = document.getElementById('sidebar');
  if (!sidebar) return;
  sidebar.classList.remove('open');
  window.syncSidebarUI();
};

window.toggleTheme = function(e) {
  if (e) e.stopPropagation();
  appSettings.appearance.theme = appSettings.appearance.theme === 'light' ? 'dark' : 'light';
  persistAppSettings();
  applyAppAppearanceSettings();
  window.syncSettingsPanelUI?.();
};

window.initTheme = function() {
  loadAppSettings();
  applyAppAppearanceSettings();
};

window.startNewChat = function(e) {
  if (e) e.stopPropagation();
  showBlankConversation({ preserveContext: false });
  if (window.matchMedia('(max-width: 768px)').matches) window.closeMobileSidebar();
};

function resolveProfileColor(value, fallback = '#6366f1') {
  if (!value) return fallback;
  if (value.startsWith('var(')) {
    const match = value.match(/var\((.*?)\)/);
    return match ? `var(${match[1]})` : fallback;
  }
  return value;
}

window.updateProfileUI = function() {
  const name = (userProfile.name || 'User').trim() || 'User';
  const sidebarName = document.getElementById('sidebar-user-name');
  const sidebarBg = document.getElementById('sidebar-avatar-bg');
  const sidebarInitial = document.getElementById('sidebar-avatar-initial');
  const sidebarImg = document.getElementById('sidebar-avatar-img');

  if (sidebarName) sidebarName.textContent = name;
  if (sidebarInitial) {
    sidebarInitial.textContent = name.charAt(0).toUpperCase();
    sidebarInitial.style.color = resolveProfileColor(userProfile.textColor, '#ffffff');
  }
  if (sidebarBg) sidebarBg.style.backgroundColor = userProfile.avatarUrl ? 'transparent' : resolveProfileColor(userProfile.bgColor, 'var(--brand-primary)');
  if (sidebarImg) {
    if (userProfile.avatarUrl) {
      sidebarImg.src = userProfile.avatarUrl;
      sidebarImg.classList.remove('hidden');
      sidebarInitial?.classList.add('hidden');
    } else {
      sidebarImg.src = '';
      sidebarImg.classList.add('hidden');
      sidebarInitial?.classList.remove('hidden');
    }
  }
};

window.fillInput = function(text) {
  const input = document.getElementById('user-input');
  if (!input) return;

  input.value = String(text ?? '');
  input.style.height = 'auto';
  input.style.height = input.scrollHeight + 'px';

  if (typeof window.updateSubmitBtnState === 'function') {
    window.updateSubmitBtnState(input.value);
  }

  input.focus();
};

window.updateSubmitBtnState = function(val) {
  const btn = document.getElementById('submit-btn');
  if (!btn) return;

  const modelReady =
    window.activeModel === 'vilotni_1' ||
    (window.activeModel === 'vilotni_2'
      ? window.vilotni2CoreState === 'ready'
      : window.vilotni15CoreState === 'ready');

  const hasText = String(val ?? '').trim().length > 0;

  if (hasText && !isProcessing && modelReady) {
    btn.style.background = 'var(--vilotni-gradient)';
    btn.style.color = 'white';
    btn.style.boxShadow = '0 0 0 1px var(--brand-primary)';
  } else {
    btn.style.background = '';
    btn.style.color = '';
    btn.style.boxShadow = '';
  }
};

window.initProfile = function() {
  try {
    const stored = JSON.parse(localStorage.getItem('vilotgri_profile') || 'null');
    if (stored && typeof stored === 'object') {
      userProfile = {
        ...userProfile,
        ...stored,
        name: typeof stored.name === 'string' && stored.name.trim() ? stored.name.trim().slice(0,20) : userProfile.name,
        avatarUrl: typeof stored.avatarUrl === 'string' && stored.avatarUrl ? stored.avatarUrl : null
      };
    }
  } catch (_) {}
  tempAvatarUrl = userProfile.avatarUrl;
  window.updateProfileUI();
};

window.syncAccountSettingsUI = function() {
  const input = document.getElementById('profile-name-input');
  const previewImg = document.getElementById('modal-avatar-img');
  const previewInitial = document.getElementById('modal-avatar-initial');
  const previewBg = document.getElementById('modal-avatar-preview');
  const photoRow = document.getElementById('account-photo-row');
  const namePreview = document.getElementById('settings-profile-name-preview');

  if (input) input.value = userProfile.name || 'User';
  if (namePreview) namePreview.textContent = userProfile.name || 'User';
  tempAvatarUrl = userProfile.avatarUrl;

  if (previewInitial) {
    previewInitial.textContent = (userProfile.name || 'User').charAt(0).toUpperCase();
    previewInitial.style.color = resolveProfileColor(userProfile.textColor, '#ffffff');
  }

  if (previewBg) {
    previewBg.style.backgroundColor = tempAvatarUrl
      ? 'transparent'
      : resolveProfileColor(userProfile.bgColor, 'var(--brand-primary)');
  }

  if (previewImg) {
    if (tempAvatarUrl) {
      previewImg.src = tempAvatarUrl;
      previewImg.classList.remove('hidden');
      previewInitial?.classList.add('hidden');
      photoRow?.classList.remove('hidden');
    } else {
      previewImg.src = '';
      previewImg.classList.add('hidden');
      previewInitial?.classList.remove('hidden');
      photoRow?.classList.add('hidden');
    }
  }

  document.querySelectorAll('#avatar-color-picker .color-btn').forEach(btn => {
    btn.classList.toggle('is-selected', btn.dataset.color === userProfile.bgColor);
  });
};

window.openProfileModal = function(event) {
  window.openSettingsPanel(event, 'account');
};

window.closeProfileModal = function() {
  // Account controls now live inside the compact Settings panel.
};

window.persistUserProfile = function(statusText = '') {
  try {
    localStorage.setItem('vilotgri_profile', JSON.stringify(userProfile));
  } catch (error) {
    console.warn('Unable to persist local profile:', error);
  }
  window.updateProfileUI?.();
  window.syncAccountSettingsUI?.();

  const status = document.getElementById('settings-account-save-status');
  if (status && statusText) {
    status.textContent = statusText;
    clearTimeout(window.__vilotniProfileStatusTimer);
    window.__vilotniProfileStatusTimer = setTimeout(() => {
      if (status.textContent === statusText) status.textContent = '';
    }, 1500);
  }
};

window.saveProfile = function() {
  const input = document.getElementById('profile-name-input');
  const nextName = input?.value?.trim();
  userProfile.name = nextName ? nextName.slice(0, 20) : 'User';
  userProfile.avatarUrl = tempAvatarUrl;
  window.persistUserProfile('Saved');
};

window.removeAvatar = function() {
  tempAvatarUrl = null;
  userProfile.avatarUrl = null;
  window.persistUserProfile('Photo removed');
};

window.setProfileAvatarColor = function(button) {
  if (!button) return;
  const color = button.dataset.color;
  const textColor = button.dataset.text;
  if (!color || !textColor) return;
  userProfile.bgColor = color;
  userProfile.textColor = textColor;
  window.persistUserProfile('Avatar updated');
};

window.handleAvatarUpload = function(event) {
  const input = event?.target;
  const file = input?.files?.[0];
  if (!file) return;
  if (!file.type?.startsWith('image/')) return;

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const MAX_DIM = 256;
      const scale = Math.min(1, MAX_DIM / Math.max(img.width, img.height));
      canvas.width = Math.max(1, Math.round(img.width * scale));
      canvas.height = Math.max(1, Math.round(img.height * scale));
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      tempAvatarUrl = canvas.toDataURL('image/jpeg', 0.88);
      userProfile.avatarUrl = tempAvatarUrl;
      window.persistUserProfile('Photo updated');
      if (input) input.value = '';
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
};

function getSettingsPalette() {
  const mode = appSettings.appearance.theme === 'light' ? 'light' : 'dark';
  return { mode, palette: appSettings.appearance.palettes[mode] };
}

let currentSettingsTab = 'account';

window.setSettingsTab = function(tab) {
  const allowed = ['account', 'chats', 'model', 'appearance'];
  if (!allowed.includes(tab)) return;
  currentSettingsTab = tab;

  document.querySelectorAll('[data-settings-tab]').forEach(btn => {
    const selected = btn.dataset.settingsTab === tab;
    btn.classList.toggle('is-selected', selected);
    btn.setAttribute('aria-selected', String(selected));
  });

  document.querySelectorAll('[data-settings-panel]').forEach(panel => {
    panel.classList.toggle('is-active', panel.dataset.settingsPanel === tab);
  });

  const scroll = document.querySelector('#settings-modal .settings-scroll');
  if (scroll) scroll.scrollTop = 0;
};

window.syncSettingsPanelUI = function() {
  const keepMemory = document.getElementById('setting-keep-memory-after-delete');
  const skipWarning = document.getElementById('setting-skip-delete-warning');
  const useModelAccent = document.getElementById('setting-use-model-accent');
  if (keepMemory) keepMemory.checked = appSettings.chats.retainContextAfterDelete;
  if (skipWarning) skipWarning.checked = appSettings.chats.skipDeleteWarning;
  if (useModelAccent) useModelAccent.checked = appSettings.appearance.useModelAccent;

  document.querySelectorAll('[data-settings-theme]').forEach(btn => {
    btn.classList.toggle('is-selected', btn.dataset.settingsTheme === appSettings.appearance.theme);
  });
  document.querySelectorAll('[data-settings-model]').forEach(btn => {
    btn.classList.toggle('is-selected', btn.dataset.settingsModel === appSettings.defaultModel);
  });

  const { palette } = getSettingsPalette();
  const colorMap = {
    'setting-color-bg': palette.bgBase,
    'setting-color-surface': palette.bgSurface,
    'setting-color-sidebar': palette.bgSidebar,
    'setting-color-elevated': palette.bgElevated,
    'setting-color-text': palette.textPrimary,
    'setting-color-accent-primary': appSettings.appearance.accentPrimary,
    'setting-color-accent-secondary': appSettings.appearance.accentSecondary
  };
  for (const [id, value] of Object.entries(colorMap)) {
    const input = document.getElementById(id);
    if (input) input.value = value;
  }

  document.querySelectorAll('.custom-accent-control').forEach(el => {
    el.classList.toggle('is-disabled', appSettings.appearance.useModelAccent);
    const input = el.querySelector('input[type="color"]');
    if (input) input.disabled = appSettings.appearance.useModelAccent;
  });

  const modeName = document.getElementById('settings-color-mode-name');
  if (modeName) modeName.textContent = appSettings.appearance.theme === 'light' ? 'Light palette' : 'Dark palette';
  window.syncAccountSettingsUI?.();
  window.setSettingsTab?.(currentSettingsTab);
};

window.openSettingsPanel = function(event, section = null) {
  if (event) event.stopPropagation();
  window.syncSettingsPanelUI();

  const modal = document.getElementById('settings-modal');
  if (!modal) return;

  const requestedTab =
    section === 'chats' ? 'chats' :
    section === 'appearance' ? 'appearance' :
    section === 'model' ? 'model' :
    section === 'account' ? 'account' :
    currentSettingsTab;

  window.setSettingsTab(requestedTab);

  modal.classList.remove('hidden');
  requestAnimationFrame(() => {
    modal.classList.add('is-open');
  });
};

window.closeSettingsPanel = function() {
  const modal = document.getElementById('settings-modal');
  if (!modal) return;
  modal.classList.remove('is-open');
  setTimeout(() => modal.classList.add('hidden'), 150);
};

window.setSettingsToggle = function(key, checked) {
  if (key === 'retainContextAfterDelete') appSettings.chats.retainContextAfterDelete = Boolean(checked);
  if (key === 'skipDeleteWarning') appSettings.chats.skipDeleteWarning = Boolean(checked);
  if (key === 'useModelAccent') appSettings.appearance.useModelAccent = Boolean(checked);
  persistAppSettings();
  applyAppAppearanceSettings();
  window.syncSettingsPanelUI();
};

window.setDefaultModelSetting = function(modelName) {
  if (!['vilotni_1', 'vilotni_1_5', 'vilotni_2'].includes(modelName)) return;
  appSettings.defaultModel = modelName;
  persistAppSettings();
  window.syncSettingsPanelUI();
};

window.setThemeSetting = function(theme) {
  if (theme !== 'dark' && theme !== 'light') return;
  appSettings.appearance.theme = theme;
  persistAppSettings();
  applyAppAppearanceSettings();
  window.syncSettingsPanelUI();
};

window.setAppearanceColor = function(key, value) {
  if (!/^#[0-9a-f]{6}$/i.test(value || '')) return;
  if (key === 'accentPrimary') appSettings.appearance.accentPrimary = value;
  else if (key === 'accentSecondary') appSettings.appearance.accentSecondary = value;
  else {
    const { palette } = getSettingsPalette();
    if (!['bgBase', 'bgSurface', 'bgSidebar', 'bgElevated', 'textPrimary'].includes(key)) return;
    palette[key] = value;
  }
  persistAppSettings();
  applyAppAppearanceSettings();
};

window.resetAppearanceSettings = function() {
  const defaults = cloneDefaultSettings().appearance;
  appSettings.appearance.palettes = defaults.palettes;
  appSettings.appearance.accentPrimary = defaults.accentPrimary;
  appSettings.appearance.accentSecondary = defaults.accentSecondary;
  appSettings.appearance.useModelAccent = defaults.useModelAccent;
  persistAppSettings();
  if (window.activeModel) window.switchActiveModel(window.activeModel);
  applyAppAppearanceSettings();
  window.syncSettingsPanelUI();
};

window.getVilotniMemoryStatus = function() {
  const agent = window.agentInstance;
  const store = agent?.memoryStore;

  return {
    ready: Boolean(
      window.vilotni15MemoryReady &&
      store &&
      agent?.memoryRetrieval
    ),
    persistent: Boolean(store?.persistent),
    working: store?.working?.length || 0,
    episodic: store?.episodes?.length || 0,
    semantic: store?.semantic?.length || 0,
    procedural: store?.procedural?.length || 0,
    phraseParserReady: typeof VilotTypePhraseParser === 'function',
    error: window.vilotni15MemoryError
      ? String(
          window.vilotni15MemoryError.message ||
          window.vilotni15MemoryError
        )
      : null
  };
};

document.addEventListener('DOMContentLoaded', async () => {
  window.initTheme();
  window.syncSidebarUI();
  window.initSavedChats();
  window.switchActiveModel(appSettings.defaultModel);
  window.syncSettingsPanelUI();
  CanvasSystem.init();
  initializeVilotni2Core().then(() => {
    window.refreshDockIndicators?.();
    if (window.activeModel === 'vilotni_2') window.switchActiveModel(window.activeModel);
  }).catch(() => {});
  window.initProfile();
  window.syncAccountSettingsUI?.();

  const sidebarViewportHandler = () => {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    if (window.matchMedia('(min-width: 769px)').matches) sidebar.classList.remove('open');
    window.syncSidebarUI();
  };
  window.addEventListener('resize', sidebarViewportHandler, { passive: true });

  document.addEventListener('keydown', event => {
    if (event.key !== 'Escape') return;
    if (!document.getElementById('delete-chat-modal')?.classList.contains('hidden')) window.closeDeleteChatModal();
    if (!document.getElementById('settings-modal')?.classList.contains('hidden')) window.closeSettingsPanel();
  });

  const input = document.getElementById('user-input');
  if (input) {
    input.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = this.scrollHeight + 'px';
      window.updateSubmitBtnState(this.value);
    });

    input.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        window.handleSend();
      }
    });
  }


  // Account controls use direct handlers so they remain independent of async engine initialization.

  // New chat uses an inline call to window.startNewChat so it remains available
  // even if a later asynchronous engine initializer fails.

  const gpuDot = document.getElementById('gpu-dot');
  const gpuText = document.getElementById('gpu-text');

  // V22 treats memory as part of the runtime rather than silently
  // disabling it when script order is wrong.
  try {
    assertVilotMemoryModules();
  } catch (memoryError) {
    console.error(
      'VilotNI 1.5 memory modules unavailable:',
      memoryError
    );

    window.vilotni15MemoryReady = false;
    window.vilotni15MemoryError = memoryError;

    if (gpuDot) {
      gpuDot.className =
        'w-2 h-2 rounded-full bg-rose-400 mr-2';
    }

    if (gpuText) {
      gpuText.textContent = 'Memory Error';
    }

    settleVilotni15Core(
      false,
      memoryError
    );

    return;
  }

  // Drop stale objects before replacing the vocabulary. Otherwise a fast
  // reload can pair an old agent with a brand-new, still-loading vocab.
  window.backendInstance = null;
  window.networkInstance = null;
  window.agentInstance = null;

  try {
    window.vocabInstance = new VocabularyManager();
    await window.vocabInstance.ready;
    const dynamicVocabSize = window.vocabInstance.vocabSize;

    if (!window.vocabInstance.loaded || !Number.isInteger(dynamicVocabSize) || dynamicVocabSize <= 0) {
      throw new Error(window.vocabInstance.loadError?.message || 'Vocabulary loaded with an invalid size.');
    }

    window.backendInstance = new WebGPUComputeBackend();
    await window.backendInstance.init();

    window.networkInstance =
      new VilotNI15Network(
        window.backendInstance,
        dynamicVocabSize
      );

    window.networkInstance.vocab =
      window.vocabInstance;

    // Prevent the agent from caching embeddings/head state before persisted
    // neural state has finished loading.
    if (
      window.networkInstance.ready &&
      typeof window.networkInstance.ready.then === 'function'
    ) {
      await window.networkInstance.ready;
    }

    window.agentInstance =
      await createAgent(
        window.networkInstance
      );

    if (gpuDot) gpuDot.className = 'w-2 h-2 rounded-full bg-emerald-400 mr-2';
    if (gpuText) gpuText.textContent = 'Active';

    settleVilotni15Core(true);
  } catch (err) {
    if (!window.vocabInstance || !window.vocabInstance.loaded || !Number.isInteger(window.vocabInstance.vocabSize) || window.vocabInstance.vocabSize <= 0) {
      console.error('VilotNI 1.5 vocabulary unavailable:', err);
      if (gpuDot) gpuDot.className = 'w-2 h-2 rounded-full bg-rose-400 mr-2';
      if (gpuText) gpuText.textContent = 'Vocabulary Error';

      settleVilotni15Core(false, err);
      return;
    }

    console.warn("VilotNI 1.5 WebGPU unavailable. Initializing CPU sparse mesh:", err);
    const fallbackVocabSize = window.vocabInstance.vocabSize;
    const poolDim = 256;
    const embDim = 64;

    const cpuLayers = [];
    for (let l = 0; l < 15; l++) {
      cpuLayers.push(new Sparse200kLayer(l, l === 0 ? embDim : 200000, 200000, 4));
    }

    const headWeights = new Float32Array(poolDim * fallbackVocabSize);
    const headGradients = new Float32Array(poolDim * fallbackVocabSize);
    const headScale = 1.65 * Math.sqrt(2.0 / poolDim);
    for (let i = 0; i < headWeights.length; i++) {
      headWeights[i] = (Math.random() * 2 - 1) * headScale;
    }

    const embeddings = new Float32Array(fallbackVocabSize * embDim);
    for (let i = 0; i < embeddings.length; i++) {
      embeddings[i] = (Math.random() * 2 - 1) * 0.25;
    }

    window.networkInstance = {
      vocabSize: fallbackVocabSize,
      vocab: window.vocabInstance,
      poolDim,
      embDim,
      layers: cpuLayers,
      headWeights,
      headGradients,
      embeddings,
      currentLatent: new Float32Array(poolDim),
      traceFresh: false,
      rewardHeadTokens: new Set(),

      async forwardSequence(tokenIds, options = null) {
        const captureTrace =
          options === true ||
          (options && options.captureTrace === true);
        const seqLen = tokenIds.length;
        const initialInput = new Float32Array(200000);

        for (let pos = 0; pos < seqLen; pos++) {
          const tokenId = tokenIds[pos];
          const offset = (tokenId % this.vocabSize) * this.embDim;
          const recency = 0.85 + 0.15 * (pos / Math.max(1, seqLen));

          for (let d = 0; d < this.embDim; d++) {
            const denom = Math.pow(10000, (2 * Math.floor(d / 2)) / this.embDim);
            const pe = d % 2 === 0 ? Math.sin(pos / denom) : Math.cos(pos / denom);
            initialInput[d] += ((this.embeddings[offset + d] + pe) * recency) / Math.sqrt(seqLen);
          }
        }

        let prev = initialInput;

        for (let l = 0; l < 15; l++) {
          const layer = this.layers[l];

          if (captureTrace) {
            if (typeof layer.clearGradients === 'function') {
              layer.clearGradients();
            } else if (layer.weightGradients) {
              layer.weightGradients.fill(0);
            }
          }

          layer.forward(
            prev,
            l === 7 ? initialInput : null,
            l === 7 ? 0.18 : 0.0
          );

          if (
            captureTrace &&
            typeof layer.accumulateGradient === 'function'
          ) {
            // Mirrors the WebGPU eligibility trace:
            // grad += outputActivation * previousActivation.
            layer.accumulateGradient(
              layer.activations,
              prev
            );
          }

          prev = layer.activations;
        }

        for (let p = 0; p < poolDim; p++) {
          let maxAct = -1e9;
          let sumAct = 0;

          // True proportional coverage of all 200,000 neurons.
          const start = Math.floor((p * 200000) / poolDim);
          const end = Math.floor(((p + 1) * 200000) / poolDim);

          for (let i = start; i < end; i++) {
            const value = prev[i];
            if (value > maxAct) maxAct = value;
            sumAct += value;
          }
          this.currentLatent[p] = Math.tanh(0.75 * maxAct + 0.25 * (sumAct / (end - start)));
        }

        const logits = new Float32Array(this.vocabSize);
        for (let v = 0; v < this.vocabSize; v++) {
          let sum = 0;
          const offset = v * poolDim;
          for (let p = 0; p < poolDim; p++) {
            sum += this.headWeights[offset + p] * this.currentLatent[p];
          }
          logits[v] = sum * 1.45;
        }

        this.traceFresh = captureTrace;
        return logits;
      },

      async captureRewardTrace(tokenIds, rewardedTokens = null) {
        const logits = await this.forwardSequence(
          tokenIds,
          { captureTrace: true }
        );

        for (const tokenId of this.rewardHeadTokens) {
          const off = tokenId * this.poolDim;
          this.headGradients.fill(
            0,
            off,
            off + this.poolDim
          );
        }
        this.rewardHeadTokens.clear();

        const sourceTokens =
          rewardedTokens && rewardedTokens.length
            ? rewardedTokens
            : [tokenIds[tokenIds.length - 1]];

        const unique = [];
        const seen = new Set();

        for (const tokenId of sourceTokens) {
          if (
            !Number.isFinite(tokenId) ||
            tokenId < 0 ||
            tokenId >= this.vocabSize ||
            seen.has(tokenId)
          ) {
            continue;
          }

          seen.add(tokenId);
          unique.push(tokenId);
        }

        const scale =
          1.0 / Math.sqrt(Math.max(1, unique.length));

        for (const tokenId of unique) {
          const off = tokenId * this.poolDim;

          for (let p = 0; p < this.poolDim; p++) {
            this.headGradients[off + p] =
              this.currentLatent[p] * scale;
          }

          this.rewardHeadTokens.add(tokenId);
        }

        return logits;
      },

      applyPolicyReward(reward, lr = 0.008, updateDeep = true) {
        for (const tokenId of this.rewardHeadTokens) {
          const off = tokenId * this.poolDim;

          for (let p = 0; p < this.poolDim; p++) {
            const idx = off + p;
            this.headWeights[idx] +=
              lr * reward * this.headGradients[idx];
            this.headWeights[idx] *= 0.9999;
            this.headGradients[idx] = 0;
          }
        }

        if (updateDeep && this.traceFresh) {
          for (const layer of this.layers) {
            layer.applyReward(reward, lr);
          }
        }

        this.traceFresh = false;
        this.rewardHeadTokens.clear();
      }
    };
    try {
      window.agentInstance =
        await createAgent(
          window.networkInstance
        );
    } catch (memoryError) {
      console.error(
        'VilotNI 1.5 CPU fallback could not initialize memory:',
        memoryError
      );

      window.vilotni15MemoryReady = false;
      window.vilotni15MemoryError = memoryError;

      if (gpuDot) {
        gpuDot.className =
          'w-2 h-2 rounded-full bg-rose-400 mr-2';
      }

      if (gpuText) {
        gpuText.textContent = 'Memory Error';
      }

      settleVilotni15Core(
        false,
        memoryError
      );

      return;
    }

    if (gpuDot) gpuDot.className = 'w-2 h-2 rounded-full bg-amber-400 mr-2';
    if (gpuText) gpuText.textContent = 'Active (CPU Sparse)';

    settleVilotni15Core(true);
  }

  window.switchActiveModel('vilotni_1_5');
});