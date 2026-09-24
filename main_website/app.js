/**
 * VilotNI Unified Controller
 * Features:
 * - Adaptive Time Thought Tag: Renders sub-second runs cleanly (e.g. 4ms, 38ms) and full seconds as X.XXs.
 * - Zero Artificial Stalls: Real-time execution with immediate DOM text commit.
 * - Dual-Engine Routing: VilotNI 1 (Cognitive NLP) & VilotNI 1.5 (3M WebGPU Network).
 * - Real-Time Dock & Carousel State: Dynamically tracks RL (Low), RSL (Medium), and Deep RSL (High).
 * - Anytime Inference Integration: Directly supports fast-exit rollouts from RLEngine.
 * - Exact-Message Reward Routing: Feedback trains the prompt/response that was actually rated.
 * - WebGPU/CPU Reward-Trace Parity: Both paths isolate policy traces and exact response head rows.
 */

window.activeModel = 'vilotni_1_5'; // 'vilotni_1' | 'vilotni_1_5'
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

// Temporary calibration instrumentation. Set this false when the confidence
// bottleneck has been identified.
window.showConfidenceDiagnostics = false;

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

window.switchActiveModel = function(modelName) {
  window.activeModel = modelName;

  const item1 = document.getElementById('model-item-1');
  const item15 = document.getElementById('model-item-1-5');
  const check1 = document.getElementById('model-check-1');
  const check15 = document.getElementById('model-check-1-5');
  const learningCard = document.getElementById('learning-mode-card');
  const triggerName = document.getElementById('trigger-model-name');
  const triggerStatus = document.getElementById('trigger-model-status');
  const headerLabel = document.getElementById('header-model-label');
  const dockIndicator = document.getElementById('dock-model-indicator');
  const sessionLabel = document.getElementById('active-session-label');
  const heroTitle = document.getElementById('empty-hero-title');
  const heroDesc = document.getElementById('empty-hero-desc');
  const suggestionsGrid = document.getElementById('suggestions-grid');

  if (modelName === 'vilotni_1') {
    if (item1) item1.className = "w-full px-2.5 py-2 rounded-xl transition-all flex items-center justify-between cursor-pointer border border-[var(--border-highlight)] bg-[var(--bg-elevated)] text-[var(--text-primary)]";
    if (item15) item15.className = "w-full px-2.5 py-2 rounded-xl transition-all flex items-center justify-between cursor-pointer border border-transparent hover:bg-[var(--bg-elevated)]/50 text-[var(--text-secondary)]";
    if (check1) check1.classList.remove('hidden');
    if (check15) check15.classList.add('hidden');

    if (learningCard) {
      learningCard.style.opacity = '0';
      learningCard.style.transform = 'translateY(10px) scale(0.95)';
      learningCard.style.maxHeight = '0px';
      learningCard.style.paddingTop = '0px';
      learningCard.style.paddingBottom = '0px';
      learningCard.style.marginBottom = '0px';
      learningCard.style.overflow = 'hidden';
      learningCard.style.pointerEvents = 'none';
    }

    if (triggerName) triggerName.textContent = "VilotNI 1";
    if (triggerStatus) triggerStatus.classList.add('hidden');
    if (headerLabel) headerLabel.textContent = "VilotNI 1";
    if (dockIndicator) dockIndicator.textContent = "VilotNI 1";
    if (sessionLabel) sessionLabel.textContent = "VilotNI 1 Session";

    window.refreshDockIndicators();

    if (heroTitle) heroTitle.textContent = "What are we building?";
    if (heroDesc) heroDesc.textContent = "Describe your UI logic. VilotNI 1 will synthesize the structural concepts and generate semantic architecture.";

    if (suggestionsGrid) {
      suggestionsGrid.innerHTML = `
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('Synthesize a dark mode navbar directly onto a user profile card layout')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] group-hover:text-[var(--brand-primary)] transition-colors">
          Logic Mixing
        </span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">
          Synthesize a dark mode navbar directly onto a user profile card layout.
        </span>
      </button>
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('Explain heuristic evaluation')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] group-hover:text-[var(--brand-secondary)] transition-colors">
          Test Cognitive Engine
        </span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">
          Ask the AI about its heuristic evaluation and internal logic structures.
        </span>
      </button>
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('Generate a clean authentication flow with Login and Sign Up forms')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] group-hover:text-emerald-400 transition-colors">
          Auth Forms
        </span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">
          Construct a split-layout authentication flow for secure access.
        </span>
      </button>
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('Build a bento box layout')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] group-hover:text-rose-400 transition-colors">
          Modern Grids
        </span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">
          Deploy a responsive, multi-span CSS Grid component commonly used in dashboards.
        </span>
      </button>
      `;
    }
  } else {
    if (item15) item15.className = "w-full px-2.5 py-2 rounded-xl transition-all flex items-center justify-between cursor-pointer border border-[var(--border-highlight)] bg-[var(--bg-elevated)] text-[var(--text-primary)]";
    if (item1) item1.className = "w-full px-2.5 py-2 rounded-xl transition-all flex items-center justify-between cursor-pointer border border-transparent hover:bg-[var(--bg-elevated)]/50 text-[var(--text-secondary)]";
    if (check15) check15.classList.remove('hidden');
    if (check1) check1.classList.add('hidden');

    if (learningCard) {
      learningCard.style.maxHeight = '300px';
      learningCard.style.paddingTop = '';
      learningCard.style.paddingBottom = '';
      learningCard.style.marginBottom = '';
      learningCard.style.overflow = 'visible';
      learningCard.style.pointerEvents = 'auto';
      void learningCard.offsetWidth;
      learningCard.style.opacity = '1';
      learningCard.style.transform = 'translateY(0) scale(1)';
    }

    if (triggerName) triggerName.textContent = "VilotNI 1.5";
    if (triggerStatus) triggerStatus.classList.remove('hidden');
    if (headerLabel) headerLabel.textContent = "VilotNI 1.5";
    if (dockIndicator) dockIndicator.textContent = "VilotNI 1.5";
    if (sessionLabel) sessionLabel.textContent = "VilotNI 1.5 Session";

    if (heroTitle) heroTitle.textContent = "VilotNI 1.5";
    if (heroDesc) heroDesc.textContent = "A substantial advancement over VilotNI 1, featuring a 3-million-neuron activation layer, adaptive learning, and no hardcoded responses.";

    if (suggestionsGrid) {
      suggestionsGrid.innerHTML = `
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('Does a Neural Inspector calculate compute speed?')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] flex items-center group-hover:text-[var(--brand-primary)] transition-colors">
          Compute Performance
        </span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">
          Inquire about processing power and accelerator speed.
        </span>
      </button>
      <button type="button" class="suggestion-card p-4 rounded-xl text-left flex flex-col space-y-1.5 cursor-pointer group" onclick="window.fillInput('How will AI model chips accelerate?')">
        <span class="font-display font-semibold text-sm text-[var(--text-primary)] flex items-center group-hover:text-[var(--brand-secondary)] transition-colors">
          Modern Processors
        </span>
        <span class="text-xs text-[var(--text-secondary)] leading-relaxed font-light">
          Discuss future hardware models and specialized chips.
        </span>
      </button>
      `;
    }
    
    window.refreshActiveModeStyling();
  }
};

window.refreshDockIndicators = function() {
  const dockModel = document.getElementById('dock-model-indicator');
  if (!dockModel) return;

  let learningIndicator = document.getElementById('dock-learning-indicator');
  let depthIndicator = document.getElementById('dock-depth-indicator');

  if (!learningIndicator) {
    learningIndicator = document.createElement('span');
    learningIndicator.id = 'dock-learning-indicator';
    learningIndicator.className = 'ml-2 text-[10px] font-mono font-bold';
    dockModel.insertAdjacentElement('afterend', learningIndicator);
  }
  if (!depthIndicator) {
    depthIndicator = document.createElement('span');
    depthIndicator.id = 'dock-depth-indicator';
    depthIndicator.className = 'ml-1 text-[10px] font-mono text-[var(--text-tertiary)]';
    learningIndicator.insertAdjacentElement('afterend', depthIndicator);
  }

  const isModel1 = window.activeModel === 'vilotni_1';
  const agent = window.agentInstance;
  const isRsl = agent?.mode === 'rsl';
  const isDeep = isRsl && Boolean(agent?.isDeepLearning);

  if (isModel1) {
    learningIndicator.textContent = 'Cognitive';
    learningIndicator.className = 'ml-2 text-[10px] font-mono font-bold text-indigo-300';
    depthIndicator.textContent = '';
  } else {
    learningIndicator.textContent = isRsl ? (isDeep ? 'Deep RSL' : 'RSL') : 'RL';
    learningIndicator.className = `ml-2 text-[10px] font-mono font-bold ${isDeep ? 'text-orange-400' : (isRsl ? 'text-pink-400' : 'text-cyan-400')}`;
    depthIndicator.textContent = isRsl ? (isDeep ? '• Adaptive Search' : '• Fast Search') : '• 256 tokens';
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
      if (depthSubtext) depthSubtext.textContent = "Fast-exits immediately upon high confidence";

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

function appendUserMessage(text) {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'flex w-full justify-end animate-fade-in';
  const bubble = document.createElement('div');
  bubble.className = 'msg-user max-w-[85%] rounded-2xl px-5 py-3.5 text-[15px] leading-relaxed shadow-sm';
  bubble.innerText = text;
  wrapper.appendChild(bubble);
  chatHistory.appendChild(wrapper);
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
  if (!d || !window.showConfidenceDiagnostics) return "";

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
    `Budget ${d.generationBudget || 0}`;

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

async function appendAIMessage(
  generatedText,
  confidencePct = null,
  codeBlock = null,
  elapsedTime = null,
  trainingState = null,
  confidenceDiagnostics = null
) {
  const chatHistory = document.getElementById('chat-history');
  if (!chatHistory) return;

  const isModel1 = window.activeModel === 'vilotni_1';
  const isRsl = !window.agentInstance || window.agentInstance.mode === 'rsl';
  const isDeep = window.agentInstance && window.agentInstance.isDeepLearning;

  const avatarGradient = isModel1
    ? "from-blue-600 to-indigo-700"
    : (!isRsl 
        ? "from-cyan-500 to-blue-600" 
        : (isDeep ? "from-amber-500 to-red-500" : "from-purple-500 to-pink-500"));

  const badgeClasses = isModel1
    ? "border-indigo-500/30 bg-indigo-500/10 text-indigo-300"
    : (!isRsl
        ? "border-cyan-500/30 bg-cyan-500/10 text-cyan-300"
        : (isDeep ? "border-orange-500/30 bg-orange-500/10 text-orange-300" : "border-purple-500/30 bg-purple-500/10 text-purple-300"));

  const modelLabel = isModel1 ? "VilotNI 1" : "VilotNI 1.5";

  const msgId = 'msg-' + Date.now();

  if (!isModel1 && trainingState) {
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

  wrapper.innerHTML = `
    <div class="flex space-x-4 max-w-full w-full">
      <div class="w-8 h-8 rounded-xl bg-gradient-to-br ${avatarGradient} flex items-center justify-center shrink-0 mt-1 shadow-lg ring-1 ring-white/10">
        <i class="ph ph-lightning text-white text-base"></i>
      </div>
      <div class="w-full overflow-hidden">
        <div class="flex items-center space-x-2.5 mb-2 flex-wrap gap-y-1">
          <span class="text-[14px] font-display font-bold text-[var(--text-primary)]">${modelLabel}</span>
          ${elapsedTime ? `
          <span class="text-[10px] font-mono px-2 py-0.5 rounded-full border border-[var(--border-subtle)] bg-[var(--bg-base)] text-[var(--text-secondary)] font-medium flex items-center space-x-1 shadow-sm" title="Execution Time">
            <i class="ph ph-timer text-[11px] text-[var(--text-tertiary)]"></i>
            <span>${elapsedTime}</span>
          </span>` : ''}
          ${confidencePct !== null ? `
          <span class="text-[10px] font-mono px-2 py-0.5 rounded-full border ${badgeClasses} font-semibold shadow-sm">
            Confidence: ${confidencePct}%
          </span>` : `
          <span class="text-[10px] font-mono px-2 py-0.5 rounded-full border ${badgeClasses} font-semibold shadow-sm">
            Cognitive NLP
          </span>`}
        </div>
        ${!isModel1 ? formatConfidenceDiagnostics(confidenceDiagnostics) : ''}
        <div id="text-${msgId}" class="leading-[1.7] text-[15px] space-y-4 text-[var(--text-primary)] font-normal"></div>
        ${codeBlock ? `
        <div class="mt-3 bg-[var(--bg-base)] border border-[var(--border-subtle)] rounded-xl p-4 overflow-x-auto text-xs font-mono text-zinc-300">
          <pre><code>${codeBlock.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</code></pre>
        </div>` : ''}
        
        <div id="action-${msgId}" class="mt-4 flex items-center space-x-2 opacity-0 group-hover/msg:opacity-100 transition-opacity duration-300">
          ${!isModel1 ? `
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
  `;

  chatHistory.appendChild(wrapper);

  const textContainer = document.getElementById(`text-${msgId}`);
  if (textContainer) {
    textContainer.innerText = generatedText || "";
  }
  scrollToBottom();
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

window.toggleDesktopSidebar = function() {
  const sidebar = document.getElementById('sidebar');
  const toggleBtn = document.getElementById('sidebar-toggle-btn');
  if (!sidebar) return;
  sidebar.classList.toggle('collapsed');
  if (toggleBtn) {
    toggleBtn.style.display = sidebar.classList.contains('collapsed') ? 'block' : 'none';
  }
};

window.toggleMobileSidebar = function() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  if (!sidebar || !overlay) return;
  const isOpen = sidebar.classList.contains('open');
  if (isOpen) {
    sidebar.classList.remove('open');
    overlay.classList.add('hidden');
  } else {
    sidebar.classList.add('open');
    overlay.classList.remove('hidden');
  }
};

window.toggleTheme = function(e) {
  if (e) e.stopPropagation();
  document.body.classList.toggle('light-mode');
};

window.fillInput = function(text) {
  const input = document.getElementById('user-input');
  if (!input) return;
  input.value = text;
  input.style.height = 'auto';
  input.style.height = input.scrollHeight + 'px';
  window.updateSubmitBtnState(text);
  input.focus();
};

window.updateSubmitBtnState = function(val) {
  const btn = document.getElementById('submit-btn');
  if (!btn) return;

  const modelReady =
    window.activeModel === 'vilotni_1' ||
    window.vilotni15CoreState === "ready";

  if (val.trim().length > 0 && !isProcessing && modelReady) {
    btn.style.background = 'var(--vilotni-gradient)';
    btn.style.color = 'white';
    btn.style.boxShadow = '0 4px 15px rgba(99, 102, 241, 0.4)';
  } else {
    btn.style.background = '';
    btn.style.color = '';
    btn.style.boxShadow = '';
  }
};

window.initProfile = function() {
  const saved = localStorage.getItem('vilotgri_profile');
  if (saved) {
    try {
      userProfile = { ...userProfile, ...JSON.parse(saved) };
    } catch(e) {}
  }
  window.updateProfileUI();
};

window.updateProfileUI = function() {
  const nameEl = document.getElementById('sidebar-user-name');
  const bgEl = document.getElementById('sidebar-avatar-bg');

  if (nameEl) nameEl.innerText = userProfile.name;
  if (bgEl) {
    bgEl.innerHTML = '';
    if (userProfile.avatarUrl) {
      const img = document.createElement('img');
      img.src = userProfile.avatarUrl;
      img.className = 'w-full h-full object-cover';
      bgEl.appendChild(img);
      bgEl.style.backgroundColor = 'transparent';
    } else {
      const span = document.createElement('span');
      span.className = 'text-xs font-bold';
      span.innerText = userProfile.name.charAt(0).toUpperCase();
      span.style.color = userProfile.textColor.startsWith('var') 
        ? `var(${userProfile.textColor.match(/var\((.*?)\)/)[1]})` 
        : userProfile.textColor;
      bgEl.appendChild(span);
      bgEl.style.backgroundColor = userProfile.bgColor.startsWith('var') 
        ? `var(${userProfile.bgColor.match(/var\((.*?)\)/)[1]})` 
        : userProfile.bgColor;
    }
  }
};

window.openProfileModal = function() {
  const modal = document.getElementById('profile-modal');
  const content = document.getElementById('profile-modal-content');
  const input = document.getElementById('profile-name-input');
  const previewImg = document.getElementById('modal-avatar-img');
  const previewInitial = document.getElementById('modal-avatar-initial');
  const removeBtn = document.getElementById('remove-avatar-btn');
  const previewBg = document.getElementById('modal-avatar-preview');

  if (!modal || !content) return;
  if (input) input.value = userProfile.name;
  tempAvatarUrl = userProfile.avatarUrl;

  if (tempAvatarUrl && previewImg && previewInitial && removeBtn && previewBg) {
    previewImg.src = tempAvatarUrl;
    previewImg.classList.remove('hidden');
    previewInitial.classList.add('hidden');
    removeBtn.classList.remove('hidden');
    previewBg.style.backgroundColor = 'transparent';
  } else if (previewImg && previewInitial && removeBtn && previewBg) {
    previewImg.src = '';
    previewImg.classList.add('hidden');
    previewInitial.classList.remove('hidden');
    previewInitial.innerText = userProfile.name.charAt(0).toUpperCase();
    removeBtn.classList.add('hidden');
    previewBg.style.backgroundColor = userProfile.bgColor.startsWith('var') 
      ? `var(${userProfile.bgColor.match(/var\((.*?)\)/)[1]})` 
      : userProfile.bgColor;
    previewInitial.style.color = userProfile.textColor.startsWith('var') 
      ? `var(${userProfile.textColor.match(/var\((.*?)\)/)[1]})` 
      : userProfile.textColor;
  }

  document.querySelectorAll('.color-btn').forEach(btn => {
    if (btn.dataset.color === userProfile.bgColor) {
      btn.classList.replace('ring-transparent', 'ring-[var(--brand-primary)]');
    } else {
      btn.classList.replace('ring-[var(--brand-primary)]', 'ring-transparent');
    }
  });

  modal.classList.remove('hidden');
  modal.classList.add('flex');
  void modal.offsetWidth;
  modal.classList.add('opacity-100');
  content.classList.replace('scale-95', 'scale-100');
};

window.closeProfileModal = function() {
  const modal = document.getElementById('profile-modal');
  const content = document.getElementById('profile-modal-content');
  if (!modal || !content) return;
  modal.classList.remove('opacity-100');
  content.classList.replace('scale-100', 'scale-95');
  setTimeout(() => {
    modal.classList.remove('flex');
    modal.classList.add('hidden');
  }, 250);
};

window.saveProfile = function() {
  const input = document.getElementById('profile-name-input');
  userProfile.name = (input && input.value.trim()) ? input.value.trim() : 'User';
  userProfile.avatarUrl = tempAvatarUrl;

  localStorage.setItem('vilotgri_profile', JSON.stringify(userProfile));
  window.updateProfileUI();
  window.closeProfileModal();
};

window.removeAvatar = function() {
  tempAvatarUrl = null;
  const previewImg = document.getElementById('modal-avatar-img');
  const previewInitial = document.getElementById('modal-avatar-initial');
  const removeBtn = document.getElementById('remove-avatar-btn');
  const previewBg = document.getElementById('modal-avatar-preview');

  if (previewImg) previewImg.classList.add('hidden');
  if (previewInitial) previewInitial.classList.remove('hidden');
  if (removeBtn) removeBtn.classList.add('hidden');
  if (previewBg) {
    previewBg.style.backgroundColor = userProfile.bgColor.startsWith('var') 
      ? `var(${userProfile.bgColor.match(/var\((.*?)\)/)[1]})` 
      : userProfile.bgColor;
  }
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
  CanvasSystem.init();
  window.initProfile();

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

  const fileInput = document.getElementById('avatar-upload-input');
  if (fileInput) {
    fileInput.addEventListener('change', function(e) {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function(event) {
        const img = new Image();
        img.onload = function() {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          const MAX_DIM = 160;
          let width = img.width;
          let height = img.height;

          if (width > height) {
            if (width > MAX_DIM) {
              height *= MAX_DIM / width;
              width = MAX_DIM;
            }
          } else {
            if (height > MAX_DIM) {
              width *= MAX_DIM / height;
              height = MAX_DIM;
            }
          }
          canvas.width = width;
          canvas.height = height;
          ctx.drawImage(img, 0, 0, width, height);

          tempAvatarUrl = canvas.toDataURL('image/jpeg', 0.85);

          const previewImg = document.getElementById('modal-avatar-img');
          const previewInitial = document.getElementById('modal-avatar-initial');
          const removeBtn = document.getElementById('remove-avatar-btn');
          const previewBg = document.getElementById('modal-avatar-preview');

          if (previewImg) { previewImg.src = tempAvatarUrl; previewImg.classList.remove('hidden'); }
          if (previewInitial) previewInitial.classList.add('hidden');
          if (removeBtn) removeBtn.classList.remove('hidden');
          if (previewBg) previewBg.style.backgroundColor = 'transparent';
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    });
  }

  document.querySelectorAll('.color-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.color-btn').forEach(b => b.classList.replace('ring-[var(--brand-primary)]', 'ring-transparent'));
      e.target.classList.replace('ring-transparent', 'ring-[var(--brand-primary)]');
      userProfile.bgColor = e.target.dataset.color;
      userProfile.textColor = e.target.dataset.text;

      if (!tempAvatarUrl) {
        const previewBg = document.getElementById('modal-avatar-preview');
        const previewInitial = document.getElementById('modal-avatar-initial');
        if (previewBg) previewBg.style.backgroundColor = userProfile.bgColor.startsWith('var') ? `var(${userProfile.bgColor.match(/var\((.*?)\)/)[1]})` : userProfile.bgColor;
        if (previewInitial) previewInitial.style.color = userProfile.textColor.startsWith('var') ? `var(${userProfile.textColor.match(/var\((.*?)\)/)[1]})` : userProfile.textColor;
      }
    });
  });

  const btnNewChat = document.getElementById('btn-new-chat');
  if (btnNewChat) {
    btnNewChat.addEventListener('click', () => {
      const chatHistory = document.getElementById('chat-history');
      const emptyState = document.getElementById('empty-state');
      const userInputEl = document.getElementById('user-input');

      if (chatHistory) {
        chatHistory.innerHTML = '';
        chatHistory.classList.add('hidden');
      }

      messageTrainingState.clear();
      resetVilotniConversationState();

      if (emptyState) emptyState.style.display = 'flex';
      if (userInputEl) {
        userInputEl.value = '';
        userInputEl.style.height = 'auto';
        window.updateSubmitBtnState('');
        userInputEl.focus();
      }
    });
  }

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