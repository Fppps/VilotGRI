/**
 * VilotNI 2 - Production cognitive worker.
 *
 * One unified six-million-neuron model. Words.json supplies lexical identities;
 * Lexicon/WordState/PhraseState/WordMatrix provide learned lexical behavior;
 * RSL resolves contextual POS and recurrent state; Correlation.js now combines
 * structural, lexical, correlation and curiosity learning between prompts. TrainingInfo.json remains a
 * reference asset and is not used to manufacture live answers.
 */
'use strict';

importScripts('./RuleStore.js?v=2-rule-data-1', './Arena.js?v=2-rule-data-1', './Trace.js?v=2-rule-data-1', './Compute.js?v=2-rule-data-1', './Storage.js?v=2-rule-data-1', './RSL.js?v=2-rule-data-1', './Processor.js?v=2-rule-data-1', './Lexicon.js?v=2-rule-data-1', './WordState.js?v=2-rule-data-1', './PhraseState.js?v=2-rule-data-1', './WordMatrix.js?v=2-rule-data-1', './QuestionState.js?v=2-rule-data-1', './ClauseState.js?v=2-rule-data-1', './MorphologyState.js?v=2-rule-data-1', './KnowledgeState.js?v=2-rule-data-1', './LanguageState.js?v=2-rule-data-1', './Memory.js?v=2-rule-data-1', './Correlation.js?v=2-rule-data-1', './MatrixEngine.js?v=2-rule-data-1', './QueryPlanner.js?v=2-rule-data-1', './EvidenceSearch.js?v=2-rule-data-1', './CandidateRanker.js?v=2-rule-data-1', './Decoder.js?v=2-rule-data-1', './FastLane.js?v=2-rule-data-1');

const Phase = Object.freeze({
  BOOTING: 'BOOTING',
  ACTIVE: 'ACTIVE',
  WARM: 'WARM',
  IDLE: 'IDLE',
  SUSPENDED: 'SUSPENDED'
});

const runtime = {
  ready: false,
  phase: Phase.BOOTING,
  visible: true,
  arena: null,
  trace: null,
  compute: null,
  rsl: null,
  processor: null,
  lexicon: null,
  wordState: null,
  phraseState: null,
  wordMatrix: null,
  languageState: null,
  decoder: null,
  storage: null,
  memory: null,
  correlation: null,
  matrix: null,
  fastLane: null,
  queryPlanner: null,
  evidenceSearch: null,
  candidateRanker: null,
  ruleStore: null,
  dialogueContext: [],
  dialogueContextCapacity: 16,
  learningQueue: [],
  learningProcessed: 0,
  learningDropped: 0,
  tickCount: 0,
  foregroundEpoch: 0,
  lastForegroundAt: performance.now(),
  timer: null,
  foregroundBusy: false,
  lastHomeostasisAt: 0,
  options: {
    stateDim: 6000000,
    activeTickMs: 24,
    warmTickMs: 100,
    idleTickMs: 400,
    warmAfterMs: 1400,
    idleAfterMs: 12000,
    activeHomeostasisMs: 240,
    warmHomeostasisMs: 520,
    idleHomeostasisMs: 900,
    suspendWhenHidden: false,
    minResponseConfidence: 80,
    latencyTargetMs: 40,
    maxForegroundSearchMs: 120, // soft threshold: enters recovery search, never lowers/terminates the confidence gate
    learningQueueMax: 512,
    fastInitialBatch: 2,
    fastRefineBatch: 4,
    fastSearchBatch: 4,
    fastFocusedBatch: 6,
    fastFocusedBatchMin: 4,
    fastFocusedBatchMax: 8,
    fastFocusedEvidenceLimit: 128,
    fastFocusedDirectLimit: 72,
    fastFocusedRSLLimit: 28,
    fastYieldEveryBatches: 6,
    wordMatrixDim: 64,
    phraseMaxN: 4,
    phraseMaxEntries: 32768,
    curiosityEnabled: true,
    curiosityWeight: 0.14,
    curiosityMaxConfidenceDrop: 3.25,
    curiosityQueueMax: 256
  },
  lastHotPathMs: 0,
  localBenchmark: null
};

function send(type, payload = {}, requestId = null) {
  postMessage({ type, requestId, ...payload });
}

function setPhase(next, reason = '') {
  if (runtime.phase === next) return;
  runtime.phase = next;
  runtime.trace?.push(
    VilotTraceEvent.PHASE_CHANGE,
    0,
    next === Phase.ACTIVE ? 1 : next === Phase.WARM ? 2 : next === Phase.IDLE ? 3 : next === Phase.SUSPENDED ? 4 : 0
  );
  send('PHASE', { phase: next, reason });
}

function currentInterval() {
  switch (runtime.phase) {
    case Phase.ACTIVE: return runtime.options.activeTickMs;
    case Phase.WARM: return runtime.options.warmTickMs;
    case Phase.IDLE: return runtime.options.idleTickMs;
    default: return 1000;
  }
}

function scheduleBackground() {
  clearTimeout(runtime.timer);
  if (!runtime.ready || runtime.phase === Phase.SUSPENDED) return;
  runtime.timer = setTimeout(backgroundTick, currentInterval());
}

function chooseAutomaticPhase(now) {
  if (runtime.phase === Phase.SUSPENDED) return;
  if (!runtime.visible && runtime.options.suspendWhenHidden) {
    setPhase(Phase.SUSPENDED, 'document-hidden');
    return;
  }

  const age = now - runtime.lastForegroundAt;
  if (age < runtime.options.warmAfterMs) setPhase(Phase.ACTIVE, 'recent-foreground');
  else if (age < runtime.options.idleAfterMs) setPhase(Phase.WARM, 'cooling');
  else setPhase(Phase.IDLE, 'idle');
}

function homeostasisInterval() {
  if (runtime.phase === Phase.ACTIVE) return Math.max(80, runtime.options.activeHomeostasisMs | 0 || 240);
  if (runtime.phase === Phase.WARM) return Math.max(120, runtime.options.warmHomeostasisMs | 0 || 520);
  return Math.max(200, runtime.options.idleHomeostasisMs | 0 || 900);
}


function exportLearningBundle() {
  return {
    schemaVersion: 11,
    architecture: 'vilotni2-production-general-language-curiosity',
    rsl: runtime.rsl?.exportLearningState?.() || null,
    wordState: runtime.wordState?.exportState?.() || null,
    phraseState: runtime.phraseState?.exportState?.() || null,
    wordMatrix: runtime.wordMatrix?.exportState?.() || null,
    languageState: runtime.languageState?.exportState?.() || null,
    correlationState: runtime.correlation?.exportState?.() || null
  };
}

function pushDialogueTurn(role, text, metadata = {}) {
  const clean = String(text || '').trim();
  if (!clean) return null;
  const turn = {
    role: String(role || 'unknown').toLowerCase(),
    text: clean.slice(0, 4096),
    at: Date.now(),
    confidence: Number.isFinite(metadata.confidence) ? Number(metadata.confidence) : null
  };
  runtime.dialogueContext.push(turn);
  const cap = Math.max(4, runtime.dialogueContextCapacity | 0 || 16);
  if (runtime.dialogueContext.length > cap) runtime.dialogueContext.splice(0, runtime.dialogueContext.length - cap);
  return turn;
}

function dialogueContextSnapshot(limit = 12) {
  const n = Math.max(1, Math.min(24, limit | 0 || 12));
  return runtime.dialogueContext.slice(-n).map(turn => ({ ...turn }));
}

function enqueueLearning(item) {
  if (!item || !item.type) return;
  const max = Math.max(64, runtime.options.learningQueueMax | 0 || 512);
  if (runtime.learningQueue.length >= max) {
    // Preserve newest foreground experiences. Drop the oldest queued replay item,
    // not model capacity or RSL state.
    runtime.learningQueue.shift();
    runtime.learningDropped++;
  }
  runtime.learningQueue.push(item);
}

async function processLearningQueue(epochAtStart, phase) {
  if (!runtime.learningQueue.length) return { processed: 0, remaining: 0, ms: 0 };
  const started = performance.now();
  const maxItems = phase === Phase.IDLE ? 12 : phase === Phase.WARM ? 5 : 2;
  const maxMs = phase === Phase.IDLE ? 18 : phase === Phase.WARM ? 10 : 5;
  let processed = 0;

  while (runtime.learningQueue.length && processed < maxItems && performance.now() - started < maxMs) {
    if (runtime.foregroundBusy || runtime.foregroundEpoch !== epochAtStart) break;
    const item = runtime.learningQueue.shift();
    if (!item) break;

    if (item.type === 'user') {
      runtime.correlation?.observeText?.(item.text, {
        source: 'user', accepted: true, episodic: true, countCycle: true
      });
    } else if (item.type === 'failure') {
      const ch = item.channels || {};
      const quality = Math.max(0, Math.min(1,
        (Number(ch.learnedPOSContext) || 0) * 0.24 +
        (Number(ch.grammarLegality) || 0) * 0.24 +
        (Number(ch.dynamicPOSCoherence) || 0) * 0.18 +
        (Number(ch.scaffoldFit) || 0) * 0.16 +
        (Number(ch.sentenceCompleteness) || 0) * 0.12 +
        (Number(ch.correlationCoherence) || 0) * 0.06
      ));
      const deficit = Math.max(0, Math.max(80, runtime.options.minResponseConfidence || 80) - (Number(item.confidence) || 0));
      const gain = Math.min(3.4, 0.95 + deficit / 28);
      runtime.wordState?.observeTrace?.(item.trace || [], {
        confidence: Number(item.confidence) || 0, accepted: false, quality, gain: Math.min(2.0, gain * 0.65)
      });
      runtime.correlation?.learnFailure?.(item.promptText || '', item.text, {
        confidence: Number(item.confidence) || 0,
        quality,
        gain,
        channels: ch
      });
    } else if (item.type === 'response') {
      // Accepted output updates recurrent context after it is already visible to
      // the user. This keeps response latency separate from consolidation.
      await runtime.rsl.observeText(item.text, {
        countTurn: false,
        syncState: false,
        source: 'model_response',
        learnTransitions: true,
        learningGain: 1.0,
        maxSteps: Math.min(192, runtime.options.maxTokens || 192)
      });
      runtime.wordState?.observeTrace?.(item.trace || [], {
        confidence: item.confidence, accepted: true, quality: Math.max(0.8, (Number(item.confidence) || 80) / 100), gain: 0.75
      });
      runtime.correlation?.observeText?.(item.text, {
        source: 'model_response', confidence: item.confidence,
        accepted: true, episodic: true, countCycle: true
      });
      runtime.correlation?.observePair?.(item.promptText || '', item.text, {
        confidence: item.confidence
      });
      runtime.languageState?.observePair?.(item.promptText || '', item.text, {
        confidence: item.confidence, gain: Math.max(0.45, (Number(item.confidence) || 80) / 100)
      });
    }

    processed++;
    runtime.learningProcessed++;
  }

  if (processed) runtime.storage?.scheduleSave?.(() => exportLearningBundle());
  return { processed, remaining: runtime.learningQueue.length, ms: performance.now() - started };
}

async function backgroundTick() {

  if (!runtime.ready || runtime.phase === Phase.SUSPENDED) return;

  const start = performance.now();
  const epochAtStart = runtime.foregroundEpoch;
  chooseAutomaticPhase(start);

  // Foreground generation has priority. Prompt/response/failure consolidation
  // lives in the learning queue, so Correlation learning never competes with
  // the blocking response path for the same worker/GPU.
  if (!runtime.foregroundBusy && epochAtStart === runtime.foregroundEpoch && runtime.phase !== Phase.SUSPENDED) {
    await processLearningQueue(epochAtStart, runtime.phase);
    if (runtime.foregroundBusy || epochAtStart !== runtime.foregroundEpoch) {
      scheduleBackground();
      return;
    }
    const phaseDecay = runtime.phase === Phase.ACTIVE ? 0.9992 : runtime.phase === Phase.WARM ? 0.997 : 0.990;
    const phaseInhibition = runtime.phase === Phase.ACTIVE ? 0.002 : runtime.phase === Phase.WARM ? 0.006 : 0.012;
    const now = performance.now();

    // A full homeostasis sweep touches all six million neurons. Structural, lexical, correlation and curiosity learning still run every background tick, while the dense
    // recurrent sweep is cadence-controlled so it cannot monopolize the GPU.
    if (now - runtime.lastHomeostasisAt >= homeostasisInterval()) {
      runtime.compute.homeostasis({
        decay: phaseDecay,
        inhibition: phaseInhibition,
        baselinePull: 0.0015,
        tick: runtime.tickCount
      });
      runtime.lastHomeostasisAt = now;
    }

    runtime.tickCount++;
    runtime.rsl?.backgroundTick?.(runtime.phase);
    runtime.correlation?.backgroundTick?.(runtime.phase, epochAtStart, () => runtime.foregroundEpoch);
  }

  runtime.trace.push(
    VilotTraceEvent.BACKGROUND_TICK,
    performance.now() - start,
    runtime.tickCount,
    runtime.foregroundEpoch
  );
  scheduleBackground();
}

async function boot(options = {}) {
  if (runtime.ready) return statusPayload();
  runtime.trace = new VilotTrace(options.traceCapacity || 512);
  runtime.trace.push(VilotTraceEvent.BOOT_BEGIN);

  Object.assign(runtime.options, options || {});
  runtime.arena = new VilotArena({
    stateDim: runtime.options.stateDim,
    maxTokens: runtime.options.maxTokens,
    maxCandidates: runtime.options.maxCandidates,
    topK: runtime.options.topK
  });

  runtime.compute = new VilotCompute(runtime.arena, runtime.trace, runtime.options);
  await runtime.compute.init('./Shaders.wgsl?v=2-rule-data-1');
  runtime.rsl = new VilotRSL(runtime.arena, runtime.compute, runtime.trace, runtime.options);
  runtime.storage = new VilotStorage({ key: 'vilotni2-learning-schema-10-6m-general-language' });
  runtime.ruleStore = new VilotRuleStore(runtime.options);
  await runtime.ruleStore.load('./', 'v=2-rule-data-1');
  runtime.processor = new VilotProcessor(runtime.arena, runtime.rsl, runtime.trace, runtime.options, runtime.ruleStore);
  await runtime.processor.load('./Words.json?v=2-rule-data-1');
  runtime.lexicon = new VilotLexicon(runtime.processor, runtime.options);
  runtime.wordState = new VilotWordState(runtime.lexicon, runtime.rsl, runtime.options);
  runtime.phraseState = new VilotPhraseState(runtime.lexicon, runtime.options);
  runtime.wordMatrix = new VilotWordMatrix(runtime.lexicon, runtime.options);
  runtime.languageState = new VilotLanguageState(runtime.processor, runtime.rsl, runtime.lexicon, runtime.wordState, runtime.phraseState, runtime.options);
  let learned = null;
  try {
    await runtime.storage.init();
    learned = await runtime.storage.load();
    if (learned?.rsl) {
      runtime.rsl.importLearningState(learned.rsl);
      runtime.wordState.importState(learned.wordState);
      runtime.phraseState.importState(learned.phraseState);
      runtime.wordMatrix.importState(learned.wordMatrix);
      runtime.languageState.importState(learned.languageState);
    } else if (learned) {
      // One-way compatibility with the previous RSL-only production state.
      runtime.rsl.importLearningState(learned);
    }
  } catch (_) {}
  runtime.memory = new VilotMemory(runtime.options);
  const lexicalLearning = { wordState: runtime.wordState, phraseState: runtime.phraseState, wordMatrix: runtime.wordMatrix, languageState: runtime.languageState };
  runtime.correlation = new VilotCorrelation(runtime.rsl, runtime.processor, runtime.memory, runtime.trace, runtime.options, lexicalLearning);
  if (learned?.correlationState) runtime.correlation.importState?.(learned.correlationState);
  runtime.matrix = new VilotMatrixEngine(runtime.options);
  runtime.queryPlanner = new VilotQueryPlanner(runtime.processor, runtime.rsl, runtime.languageState, runtime.options, runtime.ruleStore);
  runtime.evidenceSearch = new VilotEvidenceSearch(runtime.processor, runtime.rsl, lexicalLearning, runtime.correlation, runtime.options, runtime.ruleStore);
  runtime.candidateRanker = new VilotCandidateRanker(runtime.processor, runtime.options);
  runtime.decoder = new VilotDecoder(runtime.arena, runtime.rsl, runtime.processor, runtime.trace, runtime.options, runtime.matrix, lexicalLearning);
  runtime.fastLane = new VilotFastLane(
    runtime.decoder,
    runtime.correlation,
    runtime.options,
    runtime.queryPlanner,
    runtime.evidenceSearch,
    runtime.candidateRanker
  );
  runtime.ready = true;
  runtime.lastForegroundAt = performance.now();
  runtime.lastHomeostasisAt = runtime.lastForegroundAt;
  setPhase(Phase.ACTIVE, 'boot');
  runtime.trace.push(VilotTraceEvent.BOOT_READY, 0, runtime.arena.stateDim);
  scheduleBackground();
  return statusPayload();
}

function foregroundUpdate(vector, options = {}) {
  const start = performance.now();
  runtime.foregroundEpoch++;
  runtime.lastForegroundAt = start;
  setPhase(Phase.ACTIVE, 'foreground-command');

  runtime.arena.setInput(vector || [], Number.isFinite(options.scale) ? options.scale : 1);
  runtime.compute.recurrent({
    decay: Number.isFinite(options.decay) ? options.decay : 0.985,
    inputGain: Number.isFinite(options.inputGain) ? options.inputGain : 0.36,
    inhibition: Number.isFinite(options.inhibition) ? options.inhibition : 0.018,
    baselinePull: Number.isFinite(options.baselinePull) ? options.baselinePull : 0.002,
    tick: runtime.tickCount,
    denseInput: runtime.arena.input
  });
  runtime.arena.clearInput();

  runtime.lastHotPathMs = performance.now() - start;
  runtime.trace.push(
    VilotTraceEvent.FOREGROUND_UPDATE,
    runtime.lastHotPathMs,
    runtime.foregroundEpoch,
    runtime.arena.revision
  );
  scheduleBackground();

  return {
    hotPathMs: runtime.lastHotPathMs,
    stateRevision: runtime.arena.revision,
    foregroundEpoch: runtime.foregroundEpoch
  };
}

function runLocalBenchmark(iterations = 2000) {
  iterations = Math.max(50, Math.min(50000, iterations | 0));
  const arena = runtime.arena;

  // Preserve the real persistent state. Benchmark on the alternate preallocated
  // buffers and restore references afterward without allocating tensors.
  const savedRead = arena.readState;
  const savedWrite = arena.writeState;
  const savedRevision = arena.revision;
  const savedSwapCount = arena.swapCount;

  const hadDenseInput = Boolean(arena.input);
  const bench = arena.ensureBenchmarkBuffers();
  const denseInput = arena.ensureDenseInput();
  arena.readState = bench.a;
  arena.writeState = bench.b;
  bench.a.fill(0.0125);
  bench.b.fill(0);
  denseInput.fill(0.025);

  const samples = new Float64Array(iterations);
  for (let n = 0; n < iterations; n++) {
    const t0 = performance.now();
    runtime.compute.runCPU(true, 0.985, 0.36, 0.018, 0.002);
    samples[n] = performance.now() - t0;
  }

  let sum = 0;
  let min = Infinity;
  let max = 0;
  for (let i = 0; i < samples.length; i++) {
    const v = samples[i];
    sum += v;
    if (v < min) min = v;
    if (v > max) max = v;
  }

  // Sorting is diagnostic-only, never part of the hot path.
  const sorted = Array.from(samples).sort((a, b) => a - b);
  const p50 = sorted[Math.floor(sorted.length * 0.50)] || 0;
  const p95 = sorted[Math.floor(sorted.length * 0.95)] || 0;

  arena.readState = savedRead;
  arena.writeState = savedWrite;
  arena.revision = savedRevision;
  arena.swapCount = savedSwapCount;
  arena.clearInput();
  arena.releaseBenchmarkBuffers();
  if (!hadDenseInput) arena.input = null;

  runtime.localBenchmark = {
    iterations,
    meanMs: sum / iterations,
    minMs: min,
    p50Ms: p50,
    p95Ms: p95,
    maxMs: max,
    stateDim: arena.stateDim,
    note: 'CPU recurrent-kernel benchmark inside the worker; excludes message roundtrip and GPU completion.'
  };
  runtime.trace.push(VilotTraceEvent.BENCHMARK, 0, runtime.localBenchmark.meanMs, runtime.localBenchmark.p95Ms);
  return runtime.localBenchmark;
}

async function statusPayload(sync = false) {
  let energy = null;
  if (sync && runtime.compute) {
    const start = performance.now();
    energy = await runtime.compute.syncStateToCPU();
    runtime.trace?.push(VilotTraceEvent.STATE_SYNC, performance.now() - start, energy.meanAbs, energy.peak);
  }

  return {
    ready: runtime.ready,
    phase: runtime.phase,
    visible: runtime.visible,
    tickCount: runtime.tickCount,
    foregroundEpoch: runtime.foregroundEpoch,
    lastHotPathMs: runtime.lastHotPathMs,
    arena: runtime.arena?.stats?.() || null,
    compute: runtime.compute?.info?.() || null,
    energy,
    localBenchmark: runtime.localBenchmark,
    rsl: runtime.rsl?.status?.() || null,
    processor: runtime.processor?.status?.() || null,
    lexicon: runtime.lexicon?.status?.() || null,
    wordState: runtime.wordState?.status?.() || null,
    phraseState: runtime.phraseState?.status?.() || null,
    wordMatrix: runtime.wordMatrix?.status?.() || null,
    languageState: runtime.languageState?.status?.() || null,
    decoder: runtime.decoder?.status?.() || null,
    memory: runtime.memory?.status?.() || null,
    dialogueContext: { turns: runtime.dialogueContext.length, capacity: runtime.dialogueContextCapacity },
    correlation: runtime.correlation?.status?.() || null,
    matrix: runtime.matrix?.status?.() || null,
    queryPlanner: runtime.queryPlanner?.status?.() || null,
    evidenceSearch: runtime.evidenceSearch?.status?.() || null,
    candidateRanker: runtime.candidateRanker?.status?.() || null,
    ruleStore: runtime.ruleStore?.status?.() || null,
    fastLane: runtime.fastLane?.status?.() || null,
    learningQueue: { pending: runtime.learningQueue.length, processed: runtime.learningProcessed, dropped: runtime.learningDropped },
    storage: runtime.storage?.status?.() || null,
    goals: {
      neuronCount: runtime.arena?.stateDim || 0,
      minResponseConfidence: Math.max(80, Number(runtime.options.minResponseConfidence) || 80),
      latencyTargetMs: Number(runtime.options.latencyTargetMs) || 40,
      highEffort: true,
      unifiedVilotNI2: true,
      sixMillionNeuronCore: true,
      trainingInfoRequired: false,
      trainingInfoPresentAsReference: true,
      trainingInfoLoadedForResponses: false,
      dynamicPOSPerOccurrence: true,
      continuousBackgroundLearning: true,
      continuousCorrelationLearning: true,
      unifiedCorrelationLearning: true,
      curiosityExploration: true,
      confidenceRemakesUntilAboveFloor: true,
      blockingLearningDuringGeneration: false,
      batchedCandidateSearch: true,
      focusedQueryPlanning: true,
      focusedEvidenceSearch: true,
      nonInflatingCandidateRanking: true,
      externalGrammarRules: true,
      externalPunctuationRules: true,
      externalQueryPatterns: true,
      externalContextRules: true,
      semanticRelationRouting: true,
      lexicalDefinitionIndex: true,
      residentCandidateMatrix: true,
      learnedWordState: true,
      learnedPhraseState: true,
      learnedWordMatrix: true,
      adaptiveLexicalConfidence: true,
      acceptedResponseLearningDeferred: true,
      productionRuntime: true,
      orderedRecurrentBatching: true,
      sparseActiveExecution: true,
      exactZeroSkip: true,
      foregroundPriorityScheduling: true
    },
    backgroundIntervals: {
      activeMs: runtime.options.activeTickMs,
      warmMs: runtime.options.warmTickMs,
      idleMs: runtime.options.idleTickMs,
      activeHomeostasisMs: runtime.options.activeHomeostasisMs,
      warmHomeostasisMs: runtime.options.warmHomeostasisMs,
      idleHomeostasisMs: runtime.options.idleHomeostasisMs
    }
  };
}

onmessage = async event => {
  const message = event.data || {};
  const type = message.type || '';
  const requestId = message.requestId ?? null;
  runtime.trace?.push?.(VilotTraceEvent.COMMAND, 0, requestId || 0);

  try {
    if (type !== 'BOOT' && !runtime.ready) {
      await boot({});
    }

    switch (type) {
      case 'BOOT': {
        const status = await boot(message.options || {});
        send('READY', { status }, requestId);
        break;
      }

      case 'PING':
        send('PONG', { workerNow: performance.now(), stateRevision: runtime.arena.revision }, requestId);
        break;

      case 'FOREGROUND_UPDATE': {
        const result = foregroundUpdate(message.vector, message.options || {});
        send('FOREGROUND_DONE', result, requestId);
        break;
      }

      case 'RSL_OBSERVE': {
        const result = await runtime.rsl.observeText(message.text || '', message.options || {});
        runtime.lastForegroundAt = performance.now();
        runtime.foregroundEpoch++;
        setPhase(Phase.ACTIVE, 'rsl-observe');
        scheduleBackground();
        send('RSL_RESULT', { result }, requestId);
        break;
      }

      case 'RSL_STATUS':
        send('RSL_STATUS_RESULT', { rsl: runtime.rsl.status() }, requestId);
        break;


      case 'GENERATE': {
        const totalStart = performance.now();
        runtime.foregroundBusy = true;
        runtime.lastForegroundAt = totalStart;
        runtime.foregroundEpoch++;
        setPhase(Phase.ACTIVE, 'generate-fast-lane');

        const promptText = String(message.text || '');
        const promptState = await runtime.rsl.observeText(promptText, {
          ...(message.options || {}),
          countTurn: true,
          syncState: false,
          source: 'user',
          learnTransitions: true
        });
        const analysis = runtime.languageState.enrichAnalysis(runtime.processor.analyzePrompt(promptText));

        // Language study is queued immediately but is not executed before the
        // foreground response. RSL prompt activation is the only learning-related
        // state update required to understand the current request.
        enqueueLearning({ type: 'user', text: promptText });

        const minConfidence = Math.max(80, Number(runtime.options.minResponseConfidence) || 80);
        const priorDialogueContext = dialogueContextSnapshot(12);
        const search = await runtime.fastLane.run(
          promptText,
          analysis,
          { ...(message.options || {}), minConfidence, dialogueContext: priorDialogueContext },
          progress => send('RESPONSE_PROGRESS', progress)
        );

        if (search.contextRequired) {
          const clarificationText = String(search.clarificationText || 'What should I use as the context for that question?');
          runtime.foregroundBusy = false;
          scheduleBackground();
          const totalMs = performance.now() - totalStart;
          send('GENERATE_RESULT', {
            result: {
              text: clarificationText,
              confidence: null,
              contextRequired: true,
              contextResolved: false,
              contextReason: search.planSummary?.contextReason || 'context-dependent',
              attempts: 0,
              candidateBatches: 0,
              totalMs,
              latencyTargetMs: runtime.options.latencyTargetMs,
              withinLatencyTarget: totalMs < runtime.options.latencyTargetMs,
              latencyWarning: totalMs >= runtime.options.latencyTargetMs,
              fastLaneMs: search.fastLaneMs,
              fastLaneRoute: search.route,
              analysis: {
                intent: analysis.intent,
                requestedSlot: analysis.requestedSlot,
                isQuestion: analysis.isQuestion,
                lexicalCoverage: analysis.lexicalCoverage,
                contentWords: analysis.contentWords.slice(0, 24)
              }
            }
          }, requestId);
          break;
        }

        let generated = search.selected;
        const bestObserved = generated || search.bestCandidate || null;
        const bestObservedConfidence = Number(bestObserved?.confidence) || 0;
        if (!generated || !(Number(generated?.confidence) > minConfidence)) {
          throw new Error(`FastLane confidence invariant failed: response did not clear >${minConfidence}%.`);
        }
        const withheld = false;

        {
          runtime.correlation?.noteSelected?.(promptText, generated, { confidence: generated?.confidence });
          runtime.correlation?.queueCuriosityCandidates?.(
            promptText,
            search.consideredCandidates || search.failed || [],
            generated,
            analysis
          );
        }
        const emittedText = String(generated?.text || '');
        pushDialogueTurn('user', promptText);
        pushDialogueTurn('assistant', emittedText, { confidence: Number(generated?.confidence) || 0 });

        // Failed alternatives become background study material. They do not hold
        // the user hostage while Correlation's unified learner updates the model.
        for (const failed of search.failed.slice(-96)) {
          enqueueLearning({
            type: 'failure',
            promptText,
            text: String(failed?.text || ''),
            confidence: Number(failed?.confidence) || 0,
            channels: failed?.diagnostics?.channels || null,
            trace: failed?.diagnostics?.generationTrace || []
          });
        }
        {
          enqueueLearning({
            type: 'response',
            promptText,
            text: emittedText,
            confidence: Number(generated?.confidence) || 0,
            trace: generated?.diagnostics?.generationTrace || []
          });
        }

        const confidenceHistory = search.failed.map(x => Number(x?.confidence) || 0);
        if (generated) confidenceHistory.push(Number(generated?.confidence) || 0);
        const gatedDiagnostics = generated?.diagnostics
          ? {
              ...generated.diagnostics,
              confidenceGateFailed: false,
              confidenceHistory: confidenceHistory.slice(-128),
              foregroundLearningCycles: 0,
              learningDeferredToBackground: true,
              failedCandidatesQueuedForStudy: Math.min(96, search.failed.length),
              unifiedSixMillionCore: true,
              correlationActive: true,
              unifiedCorrelationActive: true,
              curiosityActive: true,
              curiositySelection: search.curiosity || null,
              batchedCandidateSearch: true,
              residentCandidateMatrix: true,
              learnedWordState: true,
              learnedPhraseState: true,
              learnedWordMatrix: true,
              generalLanguageState: true,
              tenseAspectMoodVoiceAgreement: true,
              epistemicUnknownHandling: true,
              lexicalAdaptation: generated?.diagnostics?.channels?.lexicalAdaptation ?? 0.5,
              fastLaneMs: search.fastLaneMs,
              fastLaneRoute: search.route || null,
              focusedFastPath: Boolean(search.focusedFastPath),
              queryPlan: search.planSummary || null,
              evidenceSearch: search.evidenceSummary || null,
              fastRank: search.rankSummary || null,
              batchCount: search.batchCount,
              candidateCount: search.candidateCount,
              remadeUntilAboveConfidenceFloor: true
            }
          : {
              confidenceGateFailed: true,
              confidenceHistory: confidenceHistory.slice(-128),
              bestObservedConfidence,
              confidenceFloor: minConfidence,
              foregroundLearningCycles: 0,
              learningDeferredToBackground: true,
              failedCandidatesQueuedForStudy: Math.min(96, search.failed.length),
              batchedCandidateSearch: true,
              fastLaneMs: search.fastLaneMs,
              batchCount: search.batchCount,
              candidateCount: search.candidateCount,
              searchExhausted: Boolean(search.searchExhausted),
              searchStoppedBy: search.searchStoppedBy || null,
              remadeUntilAboveConfidenceFloor: false
            };

        generated = {
          ...(generated || {}),
          text: emittedText,
          confidence: generated?.confidence,
          diagnostics: gatedDiagnostics,
          withheld,
          bestObservedConfidence,
          confidenceGate: minConfidence,
          confidenceRule: `>${minConfidence}`,
          attempts: search.candidateCount,
          candidateBatches: search.batchCount,
          selfLearningCycles: 0,
          searchExhausted: Boolean(search.searchExhausted),
          searchStoppedBy: search.searchStoppedBy || null,
          backgroundLearningQueued: runtime.learningQueue.length
        };

        runtime.foregroundBusy = false;
        scheduleBackground();
        const totalMs = performance.now() - totalStart;
        send('GENERATE_RESULT', {
          result: {
            ...generated,
            totalMs,
            latencyTargetMs: runtime.options.latencyTargetMs,
            withinLatencyTarget: totalMs < runtime.options.latencyTargetMs,
            latencyWarning: totalMs >= runtime.options.latencyTargetMs,
            promptState,
            responseState: { queuedForBackgroundConsolidation: true },
            analysis: {
              intent: analysis.intent,
              requestedSlot: analysis.requestedSlot,
              isQuestion: analysis.isQuestion,
              lexicalCoverage: analysis.lexicalCoverage,
              contentWords: analysis.contentWords.slice(0, 24)
            }
          }
        }, requestId);
        break;
      }

      case 'PROCESSOR_INSPECT': {
        send('PROCESSOR_INSPECT_RESULT', { word: runtime.processor.inspectWord(message.word || '') }, requestId);
        break;
      }

      case 'DECODER_INSPECT': {
        const results = runtime.decoder.inspectLearnedContext(message.query || '', message.limit || 8);
        send('DECODER_INSPECT_RESULT', { results }, requestId);
        break;
      }

      case 'DECODER_STATUS': {
        send('DECODER_STATUS_RESULT', { decoder: runtime.decoder.status(), processor: runtime.processor.status() }, requestId);
        break;
      }

      case 'CHECKPOINT': {
        // A checkpoint must capture the real GPU-resident state, not a stale
        // CPU mirror. This readback is deliberately outside the hot path.
        await runtime.compute.syncStateToCPU();
        runtime.arena.checkpoint();
        runtime.rsl?.checkpoint?.();
        send('CHECKPOINTED', { stateRevision: runtime.arena.revision }, requestId);
        break;
      }

      case 'RESTORE_CHECKPOINT':
        runtime.arena.restoreCheckpoint();
        runtime.rsl?.restoreCheckpoint?.();
        runtime.compute.uploadCPUState();
        send('RESTORED', { stateRevision: runtime.arena.revision }, requestId);
        break;

      case 'RESET_STATE':
        runtime.arena.resetState();
        runtime.rsl?.reset?.();
        runtime.memory?.reset?.();
        runtime.dialogueContext.length = 0;
        runtime.correlation?.reset?.();
        runtime.wordState?.reset?.();
        runtime.languageState?.reset?.();
        runtime.phraseState?.reset?.();
        runtime.wordMatrix?.reset?.();
        try { await runtime.storage?.clear?.(); } catch (_) {}
        runtime.trace?.push?.(VilotTraceEvent.RSL_RESET, 0);
        runtime.compute.uploadCPUState();
        send('RESET_DONE', { stateRevision: runtime.arena.revision }, requestId);
        break;

      case 'SET_DIALOGUE_CONTEXT': {
        const incoming = Array.isArray(message.turns) ? message.turns : [];
        runtime.dialogueContext.length = 0;
        for (const raw of incoming.slice(-runtime.dialogueContextCapacity)) {
          const text = String(raw?.text || '').trim();
          if (!text) continue;
          runtime.dialogueContext.push({
            role: String(raw?.role || 'unknown').toLowerCase(),
            text: text.slice(0, 4096),
            at: Number(raw?.at) || Date.now(),
            confidence: Number.isFinite(Number(raw?.confidence)) ? Number(raw.confidence) : null
          });
        }
        send('DIALOGUE_CONTEXT_SET', { turns: runtime.dialogueContext.length }, requestId);
        break;
      }

      case 'CLEAR_DIALOGUE_CONTEXT':
        runtime.dialogueContext.length = 0;
        send('DIALOGUE_CONTEXT_CLEARED', { turns: 0 }, requestId);
        break;

      case 'SET_VISIBILITY':
        runtime.visible = message.visible !== false;
        if (runtime.visible && runtime.phase === Phase.SUSPENDED) {
          setPhase(Phase.WARM, 'document-visible');
          runtime.lastForegroundAt = performance.now() - runtime.options.warmAfterMs;
          scheduleBackground();
        } else {
          chooseAutomaticPhase(performance.now());
          scheduleBackground();
        }
        send('VISIBILITY_SET', { visible: runtime.visible, phase: runtime.phase }, requestId);
        break;

      case 'SUSPEND':
        clearTimeout(runtime.timer);
        setPhase(Phase.SUSPENDED, message.reason || 'manual');
        runtime.trace.push(VilotTraceEvent.SUSPEND);
        send('SUSPENDED', { phase: runtime.phase }, requestId);
        break;

      case 'RESUME':
        runtime.lastForegroundAt = performance.now();
        setPhase(Phase.WARM, 'resume');
        runtime.trace.push(VilotTraceEvent.RESUME);
        scheduleBackground();
        send('RESUMED', { phase: runtime.phase }, requestId);
        break;

      case 'GET_STATUS': {
        const status = await statusPayload(Boolean(message.sync));
        send('STATUS', { status }, requestId);
        break;
      }

      case 'GET_TRACE':
        send('TRACE', { trace: runtime.trace.snapshot(message.limit || 64) }, requestId);
        break;

      case 'BENCHMARK_LOCAL':
        send('BENCHMARK_RESULT', { benchmark: runLocalBenchmark(message.iterations) }, requestId);
        break;

      case 'DESTROY':
        clearTimeout(runtime.timer);
        try { await runtime.storage?.flush?.(); } catch (_) {}
        runtime.storage?.close?.();
        runtime.compute?.destroy?.();
        runtime.ready = false;
        send('DESTROYED', {}, requestId);
        close();
        break;

      default:
        throw new Error(`Unknown VilotNI 2 worker command: ${type}`);
    }
  } catch (error) {
    runtime.foregroundBusy = false;
    runtime.trace?.push?.(VilotTraceEvent.ERROR);
    send('ERROR', {
      message: String(error?.message || error),
      stack: String(error?.stack || '')
    }, requestId);
  }
};

/**
 * Worker production research extension.
 *
 * Executable, data-driven numerical/statistical/graph utilities used by
 * diagnostics, background learning, calibration, matrix experiments and
 * future compute paths. No prompt-specific phrases or benchmark answers live
 * here. Methods stay out of the foreground hot path unless explicitly called.
 */
(() => {
  'use strict';

  class VilotWorkerProductionLab {
    constructor(options = {}) {
      this.options = { ...options };
      this.calls = 0;
      this.createdAt = performance?.now?.() ?? Date.now();
    }

    clamp(x, lo = 0, hi = 1) {
      const n = Number(x);
      if (!Number.isFinite(n)) return lo;
      return Math.max(lo, Math.min(hi, n));
    }

    sum(values) {
      let total = 0;
      for (const value of values || []) {
        const n = Number(value);
        if (Number.isFinite(n)) total += n;
      }
      return total;
    }

    mean(values) {
      let total = 0;
      let count = 0;
      for (const value of values || []) {
        const n = Number(value);
        if (!Number.isFinite(n)) continue;
        total += n;
        count++;
      }
      return count ? total / count : 0;
    }

    variance(values) {
      const input = Array.from(values || [], Number).filter(Number.isFinite);
      if (input.length < 2) return 0;
      const m = this.mean(input);
      let total = 0;
      for (const value of input) {
        const d = value - m;
        total += d * d;
      }
      return total / input.length;
    }

    stddev(values) {
      return Math.sqrt(Math.max(0, this.variance(values)));
    }

    normalize(values, epsilon = 1e-9) {
      const input = Array.from(values || [], Number);
      let norm2 = 0;
      for (const value of input) if (Number.isFinite(value)) norm2 += value * value;
      const norm = Math.sqrt(norm2) + Math.max(Number.EPSILON, epsilon);
      return input.map(value => Number.isFinite(value) ? value / norm : 0);
    }

    softmax(values, temperature = 1) {
      const input = Array.from(values || [], Number);
      if (!input.length) return [];
      const t = Math.max(1e-6, Number(temperature) || 1);
      let max = -Infinity;
      for (const value of input) if (Number.isFinite(value) && value > max) max = value;
      if (!Number.isFinite(max)) return input.map(() => 1 / input.length);
      const exps = input.map(value => Math.exp((Number.isFinite(value) ? value : -1e9) / t - max / t));
      const z = exps.reduce((a,b) => a + b, 0) || 1;
      return exps.map(value => value / z);
    }

    logSoftmax(values, temperature = 1) {
      const probs = this.softmax(values, temperature);
      return probs.map(value => Math.log(Math.max(1e-12, value)));
    }

    entropy(probabilities) {
      let h = 0;
      for (const raw of probabilities || []) {
        const p = Number(raw);
        if (!Number.isFinite(p) || p <= 0) continue;
        h -= p * Math.log2(p);
      }
      return h;
    }

    dot(a, b) {
      const n = Math.min(a?.length || 0, b?.length || 0);
      let total = 0;
      for (let i = 0; i < n; i++) total += (Number(a[i]) || 0) * (Number(b[i]) || 0);
      return total;
    }

    cosine(a, b, epsilon = 1e-9) {
      const n = Math.min(a?.length || 0, b?.length || 0);
      let dot = 0, aa = 0, bb = 0;
      for (let i = 0; i < n; i++) {
        const x = Number(a[i]) || 0;
        const y = Number(b[i]) || 0;
        dot += x * y;
        aa += x * x;
        bb += y * y;
      }
      return dot / (Math.sqrt(aa * bb) + Math.max(epsilon, Number.EPSILON));
    }

    l1Distance(a, b) {
      const n = Math.max(a?.length || 0, b?.length || 0);
      let total = 0;
      for (let i = 0; i < n; i++) total += Math.abs((Number(a?.[i]) || 0) - (Number(b?.[i]) || 0));
      return total;
    }

    l2Distance(a, b) {
      const n = Math.max(a?.length || 0, b?.length || 0);
      let total = 0;
      for (let i = 0; i < n; i++) {
        const d = (Number(a?.[i]) || 0) - (Number(b?.[i]) || 0);
        total += d * d;
      }
      return Math.sqrt(total);
    }

    topK(items, k = 8, score = x => Number(x?.score ?? x) || 0) {
      const limit = Math.max(0, k | 0);
      if (!limit) return [];
      const heap = [];
      for (const item of items || []) {
        const value = score(item);
        if (!Number.isFinite(value)) continue;
        heap.push({ item, value });
      }
      heap.sort((a,b) => b.value - a.value);
      return heap.slice(0, limit).map(entry => entry.item);
    }

    argmax(values) {
      let index = -1;
      let best = -Infinity;
      for (let i = 0; i < (values?.length || 0); i++) {
        const value = Number(values[i]);
        if (Number.isFinite(value) && value > best) {
          best = value;
          index = i;
        }
      }
      return { index, value: best };
    }

    quantile(values, q = 0.5) {
      const input = Array.from(values || [], Number).filter(Number.isFinite).sort((a,b) => a-b);
      if (!input.length) return 0;
      const p = Math.max(0, Math.min(1, Number(q) || 0));
      const pos = (input.length - 1) * p;
      const lo = Math.floor(pos), hi = Math.ceil(pos);
      if (lo === hi) return input[lo];
      const t = pos - lo;
      return input[lo] * (1 - t) + input[hi] * t;
    }

    median(values) {
      return this.quantile(values, 0.5);
    }

    mad(values) {
      const input = Array.from(values || [], Number).filter(Number.isFinite);
      if (!input.length) return 0;
      const m = this.median(input);
      return this.median(input.map(value => Math.abs(value - m)));
    }

    ema(previous, current, alpha = 0.1) {
      const a = Math.max(0, Math.min(1, Number(alpha) || 0));
      const p = Number(previous);
      const c = Number(current);
      if (!Number.isFinite(c)) return Number.isFinite(p) ? p : 0;
      if (!Number.isFinite(p)) return c;
      return p + (c - p) * a;
    }

    rollingMean(values, window = 8) {
      const input = Array.from(values || [], Number);
      const w = Math.max(1, window | 0);
      const out = new Array(input.length).fill(0);
      let sum = 0;
      for (let i = 0; i < input.length; i++) {
        sum += Number.isFinite(input[i]) ? input[i] : 0;
        if (i >= w) sum -= Number.isFinite(input[i-w]) ? input[i-w] : 0;
        out[i] = sum / Math.min(w, i + 1);
      }
      return out;
    }

    rollingVariance(values, window = 8) {
      const input = Array.from(values || [], Number);
      const w = Math.max(2, window | 0);
      const out = new Array(input.length).fill(0);
      for (let i = 0; i < input.length; i++) {
        const start = Math.max(0, i - w + 1);
        out[i] = this.variance(input.slice(start, i + 1));
      }
      return out;
    }

    zScores(values) {
      const input = Array.from(values || [], Number);
      const m = this.mean(input);
      const sd = this.stddev(input) || 1;
      return input.map(value => Number.isFinite(value) ? (value - m) / sd : 0);
    }

    pearson(a, b) {
      const n = Math.min(a?.length || 0, b?.length || 0);
      if (n < 2) return 0;
      let sx=0, sy=0, sxx=0, syy=0, sxy=0, count=0;
      for (let i=0;i<n;i++) {
        const x=Number(a[i]), y=Number(b[i]);
        if (!Number.isFinite(x)||!Number.isFinite(y)) continue;
        sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; count++;
      }
      if (count<2) return 0;
      const num=count*sxy-sx*sy;
      const den=Math.sqrt(Math.max(0,(count*sxx-sx*sx)*(count*syy-sy*sy)));
      return den ? num/den : 0;
    }

    covariance(a, b) {
      const n = Math.min(a?.length || 0, b?.length || 0);
      if (n < 2) return 0;
      const ax = Array.from(a).slice(0,n).map(Number);
      const bx = Array.from(b).slice(0,n).map(Number);
      const ma = this.mean(ax), mb = this.mean(bx);
      let total = 0, count = 0;
      for (let i=0;i<n;i++) {
        if (!Number.isFinite(ax[i]) || !Number.isFinite(bx[i])) continue;
        total += (ax[i]-ma)*(bx[i]-mb);
        count++;
      }
      return count ? total/count : 0;
    }

    weightedMean(values, weights) {
      const n = Math.min(values?.length || 0, weights?.length || 0);
      let sum = 0, weight = 0;
      for (let i=0;i<n;i++) {
        const x=Number(values[i]), w=Number(weights[i]);
        if (!Number.isFinite(x)||!Number.isFinite(w)||w<=0) continue;
        sum += x*w;
        weight += w;
      }
      return weight ? sum/weight : 0;
    }

    weightedVariance(values, weights) {
      const n = Math.min(values?.length || 0, weights?.length || 0);
      const mean = this.weightedMean(values, weights);
      let sum = 0, weight = 0;
      for (let i=0;i<n;i++) {
        const x=Number(values[i]), w=Number(weights[i]);
        if (!Number.isFinite(x)||!Number.isFinite(w)||w<=0) continue;
        const d=x-mean;
        sum += w*d*d;
        weight += w;
      }
      return weight ? sum/weight : 0;
    }

    sigmoid(x) {
      const n = Math.max(-60, Math.min(60, Number(x) || 0));
      return 1 / (1 + Math.exp(-n));
    }

    tanh(x) {
      const n = Math.max(-30, Math.min(30, Number(x) || 0));
      const e2 = Math.exp(2*n);
      return (e2 - 1) / (e2 + 1);
    }

    gelu(x) {
      const n=Number(x)||0;
      return 0.5*n*(1+Math.tanh(Math.sqrt(2/Math.PI)*(n+0.044715*n*n*n)));
    }

    swish(x, beta = 1) {
      const n=Number(x)||0;
      return n * this.sigmoid((Number(beta)||1)*n);
    }

    stableHash(value, seed = 2166136261) {
      const text = String(value ?? '');
      let h = seed >>> 0;
      for (let i=0;i<text.length;i++) {
        h ^= text.charCodeAt(i);
        h = Math.imul(h, 16777619) >>> 0;
      }
      return h >>> 0;
    }

    hashVector(tokens, dim = 64) {
      const size=Math.max(4,dim|0);
      const out=new Float32Array(size);
      for (const token of tokens||[]) {
        const text=String(token??'');
        const h=this.stableHash(text);
        const i=h%size;
        const sign=(h&1)?1:-1;
        out[i]+=sign;
      }
      return out;
    }

    ngrams(tokens, minN = 1, maxN = 3) {
      const input=Array.from(tokens||[], x=>String(x));
      const out=[];
      const lo=Math.max(1,minN|0), hi=Math.max(lo,maxN|0);
      for (let n=lo;n<=hi;n++) {
        for (let i=0;i+n<=input.length;i++) out.push(input.slice(i,i+n));
      }
      return out;
    }

    ngramCounts(tokens, minN = 1, maxN = 3) {
      const counts=new Map();
      for (const gram of this.ngrams(tokens,minN,maxN)) {
        const key=gram.join('');
        counts.set(key,(counts.get(key)||0)+1);
      }
      return counts;
    }

    transitionCounts(tokens) {
      const input=Array.from(tokens||[],x=>String(x));
      const map=new Map();
      for (let i=1;i<input.length;i++) {
        const key=input[i-1]+''+input[i];
        map.set(key,(map.get(key)||0)+1);
      }
      return map;
    }

    mutualInformation(jointCount, leftCount, rightCount, total) {
      const j=Number(jointCount)||0, l=Number(leftCount)||0, r=Number(rightCount)||0, t=Number(total)||0;
      if (j<=0||l<=0||r<=0||t<=0) return 0;
      return Math.log2((j*t)/(l*r));
    }

    decayMap(map, factor = 0.995, floor = 1e-6) {
      if (!(map instanceof Map)) return map;
      const f=Math.max(0,Math.min(1,Number(factor)||0));
      for (const [key,value] of map) {
        const next=(Number(value)||0)*f;
        if (Math.abs(next)<floor) map.delete(key); else map.set(key,next);
      }
      return map;
    }

    pruneMap(map, maxSize = 4096) {
      if (!(map instanceof Map) || map.size<=maxSize) return map;
      const entries=Array.from(map.entries()).sort((a,b)=>(Number(b[1])||0)-(Number(a[1])||0));
      map.clear();
      for (const [key,value] of entries.slice(0,Math.max(0,maxSize|0))) map.set(key,value);
      return map;
    }

    sparseAdd(target, indices, values, gain = 1) {
      const g=Number(gain)||1;
      const n=Math.min(indices?.length||0, values?.length||0);
      for (let i=0;i<n;i++) {
        const idx=indices[i]|0;
        if (idx<0||idx>=target.length) continue;
        target[idx]+=(Number(values[i])||0)*g;
      }
      return target;
    }

    sparseDot(dense, indices, values) {
      const n=Math.min(indices?.length||0,values?.length||0);
      let total=0;
      for (let i=0;i<n;i++) {
        const idx=indices[i]|0;
        if (idx<0||idx>=dense.length) continue;
        total+=(Number(dense[idx])||0)*(Number(values[i])||0);
      }
      return total;
    }

    status() {
      return {
        module: 'Worker',
        role: 'production-research-extension',
        calls: this.calls,
        methodCount: Object.getOwnPropertyNames(Object.getPrototypeOf(this)).length - 1,
        ageMs: (performance?.now?.() ?? Date.now()) - this.createdAt
      };
    }
  }

  globalThis.VilotWorkerProductionLab = VilotWorkerProductionLab;
})();
