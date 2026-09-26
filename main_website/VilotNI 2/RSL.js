/**
 * VilotNI 2 - Receptive Synaptic Logic (RSL) production core.
 *
 * The 6,000,000-neuron H_t state is persistent across turns. Lexical evidence
 * advances the state autoregressively, while sparse-active ordered batching preserves the per-token recurrence, skips
 * exact-zero inactive slots, and removes repeated command-submission overhead.
 */
(() => {
  'use strict';

  const isWordCode = code =>
    (code >= 48 && code <= 57) ||
    (code >= 65 && code <= 90) ||
    (code >= 97 && code <= 122) ||
    code === 39 || code === 45 || code >= 128;

  class VilotRSL {
    constructor(arena, compute, trace = null, options = {}) {
      this.arena = arena;
      this.compute = compute;
      this.trace = trace;
      this.stateDim = arena.stateDim;

      this.options = {
        maxLexicalStepsPerTurn: Math.max(8, options.rslMaxLexicalStepsPerTurn | 0 || 96),
        recentHashCapacity: Math.max(32, options.rslRecentHashCapacity | 0 || 256),
        receptiveSlotsPerToken: 6,
        decay: Number.isFinite(options.rslDecay) ? options.rslDecay : 0.992,
        inputGain: Number.isFinite(options.rslInputGain) ? options.rslInputGain : 0.42,
        inhibition: Number.isFinite(options.rslInhibition) ? options.rslInhibition : 0.012,
        baselinePull: Number.isFinite(options.rslBaselinePull) ? options.rslBaselinePull : 0.0015,
        maxTransitionSources: Math.max(128, options.rslMaxTransitionSources | 0 || 4096),
        maxTransitionsPerSource: Math.max(8, options.rslMaxTransitionsPerSource | 0 || 64),
        maxContextSources: Math.max(256, options.rslMaxContextSources | 0 || 8192),
        maxContextTargets: Math.max(8, options.rslMaxContextTargets | 0 || 64),
        maxMorphologyRoots: Math.max(256, options.rslMaxMorphologyRoots | 0 || 8192),
        maxCorrelationSources: Math.max(256, options.rslMaxCorrelationSources | 0 || 8192),
        maxCorrelationTargets: Math.max(8, options.rslMaxCorrelationTargets | 0 || 64)
      };

      this.recentHashes = new Uint32Array(this.options.recentHashCapacity);
      this.checkpointHashes = new Uint32Array(this.options.recentHashCapacity);
      this.recentHashCursor = 0;
      this.recentHashCount = 0;
      this.checkpointHashCursor = 0;
      this.checkpointHashCount = 0;

      this.recentSlotIds = new Uint32Array(8);
      this.recentSlotWeights = new Float32Array(8);
      this.batchIds = new Uint32Array(this.options.maxLexicalStepsPerTurn * this.options.receptiveSlotsPerToken);
      this.batchWeights = new Float32Array(this.options.maxLexicalStepsPerTurn * this.options.receptiveSlotsPerToken);

      this.turnCount = 0;
      this.totalLexicalSteps = 0;
      this.lastLexicalSteps = 0;
      this.lastCharacters = 0;
      this.lastNovelty = 1;
      this.noveltyEMA = 1;
      this.momentumEMA = 0;
      this.lastHotPathMs = 0;
      this.lastSyncMs = 0;
      this.lastHash = 0;
      this.checkpointScalars = new Float64Array(8);

      // One unified VilotNI 2 core. Word meaning is not injected from a separate
      // definition database. RSL learns token, POS_Type, contextual, and
      // morphological behavior from actual language trajectories.
      this.transitions = new Map();
      this.posTransitions = new Map();
      this.posTrigramTransitions = new Map();
      this.contextTransitions = new Map();
      this.contextWindowTransitions = new Map();
      this.wordPOSUsage = new Map();
      this.morphologyRelations = new Map();
      this.wordCorrelations = new Map();
      this.posCorrelations = new Map();
      // Kept only for migration of older saved states; new V2 does not use a
      // TrainingInfo semantic-memory channel.
      this.semanticMemory = new Map();
      this.transitionUpdates = 0;
      this.posTransitionUpdates = 0;
      this.posTrigramUpdates = 0;
      this.contextTransitionUpdates = 0;
      this.contextWindowUpdates = 0;
      this.wordPOSUpdates = 0;
      this.morphologyUpdates = 0;
      this.correlationUpdates = 0;
      this.posCorrelationUpdates = 0;
      this.semanticUpdates = 0;
      this.selfLearningCycles = 0;
      this.backgroundLearningCycles = 0;
      this.lastSelfLearningSupport = 0;
    }

    _trimTransitionMap(bucket = this.transitions) {
      if (bucket.size <= this.options.maxTransitionSources) return;
      const keys = bucket.keys();
      const remove = Math.max(1, Math.ceil(bucket.size * 0.08));
      for (let i = 0; i < remove; i++) {
        const n = keys.next();
        if (n.done) break;
        bucket.delete(n.value);
      }
    }

    _learnTransition(prev, next, source = 'shared', gain = 1.0) {
      prev = String(prev || '').toLowerCase();
      next = String(next || '').toLowerCase();
      if (!prev || !next || prev === next) return;
      let row = this.transitions.get(prev);
      if (!row) {
        row = new Map();
        this.transitions.set(prev, row);
        this._trimTransitionMap(this.transitions);
      }
      const old = row.get(next) || 0;
      const step = 0.48 * Math.max(0.1, Math.min(4.0, Number(gain) || 1.0));
      row.set(next, Math.min(8, old * 0.988 + step));
      if (row.size > this.options.maxTransitionsPerSource) {
        const weakest = Array.from(row.entries())
          .sort((a, b) => a[1] - b[1])
          .slice(0, row.size - this.options.maxTransitionsPerSource);
        for (const [word] of weakest) row.delete(word);
      }
      this.transitionUpdates++;
    }

    _learnEdge(bucket, from, to, gain = 1, cap = 8, maxSources = 4096, maxTargets = 64) {
      from = String(from || '').trim();
      to = String(to || '').trim();
      if (!from || !to) return 0;
      let row = bucket.get(from);
      if (!row) {
        row = new Map();
        bucket.set(from, row);
        if (bucket.size > maxSources) {
          const remove = Math.max(1, Math.ceil(bucket.size * 0.05));
          const keys = bucket.keys();
          for (let i = 0; i < remove; i++) {
            const n = keys.next();
            if (n.done) break;
            bucket.delete(n.value);
          }
        }
      }
      const old = row.get(to) || 0;
      const step = 0.36 * Math.max(0.05, Math.min(4, Number(gain) || 1));
      row.set(to, Math.min(cap, old * 0.994 + step));
      if (row.size > maxTargets) {
        const weakest = Array.from(row.entries()).sort((a, b) => a[1] - b[1]).slice(0, row.size - maxTargets);
        for (const [key] of weakest) row.delete(key);
      }
      return row.get(to) || 0;
    }

    seedPOSCurriculum(relations = {}, trigrams = [], gain = 0.82, cycles = 5) {
      const g = Math.max(0.1, Math.min(2.0, Number(gain) || 0.82));
      const repeats = Math.max(1, Math.min(12, cycles | 0 || 5));
      let target = 0;
      for (let cycle = 0; cycle < repeats; cycle++) target += g * (1 - cycle * 0.035);
      let edges = 0;
      let triples = 0;

      // Curriculum seeding establishes a floor rather than adding the same
      // bootstrap evidence on every page reload. Persisted learning can grow
      // above this floor naturally without startup inflation.
      for (const [from, targets] of Object.entries(relations || {})) {
        for (const to of Array.isArray(targets) ? targets : []) {
          const current = this.posTransitions.get(String(from))?.get(String(to)) || 0;
          if (current + 1e-6 >= target) continue;
          this._learnEdge(this.posTransitions, from, to, target - current, 8, 96, 48);
          this.posTransitionUpdates++;
          edges++;
        }
      }
      for (const triple of Array.isArray(trigrams) ? trigrams : []) {
        if (!Array.isArray(triple) || triple.length < 3) continue;
        const [p2, p1, next] = triple;
        const key = `${p2}>${p1}`;
        const current = this.posTrigramTransitions.get(key)?.get(String(next)) || 0;
        if (current + 1e-6 >= target) continue;
        this._learnEdge(this.posTrigramTransitions, key, next, target - current, 8, 4096, 48);
        this.posTrigramUpdates++;
        triples++;
      }
      return { edges, triples, gain: g, cycles: repeats, target };
    }

    learnPOSSequence(tokens, options = {}) {
      const rows = Array.isArray(tokens) ? tokens : [];
      if (!rows.length) return {
        posUpdates: 0, trigramUpdates: 0, contextUpdates: 0,
        contextWindowUpdates: 0, wordPOSUpdates: 0, morphologyUpdates: 0
      };
      const gain = Math.max(0.05, Math.min(4, Number(options.gain) || 1));
      let prev2Word = '';
      let prev2POS = 'START2';
      let prevWord = '';
      let prevPOS = 'START';
      let posUpdates = 0;
      let trigramUpdates = 0;
      let contextUpdates = 0;
      let contextWindowUpdates = 0;
      let wordPOSUpdates = 0;
      let morphologyUpdates = 0;

      for (const token of rows) {
        const word = String(token?.lower || token?.surface || '').toLowerCase();
        if (!word) continue;
        const root = String(token?.root || '').toLowerCase();
        const state = {
          previousWord: prev2Word,
          previousPOS: prev2POS,
          lastWord: prevWord,
          lastPOS: prevPOS
        };
        const resolved = this.resolvePOS(token?.entry || token, state, { fallbackPOS: token?.pos });
        const pos = String(resolved.pos || token?.pos || 'Other');
        if (pos === 'Punct') continue;

        this._learnEdge(this.posTransitions, prevPOS, pos, gain, 8, 96, 48);
        this.posTransitionUpdates++; posUpdates++;

        this._learnEdge(this.posTrigramTransitions, `${prev2POS}>${prevPOS}`, pos, gain, 8, 4096, 48);
        this.posTrigramUpdates++; trigramUpdates++;

        this._learnEdge(this.wordPOSUsage, word, pos, gain, 8, this.options.maxContextSources, 16);
        this.wordPOSUpdates++; wordPOSUpdates++;

        if (prevWord) {
          const from = `${prevWord}#${prevPOS}`;
          const to = `${word}#${pos}`;
          this._learnEdge(this.contextTransitions, from, to, gain, 8, this.options.maxContextSources, this.options.maxContextTargets);
          this.contextTransitionUpdates++; contextUpdates++;
        }

        if (prev2Word && prevWord) {
          const from2 = `${prev2Word}#${prev2POS}|${prevWord}#${prevPOS}`;
          const to2 = `${word}#${pos}`;
          this._learnEdge(this.contextWindowTransitions, from2, to2, gain, 8, this.options.maxContextSources, this.options.maxContextTargets);
          this.contextWindowUpdates++; contextWindowUpdates++;
        }

        if (root) {
          this.learnMorphologyForm(root, word, pos, gain * 0.35);
          morphologyUpdates++;
        }
        prev2Word = prevWord;
        prev2POS = prevPOS;
        prevWord = word;
        prevPOS = pos;
      }

      if (options.countCycle !== false) {
        if (String(options.source || '') === 'background_replay') this.backgroundLearningCycles++;
        else this.selfLearningCycles++;
      }
      return { posUpdates, trigramUpdates, contextUpdates, contextWindowUpdates, wordPOSUpdates, morphologyUpdates };
    }


    seedLexicalPOS(entries, gain = 0.72) {
      const source = entries instanceof Map ? entries.values() : (Array.isArray(entries) ? entries : []);
      const g = Math.max(0.05, Math.min(2.0, Number(gain) || 0.72));
      let words = 0;
      let posEdges = 0;
      let morphologyEdges = 0;
      for (const entry of source) {
        if (!entry || entry.active === false) continue;
        const word = String(entry.lower || entry.surface || '').toLowerCase();
        if (!word) continue;
        const possible = Array.from(new Set((entry.possiblePOS || [entry.pos || 'Other']).map(String))).slice(0, 8);
        for (const pos of possible) {
          const currentPOS = this.wordPOSUsage.get(word)?.get(pos) || 0;
          if (currentPOS + 1e-6 < g) {
            this._learnEdge(this.wordPOSUsage, word, pos, g - currentPOS, 8, this.options.maxContextSources, 16);
            this.wordPOSUpdates++;
            posEdges++;
          }
          if (entry.root) {
            const root = String(entry.root).toLowerCase();
            const key = `${word}#${pos}`;
            const morphTarget = g * 0.28;
            const currentMorph = this.morphologyRelations.get(root)?.get(key) || 0;
            if (currentMorph + 1e-6 < morphTarget) {
              this._learnEdge(this.morphologyRelations, root, key, morphTarget - currentMorph, 6, this.options.maxMorphologyRoots, 48);
              this.morphologyUpdates++;
              morphologyEdges++;
            }
          }
        }
        words++;
      }
      return { words, posEdges, morphologyEdges, gain: g };
    }

    posTransitionScore(previousPOS, nextPOS) {
      const prev = String(previousPOS || 'START');
      const next = String(nextPOS || 'Other');
      return Math.min(1, (this.posTransitions.get(prev)?.get(next) || 0) / 2.2);
    }

    posTrigramScore(previous2POS, previousPOS, nextPOS) {
      const key = `${String(previous2POS || 'START2')}>${String(previousPOS || 'START')}`;
      return Math.min(1, (this.posTrigramTransitions.get(key)?.get(String(nextPOS || 'Other')) || 0) / 2.2);
    }

    wordPOSScore(word, pos) {
      const row = this.wordPOSUsage.get(String(word || '').toLowerCase());
      return Math.min(1, (row?.get(String(pos || 'Other')) || 0) / 2.2);
    }

    contextTransitionScore(previousWord, previousPOS, nextWord, nextPOS) {
      const prevWord = String(previousWord || '').toLowerCase();
      if (!prevWord) return 0;
      const from = `${prevWord}#${String(previousPOS || 'Other')}`;
      const to = `${String(nextWord || '').toLowerCase()}#${String(nextPOS || 'Other')}`;
      return Math.min(1, (this.contextTransitions.get(from)?.get(to) || 0) / 2.2);
    }

    contextWindowScore(previous2Word, previous2POS, previousWord, previousPOS, nextWord, nextPOS) {
      const p2 = String(previous2Word || '').toLowerCase();
      const p1 = String(previousWord || '').toLowerCase();
      if (!p2 || !p1) return 0;
      const from = `${p2}#${String(previous2POS || 'Other')}|${p1}#${String(previousPOS || 'Other')}`;
      const to = `${String(nextWord || '').toLowerCase()}#${String(nextPOS || 'Other')}`;
      return Math.min(1, (this.contextWindowTransitions.get(from)?.get(to) || 0) / 2.2);
    }

    resolvePOS(entry, state = {}, options = {}) {
      const word = String(entry?.lower || entry?.surface || '').toLowerCase();
      const possible = Array.from(new Set(
        (Array.isArray(entry?.possiblePOS) && entry.possiblePOS.length
          ? entry.possiblePOS
          : [options.fallbackPOS || entry?.pos || 'Other'])
          .map(x => String(x || 'Other'))
      ));
      const prev2POS = String(state?.previousPOS || 'START2');
      const prevPOS = String(state?.lastPOS || 'START');
      const prev2Word = String(state?.previousWord || '').toLowerCase();
      const prevWord = String(state?.lastWord || '').toLowerCase();

      if (possible.length === 1) {
        const pos = possible[0];
        const wordUsage = this.wordPOSScore(word, pos);
        const bigram = this.posTransitionScore(prevPOS, pos);
        const trigram = this.posTrigramScore(prev2POS, prevPOS, pos);
        const lexicalContext = this.contextTransitionScore(prevWord, prevPOS, word, pos);
        const windowContext = this.contextWindowScore(prev2Word, prev2POS, prevWord, prevPOS, word, pos);
        const morphology = this.morphologyScore(entry?.root || '', word, pos);
        const learned = wordUsage * 0.18 + bigram * 0.19 + trigram * 0.24 + lexicalContext * 0.20 + windowContext * 0.14 + morphology * 0.05;
        const hasEvidence = wordUsage + bigram + trigram + lexicalContext + windowContext + morphology > 1e-9;
        return {
          pos,
          certainty: hasEvidence ? Math.max(0.52, Math.min(1, 0.55 + learned * 0.45)) : 0.5,
          score: hasEvidence ? Math.max(0, Math.min(1, learned)) : 0.18,
          distribution: [{ pos, score: learned, wordUsage, bigram, trigram, lexicalContext, windowContext, morphology }]
        };
      }

      const primary = String(entry?.pos || possible[0]);
      const rows = [];
      let evidenceTotal = 0;

      for (const pos of possible) {
        const wordUsage = this.wordPOSScore(word, pos);
        const bigram = this.posTransitionScore(prevPOS, pos);
        const trigram = this.posTrigramScore(prev2POS, prevPOS, pos);
        const lexicalContext = this.contextTransitionScore(prevWord, prevPOS, word, pos);
        const windowContext = this.contextWindowScore(prev2Word, prev2POS, prevWord, prevPOS, word, pos);
        const morphology = this.morphologyScore(entry?.root || '', word, pos);
        const learned = wordUsage * 0.18 + bigram * 0.19 + trigram * 0.24 + lexicalContext * 0.20 + windowContext * 0.14 + morphology * 0.05;
        evidenceTotal += wordUsage + bigram + trigram + lexicalContext + windowContext + morphology;
        // Primary POS is only a cold-start tie breaker. Once learned evidence
        // exists, context can override it freely.
        const coldStart = pos === primary ? 0.035 : 0;
        rows.push({ pos, score: learned + coldStart, learned, wordUsage, bigram, trigram, lexicalContext, windowContext, morphology });
      }
      rows.sort((a, b) => b.score - a.score);
      const best = rows[0];
      const second = rows[1];
      if (evidenceTotal <= 1e-9) {
        const fallback = rows.find(x => x.pos === primary) || best;
        return { pos: fallback.pos, certainty: 0.5, score: 0.5, distribution: rows };
      }
      const margin = Math.max(0, best.score - (second?.score || 0));
      const certainty = Math.max(0.5, Math.min(1, 0.55 + margin * 1.7 + best.learned * 0.25));
      return { pos: best.pos, certainty, score: Math.max(0, Math.min(1, best.learned)), distribution: rows };
    }

    contextCandidates(previousWord, previousPOS, limit = 24) {
      const word = String(previousWord || '').toLowerCase();
      if (!word) return [];
      const row = this.contextTransitions.get(`${word}#${String(previousPOS || 'Other')}`);
      if (!row) return [];
      return Array.from(row.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.max(1, limit | 0))
        .map(([key, weight]) => {
          const split = key.lastIndexOf('#');
          return {
            word: split >= 0 ? key.slice(0, split) : key,
            pos: split >= 0 ? key.slice(split + 1) : 'Other',
            score: Math.min(1, weight / 2.2)
          };
        });
    }

    contextWindowCandidates(previous2Word, previous2POS, previousWord, previousPOS, limit = 24) {
      const p2 = String(previous2Word || '').toLowerCase();
      const p1 = String(previousWord || '').toLowerCase();
      if (!p2 || !p1) return [];
      const row = this.contextWindowTransitions.get(`${p2}#${String(previous2POS || 'Other')}|${p1}#${String(previousPOS || 'Other')}`);
      if (!row) return [];
      return Array.from(row.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.max(1, limit | 0))
        .map(([key, weight]) => {
          const split = key.lastIndexOf('#');
          return { word: split >= 0 ? key.slice(0, split) : key, pos: split >= 0 ? key.slice(split + 1) : 'Other', score: Math.min(1, weight / 2.2) };
        });
    }

    _weakenEdge(bucket, from, to, amount = 0.15) {
      const row = bucket.get(String(from || ''));
      if (!row) return 0;
      const key = String(to || '');
      const old = row.get(key) || 0;
      if (old <= 0) return 0;
      const next = old * Math.max(0.05, 1 - Math.max(0.01, Math.min(0.75, amount)));
      if (next < 0.025) row.delete(key); else row.set(key, next);
      if (!row.size) bucket.delete(String(from || ''));
      return old - next;
    }

    penalizePOSSequence(tokens, options = {}) {
      const rows = Array.isArray(tokens) ? tokens : [];
      const strength = Math.max(0.05, Math.min(2.5, Number(options.strength) || 0.5));
      const amount = Math.min(0.45, 0.06 + strength * 0.08);
      let prev2Word = '', prev2POS = 'START2', prevWord = '', prevPOS = 'START';
      let weakened = 0;
      for (const token of rows) {
        const word = String(token?.lower || token?.surface || '').toLowerCase();
        if (!word) continue;
        const resolved = this.resolvePOS(token?.entry || token, { previousWord: prev2Word, previousPOS: prev2POS, lastWord: prevWord, lastPOS: prevPOS }, { fallbackPOS: token?.pos });
        const pos = String(resolved.pos || token?.pos || 'Other');
        if (pos === 'Punct') continue;
        weakened += this._weakenEdge(this.posTransitions, prevPOS, pos, amount);
        weakened += this._weakenEdge(this.posTrigramTransitions, `${prev2POS}>${prevPOS}`, pos, amount);
        weakened += this._weakenEdge(this.wordPOSUsage, word, pos, amount * 0.7);
        if (prevWord) weakened += this._weakenEdge(this.contextTransitions, `${prevWord}#${prevPOS}`, `${word}#${pos}`, amount);
        if (prev2Word && prevWord) weakened += this._weakenEdge(this.contextWindowTransitions, `${prev2Word}#${prev2POS}|${prevWord}#${prevPOS}`, `${word}#${pos}`, amount);
        prev2Word = prevWord; prev2POS = prevPOS; prevWord = word; prevPOS = pos;
      }
      return { weakened, amount };
    }

    penalizeTextTransitions(text, processor, options = {}) {
      const tokens = processor?.tokenize?.(String(text || ''), 192) || [];
      const strength = Math.max(0.05, Math.min(2.5, Number(options.strength) || 0.5));
      const amount = Math.min(0.40, 0.05 + strength * 0.07);
      let prev = '';
      let weakened = 0;
      for (const token of tokens) {
        if (token.pos === 'Punct') continue;
        const word = String(token.lower || '').toLowerCase();
        if (prev && word) weakened += this._weakenEdge(this.transitions, prev, word, amount);
        prev = word;
      }
      return { weakened, amount };
    }

    learnCorrelation(fromWord, toWord, options = {}) {
      const from = String(fromWord || '').toLowerCase().trim();
      const to = String(toWord || '').toLowerCase().trim();
      if (!from || !to || from === to) return 0;
      const gain = Math.max(0.02, Math.min(3, Number(options.gain) || 1));
      const weight = this._learnEdge(
        this.wordCorrelations,
        from,
        to,
        gain,
        8,
        this.options.maxCorrelationSources,
        this.options.maxCorrelationTargets
      );
      this.correlationUpdates++;
      if (options.symmetric) {
        const reverseGain = Math.max(0.02, Math.min(1, Number(options.reverseGain) || 0.75));
        this._learnEdge(
          this.wordCorrelations,
          to,
          from,
          gain * reverseGain,
          8,
          this.options.maxCorrelationSources,
          this.options.maxCorrelationTargets
        );
        this.correlationUpdates++;
      }
      return Math.min(1, weight / 2.4);
    }

    learnPOSCorrelation(fromPOS, toPOS, options = {}) {
      const from = String(fromPOS || 'Other');
      const to = String(toPOS || 'Other');
      if (!from || !to) return 0;
      const distance = Math.max(1, Number(options.distance) || 1);
      const gain = Math.max(0.02, Math.min(2, Number(options.gain) || 1)) / Math.sqrt(distance);
      const weight = this._learnEdge(this.posCorrelations, from, to, gain, 8, 96, 64);
      this.posCorrelationUpdates++;
      if (options.symmetric) {
        this._learnEdge(this.posCorrelations, to, from, gain * 0.72, 8, 96, 64);
        this.posCorrelationUpdates++;
      }
      return Math.min(1, weight / 2.4);
    }

    correlationScore(seedWord, candidateWord) {
      const seed = String(seedWord || '').toLowerCase().trim();
      const candidate = String(candidateWord || '').toLowerCase().trim();
      if (!seed || !candidate || seed === candidate) return 0;
      return Math.min(1, (this.wordCorrelations.get(seed)?.get(candidate) || 0) / 2.4);
    }

    correlationSupport(seedWords, candidateWord) {
      const candidate = String(candidateWord || '').toLowerCase().trim();
      const seeds = Array.from(new Set((Array.isArray(seedWords) ? seedWords : [seedWords])
        .map(x => String(x || '').toLowerCase().trim()).filter(Boolean)));
      if (!candidate || !seeds.length) return 0;
      let sum = 0;
      let max = 0;
      let count = 0;
      for (const seed of seeds.slice(0, 16)) {
        if (seed === candidate) continue;
        const score = this.correlationScore(seed, candidate);
        if (score > 0) {
          sum += score;
          max = Math.max(max, score);
          count++;
        }
      }
      if (!count) return 0;
      return Math.min(1, max * 0.62 + (sum / count) * 0.38);
    }

    correlationCandidates(seedWords, limit = 32) {
      const seeds = Array.from(new Set((Array.isArray(seedWords) ? seedWords : [seedWords])
        .map(x => String(x || '').toLowerCase().trim()).filter(Boolean)));
      const scores = new Map();
      for (const seed of seeds.slice(0, 16)) {
        const row = this.wordCorrelations.get(seed);
        if (!row) continue;
        for (const [word, weight] of row.entries()) {
          if (!word || seeds.includes(word)) continue;
          scores.set(word, (scores.get(word) || 0) + Math.min(1, weight / 2.4));
        }
      }
      return Array.from(scores.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.max(1, limit | 0))
        .map(([word, score]) => ({ word, score: Math.min(1, score / Math.max(1, Math.min(4, seeds.length))) }));
    }

    penalizeCorrelationText(text, processor, options = {}) {
      const tokens = (processor?.tokenize?.(String(text || ''), 192) || []).filter(token => token.pos !== 'Punct');
      const strength = Math.max(0.03, Math.min(0.7, Number(options.strength) || 0.15));
      let weakened = 0;
      for (let i = 0; i < tokens.length; i++) {
        const from = String(tokens[i]?.lower || '').toLowerCase();
        if (!from) continue;
        for (let j = i + 1; j < Math.min(tokens.length, i + 5); j++) {
          const to = String(tokens[j]?.lower || '').toLowerCase();
          if (!to || to === from) continue;
          const amount = strength / Math.max(1, j - i);
          weakened += this._weakenEdge(this.wordCorrelations, from, to, amount);
          weakened += this._weakenEdge(this.wordCorrelations, to, from, amount * 0.5);
        }
      }
      return { weakened, strength };
    }

    learnMorphologyForm(root, surface, pos, gain = 1) {
      root = String(root || '').toLowerCase();
      surface = String(surface || '').toLowerCase();
      pos = String(pos || 'Other');
      if (!root || !surface) return 0;
      const weight = this._learnEdge(
        this.morphologyRelations,
        root,
        `${surface}#${pos}`,
        gain,
        6,
        this.options.maxMorphologyRoots,
        48
      );
      this.morphologyUpdates++;
      return Math.min(1, weight / 2.0);
    }

    morphologyScore(root, surface, pos) {
      root = String(root || '').toLowerCase();
      if (!root) return 0;
      const key = `${String(surface || '').toLowerCase()}#${String(pos || 'Other')}`;
      return Math.min(1, (this.morphologyRelations.get(root)?.get(key) || 0) / 2.0);
    }

    structureSupport(text, processor) {
      const tokens = processor?.tokenize?.(String(text || ''), 192) || [];
      let prev2Word = '';
      let prev2POS = 'START2';
      let prevWord = '';
      let prevPOS = 'START';
      let sum = 0;
      let count = 0;
      for (const token of tokens) {
        if (token.pos === 'Punct') continue;
        const state = { previousWord: prev2Word, previousPOS: prev2POS, lastWord: prevWord, lastPOS: prevPOS };
        const resolved = this.resolvePOS(token.entry || token, state, { fallbackPOS: token.pos });
        const pos = resolved.pos || token.pos;
        const posScore = this.posTransitionScore(prevPOS, pos);
        const trigram = this.posTrigramScore(prev2POS, prevPOS, pos);
        const wordPOS = this.wordPOSScore(token.lower, pos);
        const context = prevWord ? this.contextTransitionScore(prevWord, prevPOS, token.lower, pos) : 0;
        const window = prev2Word && prevWord ? this.contextWindowScore(prev2Word, prev2POS, prevWord, prevPOS, token.lower, pos) : 0;
        const morphology = this.morphologyScore(token.root, token.lower, pos);
        sum += posScore * 0.20 + trigram * 0.24 + wordPOS * 0.14 + context * 0.19 + window * 0.17 + morphology * 0.06;
        count++;
        prev2Word = prevWord; prev2POS = prevPOS; prevWord = token.lower; prevPOS = pos;
      }
      return count ? Math.max(0, Math.min(1, sum / count)) : 0;
    }

    learnedCandidates(previousWord, _variant = 'normal', limit = 24) {
      const previous = String(previousWord || '').toLowerCase();
      if (!previous) return [];
      const row = this.transitions.get(previous);
      if (!row) return [];
      return Array.from(row.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, Math.max(1, limit | 0))
        .map(([word, weight]) => ({ word, score: Math.min(1, weight / 2.0) }));
    }

    transitionScore(previousWord, candidateWord, _variant = 'normal') {
      const prev = String(previousWord || '').toLowerCase();
      const next = String(candidateWord || '').toLowerCase();
      if (!prev || !next) return 0;
      return Math.min(1, (this.transitions.get(prev)?.get(next) || 0) / 2.0);
    }

    learnTextTransitions(text, options = {}) {
      text = String(text || '');
      const gain = Math.max(0.1, Math.min(4.0, Number(options.gain) || 1.0));
      let previous = '';
      let tokenStart = -1;
      let learned = 0;
      for (let i = 0; i <= text.length; i++) {
        const code = i < text.length ? text.charCodeAt(i) : 32;
        const word = i < text.length && isWordCode(code);
        if (word) {
          if (tokenStart < 0) tokenStart = i;
        } else if (tokenStart >= 0) {
          if (i > tokenStart) {
            const current = text.slice(tokenStart, i).toLowerCase();
            if (previous) {
              this._learnTransition(previous, current, options.source || 'self_learning', gain);
              learned++;
            }
            previous = current;
          }
          tokenStart = -1;
        }
      }
      if (options.selfLearning !== false) this.selfLearningCycles++;
      this.lastSelfLearningSupport = this.sequenceSupport(text);
      return {
        learnedTransitions: learned,
        support: this.lastSelfLearningSupport,
        cycles: this.selfLearningCycles
      };
    }

    sequenceSupport(text) {
      text = String(text || '');
      let previous = '';
      let tokenStart = -1;
      let sum = 0;
      let count = 0;
      for (let i = 0; i <= text.length; i++) {
        const code = i < text.length ? text.charCodeAt(i) : 32;
        const word = i < text.length && isWordCode(code);
        if (word) {
          if (tokenStart < 0) tokenStart = i;
        } else if (tokenStart >= 0) {
          if (i > tokenStart) {
            const current = text.slice(tokenStart, i).toLowerCase();
            if (previous) {
              sum += this.transitionScore(previous, current);
              count++;
            }
            previous = current;
          }
          tokenStart = -1;
        }
      }
      return count ? Math.max(0, Math.min(1, sum / count)) : 0.5;
    }

    learnSemanticFrame(frame, options = {}) {
      if (!frame) return 0;
      const subject = String(frame.subject || '').toLowerCase().trim();
      if (!subject) return 0;
      const terms = [];
      for (const field of ['relation', 'complement', 'mechanism', 'capability']) {
        const text = String(frame[field] || '').toLowerCase();
        const found = text.match(/[a-z0-9]+(?:['-][a-z0-9]+)*/g) || [];
        for (const token of found) if (token.length > 2) terms.push(token);
      }
      let row = this.semanticMemory.get(subject);
      if (!row) {
        row = new Map();
        this.semanticMemory.set(subject, row);
      }
      const gain = Math.max(0.25, Math.min(3.0, Number(options.gain) || 1.0));
      for (const token of terms.slice(0, 64)) {
        row.set(token, Math.min(5, (row.get(token) || 0) * 0.99 + 0.18 * gain));
        this.semanticUpdates++;
      }
      return terms.length;
    }

    semanticCandidates(subject, _variant = 'normal', limit = 24) {
      const row = this.semanticMemory.get(String(subject || '').toLowerCase().trim());
      if (!row) return [];
      return Array.from(row.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([word, weight]) => ({ word, score: Math.min(1, weight / 2) }));
    }

    exportLearningState() {
      const packBucket = bucket => Array.from(bucket.entries())
        .map(([prev, row]) => [prev, Array.from(row.entries())]);
      return {
        schemaVersion: 8,
        savedAt: Date.now(),
        transitions: packBucket(this.transitions),
        posTransitions: packBucket(this.posTransitions),
        posTrigramTransitions: packBucket(this.posTrigramTransitions),
        contextTransitions: packBucket(this.contextTransitions),
        contextWindowTransitions: packBucket(this.contextWindowTransitions),
        wordPOSUsage: packBucket(this.wordPOSUsage),
        morphologyRelations: packBucket(this.morphologyRelations),
        wordCorrelations: packBucket(this.wordCorrelations),
        posCorrelations: packBucket(this.posCorrelations),
        transitionUpdates: this.transitionUpdates,
        posTransitionUpdates: this.posTransitionUpdates,
        posTrigramUpdates: this.posTrigramUpdates,
        contextTransitionUpdates: this.contextTransitionUpdates,
        contextWindowUpdates: this.contextWindowUpdates,
        wordPOSUpdates: this.wordPOSUpdates,
        morphologyUpdates: this.morphologyUpdates,
        correlationUpdates: this.correlationUpdates,
        posCorrelationUpdates: this.posCorrelationUpdates,
        selfLearningCycles: this.selfLearningCycles,
        backgroundLearningCycles: this.backgroundLearningCycles
      };
    }

    importLearningState(state) {
      if (!state || ![3, 4, 5, 6, 7, 8].includes(Number(state.schemaVersion))) return false;
      const unpackInto = (target, rows, merge = false, maxSources = this.options.maxTransitionSources) => {
        if (!merge) target.clear();
        for (const item of Array.isArray(rows) ? rows : []) {
          const key = String(item?.[0] || '');
          if (!key) continue;
          let row = target.get(key);
          if (!row) {
            row = new Map();
            target.set(key, row);
          }
          for (const pair of Array.isArray(item?.[1]) ? item[1] : []) {
            const word = String(pair?.[0] || '');
            const weight = Number(pair?.[1] || 0);
            if (!word || !Number.isFinite(weight) || weight <= 0) continue;
            row.set(word, Math.max(row.get(word) || 0, Math.min(8, weight)));
          }
          if (target.size >= maxSources) break;
        }
      };

      this.transitions.clear();
      this.posTransitions.clear();
      this.posTrigramTransitions.clear();
      this.contextTransitions.clear();
      this.contextWindowTransitions.clear();
      this.wordPOSUsage.clear();
      this.morphologyRelations.clear();
      this.wordCorrelations.clear();
      this.posCorrelations.clear();
      this.semanticMemory.clear();

      if (Number(state.schemaVersion) === 7) {
        unpackInto(this.transitions, state.transitions, false);
        unpackInto(this.posTransitions, state.posTransitions, false, 96);
        unpackInto(this.posTrigramTransitions, state.posTrigramTransitions, false, 4096);
        unpackInto(this.contextTransitions, state.contextTransitions, false, this.options.maxContextSources);
        unpackInto(this.contextWindowTransitions, state.contextWindowTransitions, false, this.options.maxContextSources);
        unpackInto(this.wordPOSUsage, state.wordPOSUsage, false, this.options.maxContextSources);
        unpackInto(this.morphologyRelations, state.morphologyRelations, false, this.options.maxMorphologyRoots);
        unpackInto(this.wordCorrelations, state.wordCorrelations, false, this.options.maxCorrelationSources);
        unpackInto(this.posCorrelations, state.posCorrelations, false, 96);
      } else if (Number(state.schemaVersion) === 6) {
        unpackInto(this.transitions, state.transitions, false);
        unpackInto(this.posTransitions, state.posTransitions, false, 96);
        unpackInto(this.posTrigramTransitions, state.posTrigramTransitions, false, 4096);
        unpackInto(this.contextTransitions, state.contextTransitions, false, this.options.maxContextSources);
        unpackInto(this.contextWindowTransitions, state.contextWindowTransitions, false, this.options.maxContextSources);
        unpackInto(this.wordPOSUsage, state.wordPOSUsage, false, this.options.maxContextSources);
        unpackInto(this.morphologyRelations, state.morphologyRelations, false, this.options.maxMorphologyRoots);
      } else if (Number(state.schemaVersion) === 5) {
        unpackInto(this.transitions, state.transitions, false);
        unpackInto(this.posTransitions, state.posTransitions, false, 96);
        unpackInto(this.contextTransitions, state.contextTransitions, false, this.options.maxContextSources);
        unpackInto(this.morphologyRelations, state.morphologyRelations, false, this.options.maxMorphologyRoots);
      } else if (Number(state.schemaVersion) === 4) {
        unpackInto(this.transitions, state.transitions, false);
        // Older semantic memory is intentionally not promoted into the new
        // POS/context learner. It remains absent instead of becoming facts.
      } else {
        unpackInto(this.transitions, state.sharedTransitions, true);
        unpackInto(this.transitions, state.standardTransitions, true);
        unpackInto(this.transitions, state.ultraTransitions, true);
      }

      this.transitionUpdates = Math.max(0, Number(state.transitionUpdates) || 0);
      this.posTransitionUpdates = Math.max(0, Number(state.posTransitionUpdates) || 0);
      this.posTrigramUpdates = Math.max(0, Number(state.posTrigramUpdates) || 0);
      this.contextTransitionUpdates = Math.max(0, Number(state.contextTransitionUpdates) || 0);
      this.contextWindowUpdates = Math.max(0, Number(state.contextWindowUpdates) || 0);
      this.wordPOSUpdates = Math.max(0, Number(state.wordPOSUpdates) || 0);
      this.morphologyUpdates = Math.max(0, Number(state.morphologyUpdates) || 0);
      this.correlationUpdates = Math.max(0, Number(state.correlationUpdates) || 0);
      this.posCorrelationUpdates = Math.max(0, Number(state.posCorrelationUpdates) || 0);
      this.semanticUpdates = 0;
      this.selfLearningCycles = Math.max(0, Number(state.selfLearningCycles) || 0);
      this.backgroundLearningCycles = Math.max(0, Number(state.backgroundLearningCycles) || 0);
      return true;
    }


    _seenRecently(hash) {
      const count = this.recentHashCount;
      const cap = this.recentHashes.length;
      for (let i = 0; i < count; i++) {
        const idx = (this.recentHashCursor - 1 - i + cap) % cap;
        if (this.recentHashes[idx] === hash) return true;
      }
      return false;
    }

    _rememberHash(hash) {
      this.recentHashes[this.recentHashCursor] = hash >>> 0;
      this.recentHashCursor = (this.recentHashCursor + 1) % this.recentHashes.length;
      this.recentHashCount = Math.min(this.recentHashes.length, this.recentHashCount + 1);
    }

    _encodeAndQueue(text, start, end, step) {
      let h1 = 2166136261 >>> 0;
      let h2 = 0x9e3779b9 >>> 0;
      let vowels = 0;
      let upper = 0;

      for (let i = start; i < end; i++) {
        let code = text.charCodeAt(i);
        if (code >= 65 && code <= 90) { upper++; code += 32; }
        h1 ^= code;
        h1 = Math.imul(h1, 16777619) >>> 0;
        h2 ^= (code + 0x7ed55d16 + (h2 << 12)) >>> 0;
        h2 = (h2 ^ (h2 >>> 19)) >>> 0;
        if (code === 97 || code === 101 || code === 105 || code === 111 || code === 117 || code === 121) vowels++;
      }

      const length = Math.max(1, end - start);
      h2 ^= Math.imul(length, 0x85ebca6b) >>> 0;
      h2 ^= Math.imul(vowels + 1, 0xc2b2ae35) >>> 0;
      h2 ^= Math.imul(upper + 1, 0x27d4eb2f) >>> 0;

      const relationHash = (h1 ^ Math.imul(this.lastHash || 0x811c9dc5, 2246822519)) >>> 0;
      const lengthSignal = Math.min(1, length / 12);
      const vowelRatio = Math.min(1, vowels / Math.max(1, length));
      const s0 = h1 % this.stateDim;
      const s1 = h2 % this.stateDim;
      const s2 = ((h1 >>> 16) ^ h2) % this.stateDim;
      const s3 = relationHash % this.stateDim;
      const s4 = (Math.imul(h1 ^ h2, 2654435761) >>> 0) % this.stateDim;
      const s5 = ((h2 >>> 11) + step * 17 + length * 13) % this.stateDim;
      const sign1 = (h1 & 1) ? 1 : -1;
      const sign2 = (h2 & 1) ? 1 : -1;
      const w0 = 0.95 * sign1;
      const w1 = 0.78 * sign2;
      const w2 = 0.58 * (sign1 * sign2);
      const w3 = 0.46;
      const w4 = 0.30 + lengthSignal * 0.28;
      const w5 = 0.18 + vowelRatio * 0.30;

      const base = step * this.options.receptiveSlotsPerToken;
      this.batchIds[base] = s0; this.batchWeights[base] = w0;
      this.batchIds[base + 1] = s1; this.batchWeights[base + 1] = w1;
      this.batchIds[base + 2] = s2; this.batchWeights[base + 2] = w2;
      this.batchIds[base + 3] = s3; this.batchWeights[base + 3] = w3;
      this.batchIds[base + 4] = s4; this.batchWeights[base + 4] = w4;
      this.batchIds[base + 5] = s5; this.batchWeights[base + 5] = w5;
      this.recentSlotIds[0] = s0; this.recentSlotWeights[0] = w0;
      this.recentSlotIds[1] = s1; this.recentSlotWeights[1] = w1;
      this.recentSlotIds[2] = s2; this.recentSlotWeights[2] = w2;
      this.recentSlotIds[3] = s3; this.recentSlotWeights[3] = w3;
      this.recentSlotIds[4] = s4; this.recentSlotWeights[4] = w4;
      this.recentSlotIds[5] = s5; this.recentSlotWeights[5] = w5;

      const repeated = this._seenRecently(h1);
      this._rememberHash(h1);
      this.lastHash = h1;
      return repeated;
    }

    async observeText(text, options = {}) {
      text = String(text || '');
      const startTime = performance.now();
      const maxSteps = Math.max(1, Math.min(
        this.options.maxLexicalStepsPerTurn,
        Number.isFinite(options.maxSteps) ? options.maxSteps | 0 : this.options.maxLexicalStepsPerTurn
      ));

      let lexicalSteps = 0;
      let repeated = 0;
      let tokenStart = -1;
      let previousLearnedWord = '';
      const learningSource = String(options.source || 'user');
      const learningEnabled = options.learnTransitions !== false;

      for (let i = 0; i <= text.length && lexicalSteps < maxSteps; i++) {
        const code = i < text.length ? text.charCodeAt(i) : 32;
        const word = i < text.length && isWordCode(code);

        if (word) {
          if (tokenStart < 0) tokenStart = i;
        } else if (tokenStart >= 0) {
          if (i > tokenStart) {
            if (this._encodeAndQueue(text, tokenStart, i, lexicalSteps)) repeated++;
            if (learningEnabled) {
              const currentLearnedWord = text.slice(tokenStart, i).toLowerCase();
              if (previousLearnedWord) this._learnTransition(previousLearnedWord, currentLearnedWord, learningSource, options.learningGain || 1.0);
              previousLearnedWord = currentLearnedWord;
            }
            lexicalSteps++;
          }
          tokenStart = -1;
        }
      }

      if (lexicalSteps > 0) {
        this.compute.recurrentBatch({
          ids: this.batchIds,
          weights: this.batchWeights,
          count: lexicalSteps,
          decay: this.options.decay,
          inputGain: this.options.inputGain,
          inhibition: this.options.inhibition,
          baselinePull: this.options.baselinePull,
          tick: this.totalLexicalSteps
        });
      }

      const novelty = lexicalSteps > 0
        ? Math.max(0, Math.min(1, 1 - repeated / lexicalSteps))
        : 0;

      if (options.countTurn !== false) this.turnCount++;
      this.totalLexicalSteps += lexicalSteps;
      this.lastLexicalSteps = lexicalSteps;
      this.lastCharacters = text.length;
      this.lastNovelty = novelty;
      this.noveltyEMA = this.noveltyEMA * 0.82 + novelty * 0.18;
      const stepPressure = Math.min(1, lexicalSteps / Math.max(1, maxSteps));
      this.momentumEMA = Math.min(1, this.momentumEMA * 0.78 + stepPressure * 0.22 + novelty * 0.08);
      this.lastHotPathMs = performance.now() - startTime;

      let energy = null;
      this.lastSyncMs = 0;
      if (options.syncState === true) {
        const syncStart = performance.now();
        energy = await this.compute.syncStateToCPU();
        this.lastSyncMs = performance.now() - syncStart;
      }

      this.trace?.push?.(
        globalThis.VilotTraceEvent?.RSL_OBSERVE || 18,
        this.lastHotPathMs,
        lexicalSteps,
        novelty,
        this.momentumEMA
      );

      return {
        accepted: lexicalSteps > 0,
        lexicalSteps,
        characters: text.length,
        repeatedLexicalSignals: repeated,
        novelty,
        noveltyEMA: this.noveltyEMA,
        momentum: this.momentumEMA,
        hotPathMs: this.lastHotPathMs,
        stateSyncMs: this.lastSyncMs,
        stateRevision: this.arena.revision,
        totalLexicalSteps: this.totalLexicalSteps,
        turnCount: this.turnCount,
        energy,
        recentSlots: this._recentSlotsSnapshot(),
        note: 'RSL state advanced autoregressively with ordered recurrent batching and is available to Decoder.js as live-state evidence.'
      };
    }

    _hashTextWord(word) {
      const text = String(word || '');
      let h1 = 2166136261 >>> 0;
      let h2 = 0x9e3779b9 >>> 0;
      let length = 0;
      for (let i = 0; i < text.length; i++) {
        let code = text.charCodeAt(i);
        if (code >= 65 && code <= 90) code += 32;
        if (!isWordCode(code)) continue;
        length++;
        h1 ^= code;
        h1 = Math.imul(h1, 16777619) >>> 0;
        h2 ^= (code + 0x7ed55d16 + (h2 << 12)) >>> 0;
        h2 = (h2 ^ (h2 >>> 19)) >>> 0;
      }
      h2 ^= Math.imul(Math.max(1, length), 0x85ebca6b) >>> 0;
      return { h1, h2, length };
    }

    activationForWord(word) {
      const { h1, h2, length } = this._hashTextWord(word);
      if (!length) return 0.5;
      const slots = [
        h1 % this.stateDim,
        h2 % this.stateDim,
        ((h1 >>> 16) ^ h2) % this.stateDim,
        (Math.imul(h1 ^ h2, 2654435761) >>> 0) % this.stateDim
      ];

      // CPU fallback can read the recurrent state directly. Under WebGPU the
      // CPU mirror is intentionally stale, so use recent receptive-slot
      // agreement instead of forcing a latency-destroying readback.
      if (this.compute?.mode !== 'webgpu') {
        let sum = 0;
        for (const slot of slots) sum += Math.abs(this.arena.readState[slot] || 0);
        return Math.max(0, Math.min(1, 0.35 + (sum / slots.length) * 0.9));
      }

      let matches = 0;
      for (const slot of slots) {
        for (let j = 0; j < 6; j++) {
          if (this.recentSlotIds[j] === slot) { matches++; break; }
        }
      }
      const recentlySeen = this._seenRecently(h1) ? 1 : 0;
      return Math.max(0, Math.min(1, 0.34 + matches * 0.12 + recentlySeen * 0.18 + this.momentumEMA * 0.08));
    }

    _recentSlotsSnapshot() {
      const out = [];
      for (let i = 0; i < 6; i++) {
        out.push({
          slot: this.recentSlotIds[i],
          weight: Number(this.recentSlotWeights[i].toFixed(4))
        });
      }
      return out;
    }

    backgroundTick(phase) {
      const factor = phase === 'IDLE' ? 0.94 : phase === 'WARM' ? 0.975 : 0.992;
      this.momentumEMA *= factor;
      this.noveltyEMA = 0.5 + (this.noveltyEMA - 0.5) * factor;
    }

    checkpoint() {
      this.checkpointHashes.set(this.recentHashes);
      this.checkpointHashCursor = this.recentHashCursor;
      this.checkpointHashCount = this.recentHashCount;
      this.checkpointScalars[0] = this.turnCount;
      this.checkpointScalars[1] = this.totalLexicalSteps;
      this.checkpointScalars[2] = this.lastLexicalSteps;
      this.checkpointScalars[3] = this.lastNovelty;
      this.checkpointScalars[4] = this.noveltyEMA;
      this.checkpointScalars[5] = this.momentumEMA;
      this.checkpointScalars[6] = this.lastHash;
      this.checkpointScalars[7] = this.lastCharacters;
    }

    restoreCheckpoint() {
      this.recentHashes.set(this.checkpointHashes);
      this.recentHashCursor = this.checkpointHashCursor;
      this.recentHashCount = this.checkpointHashCount;
      this.turnCount = this.checkpointScalars[0] | 0;
      this.totalLexicalSteps = this.checkpointScalars[1] | 0;
      this.lastLexicalSteps = this.checkpointScalars[2] | 0;
      this.lastNovelty = this.checkpointScalars[3];
      this.noveltyEMA = this.checkpointScalars[4];
      this.momentumEMA = this.checkpointScalars[5];
      this.lastHash = this.checkpointScalars[6] >>> 0;
      this.lastCharacters = this.checkpointScalars[7] | 0;
    }

    reset() {
      this.recentHashes.fill(0);
      this.checkpointHashes.fill(0);
      this.recentSlotIds.fill(0);
      this.recentSlotWeights.fill(0);
      this.recentHashCursor = 0;
      this.recentHashCount = 0;
      this.checkpointHashCursor = 0;
      this.checkpointHashCount = 0;
      this.turnCount = 0;
      this.totalLexicalSteps = 0;
      this.lastLexicalSteps = 0;
      this.lastCharacters = 0;
      this.lastNovelty = 1;
      this.noveltyEMA = 1;
      this.momentumEMA = 0;
      this.lastHotPathMs = 0;
      this.lastSyncMs = 0;
      this.lastHash = 0;
      this.checkpointScalars.fill(0);
      this.transitions.clear();
      this.posTransitions.clear();
      this.posTrigramTransitions.clear();
      this.contextTransitions.clear();
      this.contextWindowTransitions.clear();
      this.wordPOSUsage.clear();
      this.morphologyRelations.clear();
      this.wordCorrelations.clear();
      this.posCorrelations.clear();
      this.semanticMemory.clear();
      this.transitionUpdates = 0;
      this.posTransitionUpdates = 0;
      this.posTrigramUpdates = 0;
      this.contextTransitionUpdates = 0;
      this.contextWindowUpdates = 0;
      this.wordPOSUpdates = 0;
      this.morphologyUpdates = 0;
      this.correlationUpdates = 0;
      this.posCorrelationUpdates = 0;
      this.semanticUpdates = 0;
      this.selfLearningCycles = 0;
      this.backgroundLearningCycles = 0;
      this.lastSelfLearningSupport = 0;
    }

    status() {
      return {
        version: 2,
        stateModel: 'dynamic-Ht-autoregressive',
        stateDim: this.stateDim,
        turnCount: this.turnCount,
        totalLexicalSteps: this.totalLexicalSteps,
        lastLexicalSteps: this.lastLexicalSteps,
        lastCharacters: this.lastCharacters,
        lastNovelty: this.lastNovelty,
        noveltyEMA: this.noveltyEMA,
        momentum: this.momentumEMA,
        lastHotPathMs: this.lastHotPathMs,
        lastSyncMs: this.lastSyncMs,
        recentHashCount: this.recentHashCount,
        recentSlots: this._recentSlotsSnapshot(),
        learning: {
          transitionSources: this.transitions.size,
          posTransitionSources: this.posTransitions.size,
          posTrigramSources: this.posTrigramTransitions.size,
          contextTransitionSources: this.contextTransitions.size,
          contextWindowSources: this.contextWindowTransitions.size,
          wordPOSWords: this.wordPOSUsage.size,
          morphologyRoots: this.morphologyRelations.size,
          correlationSources: this.wordCorrelations.size,
          posCorrelationSources: this.posCorrelations.size,
          transitionUpdates: this.transitionUpdates,
          posTransitionUpdates: this.posTransitionUpdates,
          posTrigramUpdates: this.posTrigramUpdates,
          contextTransitionUpdates: this.contextTransitionUpdates,
          contextWindowUpdates: this.contextWindowUpdates,
          wordPOSUpdates: this.wordPOSUpdates,
          morphologyUpdates: this.morphologyUpdates,
          correlationUpdates: this.correlationUpdates,
          posCorrelationUpdates: this.posCorrelationUpdates,
          selfLearningCycles: this.selfLearningCycles,
          backgroundLearningCycles: this.backgroundLearningCycles,
          lastSelfLearningSupport: this.lastSelfLearningSupport,
          unifiedVilotNI2: true
        }
      };
    }
  }

  globalThis.VilotRSL = VilotRSL;
})();
