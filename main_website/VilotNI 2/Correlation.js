/**
 * VilotNI 2 - Correlation.js
 *
 * Production continuous structural, lexical, correlation, morphology and
 * curiosity learning. This file intentionally combines the former misspelled
 * structural-learning role with Correlation.js so there is one background learner.
 *
 * Correlation does not contain canned answers or hand-written subject facts.
 * It studies observed language, failed and accepted trajectories, POS behavior,
 * morphology, prompt/response alignment and safe alternative response paths.
 */
(() => {
  'use strict';

  const clamp01 = value => Math.max(0, Math.min(1, Number(value) || 0));

  class VilotCorrelation {
    constructor(rsl, processor, memory, trace = null, options = {}, lexical = null) {
      this.rsl = rsl;
      this.processor = processor;
      this.memory = memory;
      this.trace = trace;
      this.lexical = lexical || {};
      this.options = {
        window: Math.max(2, Math.min(10, options.correlationWindow | 0 || 4)),
        activeStudyUnits: Math.max(2, options.correlationActiveStudyUnits | 0 || 6),
        warmStudyUnits: Math.max(4, options.correlationWarmStudyUnits | 0 || 28),
        idleStudyUnits: Math.max(8, options.correlationIdleStudyUnits | 0 || 96),
        morphologyEvery: Math.max(1, options.correlationMorphologyEvery | 0 || 3),
        curiosityEvery: Math.max(1, options.correlationCuriosityEvery | 0 || 2),
        maxTickMs: Number.isFinite(options.correlationMaxTickMs)
          ? Math.max(1, options.correlationMaxTickMs)
          : 6.5,
        maxBurstMs: Number.isFinite(options.correlationMaxBurstMs)
          ? Math.max(0.75, options.correlationMaxBurstMs)
          : 3.0,
        replayGain: Number.isFinite(options.correlationReplayGain)
          ? options.correlationReplayGain
          : 0.32,
        morphologyGain: Number.isFinite(options.correlationMorphologyGain)
          ? options.correlationMorphologyGain
          : 0.22,
        crossTurnGain: Number.isFinite(options.correlationCrossTurnGain) ? options.correlationCrossTurnGain : 0.34,
        failurePenalty: Number.isFinite(options.correlationFailurePenalty) ? options.correlationFailurePenalty : 0.20,
        curiosityEnabled: options.curiosityEnabled !== false,
        curiosityWeight: Number.isFinite(options.curiosityWeight) ? Math.max(0, Math.min(0.35, options.curiosityWeight)) : 0.14,
        curiosityMaxConfidenceDrop: Number.isFinite(options.curiosityMaxConfidenceDrop)
          ? Math.max(0, Math.min(8, options.curiosityMaxConfidenceDrop))
          : 3.25,
        curiosityQueueMax: Math.max(32, options.curiosityQueueMax | 0 || 256),
        curiosityStudyGain: Number.isFinite(options.curiosityStudyGain)
          ? Math.max(0.03, Math.min(0.5, options.curiosityStudyGain))
          : 0.16
      };

      this.tickCount = 0;
      this.studyCycles = 0;
      this.replayCycles = 0;
      this.crossTurnCycles = 0;
      this.failureCycles = 0;
      this.morphologyCycles = 0;
      this.observedTexts = 0;
      this.observedPairs = 0;
      this.observedEpisodes = 0;
      this.tokensStudied = 0;
      this.pairUpdates = 0;
      this.lastPairUpdates = 0;
      this.lastStudyMs = 0;
      this.lastReplayIndex = 0;
      this.lastEpisodeId = 0;
      this.lastConfidence = 0;
      this.rootCursor = 0;
      this.rootKeys = [];

      this.curiosityQueue = [];
      this.curiosityQueuedKeys = new Set();
      this.curiositySelections = 0;
      this.curiosityAlternativeSelections = 0;
      this.curiosityStudyCycles = 0;
      this.curiosityRejected = 0;
      this.curiosityCounter = 0;
      this.lastCuriosity = null;
      this.responseVisits = new Map();
      this.routeVisits = new Map();
      this.lastAnswerForPrompt = new Map();

      this.refreshLexicalStudyIndex();
      this.posCurriculumSeed = this.rsl?.seedPOSCurriculum?.(
        this.processor?.posTypeRelations || {},
        this.processor?.posTypeTrigrams || [],
        Number.isFinite(options.correlationPOSSeedGain)
          ? options.correlationPOSSeedGain
          : 0.82
      ) || { edges: 0, triples: 0 };
      this.lexicalPOSSeed = this.rsl?.seedLexicalPOS?.(
        this.processor?.entries || new Map(),
        Number.isFinite(options.correlationLexicalPOSSeedGain)
          ? options.correlationLexicalPOSSeedGain
          : 0.74
      ) || { words: 0, posEdges: 0, morphologyEdges: 0 };
    }

    refreshLexicalStudyIndex() {
      this.rootKeys = Array.from(this.processor?.rootMembers?.keys?.() || []);
      if (this.rootCursor >= this.rootKeys.length) this.rootCursor = 0;
    }

    _fingerprint(text) {
      return String(text || '')
        .toLowerCase()
        .replace(/[^a-z0-9'\-]+/g, ' ')
        .trim()
        .replace(/\s+/g, ' ')
        .slice(0, 320);
    }

    _promptKey(text) {
      return this._fingerprint(text).slice(0, 180);
    }

    _tokens(text, limit = 192) {
      const raw = (this.processor?.tokenize?.(String(text || ''), limit) || [])
        .filter(token => token.pos !== 'Punct');
      if (!raw.length) return [];

      const state = { previousWord: '', previousPOS: 'START2', lastWord: '', lastPOS: 'START' };
      const out = [];
      for (const token of raw) {
        const resolution = this.rsl?.resolvePOS?.(token.entry, state, { fallbackPOS: token.pos }) || {
          pos: token.pos,
          certainty: 0.5,
          score: 0.5
        };
        const pos = String(resolution.pos || token.pos || 'Other');
        const entry = this.processor?.materializePOS?.(token.entry, pos) || token.entry;
        out.push({
          ...token,
          word: String(token.lower || token.surface || '').toLowerCase(),
          pos,
          posCertainty: Number(resolution.certainty) || 0.5,
          certainty: Number(resolution.certainty) || 0.5,
          root: String(token.root || entry?.root || '').toLowerCase(),
          contentWord: Boolean(entry?.contentWord ?? token.contentWord),
          entry
        });
        state.previousWord = state.lastWord;
        state.previousPOS = state.lastPOS;
        state.lastWord = String(token.lower || '').toLowerCase();
        state.lastPOS = pos;
      }
      return out.filter(row => row.word || row.lower);
    }

    _sourceGain(source, confidence = null) {
      const s = String(source || 'unknown');
      if (s === 'user') return 1.0;
      if (s === 'model_response') {
        const c = Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence / 100)) : 0.75;
        return 0.40 + c * 0.60;
      }
      if (s === 'self_learning') return 0.48;
      if (s === 'curiosity_probe') return this.options.curiosityStudyGain;
      if (s === 'background_replay') return this.options.replayGain;
      return 0.55;
    }

    _learnWindow(tokens, gain = 1.0, source = 'observed') {
      const rows = Array.isArray(tokens) ? tokens : [];
      if (rows.length < 2) return { updates: 0, tokens: rows.length };
      const baseGain = Math.max(0.03, Math.min(3.0, Number(gain) || 1));
      let updates = 0;

      for (let i = 0; i < rows.length; i++) {
        const from = rows[i];
        if (!from?.word) continue;
        const maxJ = Math.min(rows.length, i + 1 + this.options.window);
        for (let j = i + 1; j < maxJ; j++) {
          const to = rows[j];
          if (!to?.word || to.word === from.word) continue;
          const distance = j - i;
          const distanceGain = baseGain / Math.max(1, distance);
          const certaintyGain = Math.max(0.35, Math.min(1, ((from.certainty || 0.5) + (to.certainty || 0.5)) * 0.5));
          const g = distanceGain * certaintyGain;
          this.rsl?.learnCorrelation?.(from.word, to.word, {
            gain: g,
            source,
            symmetric: distance > 1,
            reverseGain: distance > 1 ? 0.74 : 0.24
          });
          this.rsl?.learnPOSCorrelation?.(from.pos, to.pos, {
            gain: g * 0.70,
            distance,
            symmetric: false
          });
          updates++;
        }
      }

      this.tokensStudied += rows.length;
      this.pairUpdates += updates;
      this.lastPairUpdates = updates;
      return { updates, tokens: rows.length };
    }

    observeText(text, metadata = {}) {
      const tokens = this._tokens(text);
      if (!tokens.length) return { learned: false, tokens: 0, updates: 0 };
      const source = String(metadata.source || 'unknown');
      const confidence = Number.isFinite(metadata.confidence) ? Number(metadata.confidence) : null;
      const gain = Number.isFinite(metadata.gain) ? Number(metadata.gain) : this._sourceGain(source, confidence);

      let episode = null;
      if (metadata.episodic !== false) {
        episode = this.memory?.remember?.(text, {
          source,
          confidence,
          accepted: metadata.accepted !== false,
          episodic: true
        }) || null;
      }

      const structural = this.rsl?.learnPOSSequence?.(tokens, {
        source,
        gain,
        countCycle: metadata.countCycle !== false
      }) || null;
      const correlated = this._learnWindow(tokens, gain, source);

      this.lexical.wordState?.observeText?.(text, this.processor, {
        source, confidence, accepted: metadata.accepted !== false, gain
      });
      this.lexical.phraseState?.observeText?.(text, this.processor, this.rsl, {
        source, confidence, accepted: metadata.accepted !== false, gain
      });
      this.lexical.languageState?.observeText?.(text, {
        source, confidence, accepted: metadata.accepted !== false, gain
      });
      this.lexical.wordMatrix?.observeWords?.(tokens.map(x => x.word), {
        source, confidence, accepted: metadata.accepted !== false, gain
      });

      this.observedTexts++;
      this.observedEpisodes += episode ? 1 : 0;
      if (episode) this.lastEpisodeId = episode.id;
      this.lastConfidence = confidence ?? this.lastConfidence;
      return {
        learned: true,
        tokens: tokens.length,
        episodeId: episode?.id || null,
        gain,
        updates: correlated.updates || 0,
        posUpdates: structural?.posUpdates || 0,
        trigramUpdates: structural?.trigramUpdates || 0,
        contextUpdates: structural?.contextUpdates || 0,
        morphologyUpdates: structural?.morphologyUpdates || 0
      };
    }

    observePair(promptText, responseText, metadata = {}) {
      const prompt = this._tokens(promptText, 96).filter(x => x.contentWord);
      const response = this._tokens(responseText, 128).filter(x => x.contentWord);
      if (!prompt.length || !response.length) return { learned: false, updates: 0 };
      const confidence = Number.isFinite(metadata.confidence) ? Number(metadata.confidence) : 80;
      const confidenceGain = Math.max(0.30, Math.min(1, confidence / 100));
      const gain = Number.isFinite(metadata.gain)
        ? Math.max(0.03, Math.min(1.5, Number(metadata.gain)))
        : Math.max(0.05, Math.min(1.5, this.options.crossTurnGain * confidenceGain));
      let updates = 0;

      for (const p of prompt.slice(0, 16)) {
        for (const r of response.slice(0, 24)) {
          if (!p.word || !r.word || p.word === r.word) continue;
          this.rsl?.learnCorrelation?.(p.word, r.word, {
            gain,
            source: metadata.source || 'prompt-response',
            symmetric: true,
            reverseGain: 0.82
          });
          updates++;
        }
      }
      this.lexical.wordMatrix?.observePair?.(promptText, responseText, this.processor, { confidence, gain });
      this.observedPairs++;
      this.crossTurnCycles++;
      this.pairUpdates += updates;
      this.lastPairUpdates = updates;
      return { learned: true, updates, gain };
    }

    learnCandidate(text, metadata = {}) {
      const tokens = this._tokens(text);
      if (!tokens.length) return { learned: false, tokens: 0 };
      const confidence = Number.isFinite(metadata.confidence) ? Number(metadata.confidence) : 0;
      const structureSupport = Number.isFinite(metadata.structureSupport)
        ? Number(metadata.structureSupport)
        : (this.rsl?.structureSupport?.(text, this.processor) ?? 0);
      const grammar = Number.isFinite(metadata.grammarLegality) ? Number(metadata.grammarLegality) : 0;
      const posCoherence = Number.isFinite(metadata.posCoherence) ? Number(metadata.posCoherence) : structureSupport;
      const scaffoldFit = Number.isFinite(metadata.scaffoldFit) ? Number(metadata.scaffoldFit) : 0;
      const sentenceComplete = Number.isFinite(metadata.sentenceCompleteness) ? Number(metadata.sentenceCompleteness) : 0;
      const quality = clamp01(
        structureSupport * 0.25 +
        grammar * 0.25 +
        posCoherence * 0.20 +
        scaffoldFit * 0.20 +
        sentenceComplete * 0.10
      );
      const gain = Math.max(0.15, Math.min(3.0, Number(metadata.gain) || 1.0));

      if (quality >= 0.68) {
        const learned = this.rsl?.learnPOSSequence?.(tokens, {
          source: 'self_learning',
          gain: gain * Math.max(0.20, Math.min(0.75, quality - 0.25)),
          countCycle: true
        }) || null;
        const tokenLearned = this.rsl?.learnTextTransitions?.(text, {
          source: 'self_learning',
          gain: gain * Math.max(0.18, Math.min(0.65, quality - 0.30)),
          selfLearning: true
        }) || null;
        this.lexical.wordState?.observeText?.(text, this.processor, {
          source: 'self_learning', confidence, accepted: true, quality, gain: gain * 0.55
        });
        this.lexical.phraseState?.observeText?.(text, this.processor, this.rsl, {
          source: 'self_learning', confidence, accepted: true, quality, gain: gain * 0.55
        });
        this.lexical.languageState?.observeText?.(text, {
          source: 'self_learning', confidence, accepted: true, gain: gain * 0.40
        });
        return {
          learned: true,
          mode: 'reinforce-coherent',
          tokens: tokens.length,
          quality,
          confidence,
          gain,
          posUpdates: learned?.posUpdates || 0,
          trigramUpdates: learned?.trigramUpdates || 0,
          contextUpdates: learned?.contextUpdates || 0,
          contextWindowUpdates: learned?.contextWindowUpdates || 0,
          wordPOSUpdates: learned?.wordPOSUpdates || 0,
          learnedTransitions: tokenLearned?.learnedTransitions || 0,
          structuralSupport: structureSupport
        };
      }

      const penaltyStrength = Math.max(0.2, Math.min(2.2, gain * (1.15 - quality)));
      const posPenalty = this.rsl?.penalizePOSSequence?.(tokens, { strength: penaltyStrength }) || { weakened: 0 };
      const tokenPenalty = this.rsl?.penalizeTextTransitions?.(text, this.processor, { strength: penaltyStrength }) || { weakened: 0 };
      this.lexical.wordState?.observeText?.(text, this.processor, {
        source: 'self_learning', confidence, accepted: false, quality, gain: penaltyStrength
      });
      this.lexical.phraseState?.observeText?.(text, this.processor, this.rsl, {
        source: 'self_learning', confidence, accepted: false, quality, gain: penaltyStrength
      });
      this.lexical.languageState?.observeText?.(text, {
        source: 'self_learning_failure', confidence, accepted: false, gain: Math.max(0.15, penaltyStrength * 0.22)
      });
      if (this.rsl) this.rsl.selfLearningCycles = (this.rsl.selfLearningCycles || 0) + 1;
      return {
        learned: true,
        mode: 'weaken-incoherent',
        tokens: tokens.length,
        quality,
        confidence,
        gain,
        weakenedPOS: posPenalty.weakened || 0,
        weakenedTokens: tokenPenalty.weakened || 0,
        structuralSupport: structureSupport
      };
    }

    learnFailure(promptText, candidateText, metadata = {}) {
      const channels = metadata.channels || {};
      const inferredQuality = clamp01(
        (Number(channels.learnedPOSContext) || 0) * 0.24 +
        (Number(channels.grammarLegality) || 0) * 0.24 +
        (Number(channels.dynamicPOSCoherence) || 0) * 0.18 +
        (Number(channels.scaffoldFit) || 0) * 0.16 +
        (Number(channels.sentenceCompleteness) || 0) * 0.12 +
        (Number(channels.correlationCoherence) || 0) * 0.06
      );
      const quality = Number.isFinite(metadata.quality) ? clamp01(metadata.quality) : inferredQuality;
      const confidence = Math.max(0, Math.min(100, Number(metadata.confidence) || 0));
      const deficit = Math.max(0, 80 - confidence) / 80;
      const gain = Number.isFinite(metadata.gain) ? Number(metadata.gain) : Math.min(3.4, 0.95 + Math.max(0, 80 - confidence) / 28);
      const structural = this.learnCandidate(candidateText, {
        confidence,
        gain,
        structureSupport: channels.learnedPOSContext,
        grammarLegality: channels.grammarLegality,
        posCoherence: channels.dynamicPOSCoherence,
        scaffoldFit: channels.scaffoldFit,
        sentenceCompleteness: channels.sentenceCompleteness
      });

      const strength = Math.max(0.06, Math.min(0.55,
        this.options.failurePenalty * (0.65 + deficit) * (1.15 - quality)
      ));
      const weakened = this.rsl?.penalizeCorrelationText?.(candidateText, this.processor, { strength }) || { weakened: 0 };
      this.lexical.wordMatrix?.observeText?.(candidateText, this.processor, {
        source: 'failed_response', confidence, accepted: false, quality, gain: Math.max(0.2, strength * 2.0)
      });

      let promptUpdates = 0;
      if (quality >= 0.64 && candidateText) {
        promptUpdates = this.observePair(promptText, candidateText, {
          confidence: Math.min(79.9, confidence),
          gain: 0.12,
          source: 'failed-but-coherent'
        }).updates || 0;
      }
      this.failureCycles++;
      return {
        weakened: weakened.weakened || 0,
        strength,
        promptUpdates,
        quality,
        structural
      };
    }

    _candidateQuality(candidate) {
      const ch = candidate?.diagnostics?.channels || {};
      return clamp01(
        (Number(ch.grammarLegality) || 0) * 0.25 +
        (Number(ch.dynamicPOSCoherence) || 0) * 0.20 +
        (Number(ch.scaffoldFit) || 0) * 0.18 +
        (Number(ch.sentenceCompleteness) || 0) * 0.15 +
        (Number(ch.languageStateSupport) || 0) * 0.12 +
        (Number(ch.correlationCoherence) || 0) * 0.10
      );
    }

    _tokenSet(text) {
      return new Set(this._fingerprint(text).split(' ').filter(Boolean));
    }

    _jaccardDistance(aText, bText) {
      const a = this._tokenSet(aText);
      const b = this._tokenSet(bText);
      if (!a.size && !b.size) return 0;
      let intersection = 0;
      for (const token of a) if (b.has(token)) intersection++;
      const union = a.size + b.size - intersection;
      return union ? 1 - intersection / union : 0;
    }

    curiosityScore(promptText, candidate, analysis = null) {
      const text = String(candidate?.text || '');
      const fp = this._fingerprint(text);
      if (!fp) return 0;
      const promptKey = this._promptKey(promptText);
      const routeKey = `${promptKey}=>${fp}`;
      const responseVisits = this.responseVisits.get(fp) || 0;
      const routeVisits = this.routeVisits.get(routeKey) || 0;
      const novelty = 1 / (1 + routeVisits * 0.90 + responseVisits * 0.24);
      const last = this.lastAnswerForPrompt.get(promptKey) || '';
      const divergence = last ? this._jaccardDistance(last, fp) : 0.35;
      const words = fp.split(' ').filter(Boolean);
      const lexicalDiversity = words.length ? new Set(words).size / words.length : 0;
      const structure = this._candidateQuality(candidate);
      const requestedFit = analysis?.requestedSlot && analysis?.requestedSlot !== 'general'
        ? clamp01(Number(candidate?.diagnostics?.channels?.scaffoldFit) || structure)
        : structure;
      return clamp01(
        novelty * 0.42 +
        divergence * 0.24 +
        lexicalDiversity * 0.12 +
        structure * 0.14 +
        requestedFit * 0.08
      );
    }

    selectCuriousCandidate(promptText, candidates, minConfidence = 80, analysis = null) {
      const eligible = (Array.isArray(candidates) ? candidates : [])
        .filter(candidate => (Number(candidate?.confidence) || 0) > minConfidence);
      if (!eligible.length) return { selected: null, curiosity: null };
      eligible.sort((a, b) => (Number(b.confidence) || 0) - (Number(a.confidence) || 0));
      const best = eligible[0];
      const bestConfidence = Number(best.confidence) || 0;
      const safe = eligible.filter(candidate =>
        bestConfidence - (Number(candidate.confidence) || 0) <= this.options.curiosityMaxConfidenceDrop
      );

      if (!this.options.curiosityEnabled || safe.length < 2) {
        return {
          selected: best,
          curiosity: {
            enabled: this.options.curiosityEnabled,
            alternativeChosen: false,
            candidateCount: safe.length,
            score: this.curiosityScore(promptText, best, analysis),
            confidenceDrop: 0
          }
        };
      }

      let selected = best;
      let selectedCuriosity = this.curiosityScore(promptText, best, analysis);
      let selectedScore = (bestConfidence / 100) * (1 - this.options.curiosityWeight) + selectedCuriosity * this.options.curiosityWeight;
      for (const candidate of safe.slice(1)) {
        const curiosity = this.curiosityScore(promptText, candidate, analysis);
        const confidence = Number(candidate.confidence) || 0;
        const adjusted = (confidence / 100) * (1 - this.options.curiosityWeight) + curiosity * this.options.curiosityWeight;
        if (adjusted > selectedScore + 1e-9) {
          selected = candidate;
          selectedCuriosity = curiosity;
          selectedScore = adjusted;
        }
      }

      this.curiositySelections++;
      const alternativeChosen = selected !== best;
      if (alternativeChosen) this.curiosityAlternativeSelections++;
      const detail = {
        enabled: true,
        alternativeChosen,
        candidateCount: safe.length,
        score: selectedCuriosity,
        adjustedScore: selectedScore,
        bestConfidence,
        selectedConfidence: Number(selected.confidence) || 0,
        confidenceDrop: Math.max(0, bestConfidence - (Number(selected.confidence) || 0))
      };
      this.lastCuriosity = detail;
      return { selected, curiosity: detail };
    }

    noteSelected(promptText, candidate, metadata = {}) {
      const text = String(candidate?.text || candidate || '');
      const fp = this._fingerprint(text);
      if (!fp) return null;
      const promptKey = this._promptKey(promptText);
      const routeKey = `${promptKey}=>${fp}`;
      this.responseVisits.set(fp, Math.min(65535, (this.responseVisits.get(fp) || 0) + 1));
      this.routeVisits.set(routeKey, Math.min(65535, (this.routeVisits.get(routeKey) || 0) + 1));
      this.lastAnswerForPrompt.set(promptKey, fp);
      this.curiosityCounter++;
      return {
        promptKey,
        fingerprint: fp,
        responseVisits: this.responseVisits.get(fp),
        routeVisits: this.routeVisits.get(routeKey),
        confidence: Number(metadata.confidence ?? candidate?.confidence) || 0
      };
    }

    queueCuriosityCandidates(promptText, candidates, selected, analysis = null) {
      if (!this.options.curiosityEnabled) return 0;
      const selectedFp = this._fingerprint(selected?.text || selected || '');
      const selectedConfidence = Number(selected?.confidence) || 0;
      let queued = 0;
      for (const candidate of Array.isArray(candidates) ? candidates : []) {
        const fp = this._fingerprint(candidate?.text || '');
        if (!fp || fp === selectedFp) continue;
        const confidence = Number(candidate?.confidence) || 0;
        if (confidence < Math.max(68, selectedConfidence - 12)) continue;
        const quality = this._candidateQuality(candidate);
        if (quality < 0.58) continue;
        const key = `${this._promptKey(promptText)}=>${fp}`;
        if (this.curiosityQueuedKeys.has(key)) continue;
        if (this.curiosityQueue.length >= this.options.curiosityQueueMax) {
          const dropped = this.curiosityQueue.shift();
          if (dropped?.key) this.curiosityQueuedKeys.delete(dropped.key);
        }
        this.curiosityQueue.push({
          key,
          promptText: String(promptText || ''),
          text: String(candidate.text || ''),
          confidence,
          quality,
          curiosity: this.curiosityScore(promptText, candidate, analysis),
          channels: candidate?.diagnostics?.channels || null
        });
        this.curiosityQueuedKeys.add(key);
        queued++;
      }
      return queued;
    }

    _studyCuriosityProbe() {
      const probe = this.curiosityQueue.shift();
      if (!probe) return false;
      this.curiosityQueuedKeys.delete(probe.key);
      const tokens = this._tokens(probe.text);
      if (!tokens.length) return false;

      // Curiosity explores language structure without promoting an unshown answer
      // to trusted factual knowledge. Strong alternatives reinforce POS, phrase and
      // morphology behavior only. Weak alternatives are structurally weakened.
      if (probe.quality >= 0.70) {
        const gain = this.options.curiosityStudyGain * (0.75 + probe.curiosity * 0.50);
        this.rsl?.learnPOSSequence?.(tokens, {
          source: 'curiosity_probe',
          gain,
          countCycle: true
        });
        this.lexical.phraseState?.observeText?.(probe.text, this.processor, this.rsl, {
          source: 'curiosity_probe', confidence: probe.confidence, accepted: true, quality: probe.quality, gain
        });
        this.lexical.languageState?.observeText?.(probe.text, {
          source: 'curiosity_probe', confidence: probe.confidence, accepted: true, gain: gain * 0.8
        });
        this.curiosityStudyCycles++;
        this.studyCycles++;
        return true;
      }

      const strength = Math.max(0.05, (0.70 - probe.quality) * 0.45);
      this.rsl?.penalizePOSSequence?.(tokens, { strength });
      this.curiosityRejected++;
      this.curiosityStudyCycles++;
      return true;
    }

    _studyEpisodeStructural(episode) {
      if (!episode?.text) return false;
      const tokens = this._tokens(episode.text);
      if (!tokens.length) return false;
      const sourceGain = this._sourceGain(episode.source, episode.confidence);
      const replayGain = Math.max(0.08, Math.min(0.75, sourceGain * this.options.replayGain));
      this.rsl?.learnPOSSequence?.(tokens, {
        source: 'background_replay',
        gain: replayGain,
        countCycle: true
      });
      this.lexical.wordState?.observeText?.(episode.text, this.processor, {
        source: 'background_replay', confidence: episode.confidence, accepted: episode.accepted !== false, gain: replayGain
      });
      this.lexical.phraseState?.observeText?.(episode.text, this.processor, this.rsl, {
        source: 'background_replay', confidence: episode.confidence, accepted: episode.accepted !== false, gain: replayGain
      });
      this.lexical.languageState?.observeText?.(episode.text, {
        source: 'background_replay', confidence: episode.confidence, accepted: episode.accepted !== false, gain: replayGain * 0.75
      });
      this.replayCycles++;
      this.studyCycles++;
      return true;
    }

    _studyEpisodeCorrelation(episode) {
      if (!episode?.text) return false;
      const tokens = this._tokens(episode.text);
      if (!tokens.length) return false;
      const gain = this._sourceGain('background_replay', episode.confidence) *
        Math.max(0.55, this._sourceGain(episode.source, episode.confidence));
      const result = this._learnWindow(tokens, gain, 'background_replay');
      if (result.updates > 0) {
        this.studyCycles++;
        return true;
      }
      return false;
    }

    _studyMorphology() {
      if (!this.rootKeys.length) this.refreshLexicalStudyIndex();
      if (!this.rootKeys.length) return false;
      const root = this.rootKeys[this.rootCursor % this.rootKeys.length];
      this.rootCursor = (this.rootCursor + 1) % Math.max(1, this.rootKeys.length);
      const members = Array.from(this.processor?.rootMembers?.get?.(root) || []);
      if (!members.length) return false;
      let learned = 0;
      for (const lower of members.slice(0, 24)) {
        const entry = this.processor?.entries?.get?.(lower);
        if (!entry || entry.active === false) continue;
        this.rsl?.learnMorphologyForm?.(root, entry.surface, entry.pos, this.options.morphologyGain);
        learned++;
      }
      if (learned) {
        this.morphologyCycles++;
        this.studyCycles++;
      }
      return learned > 0;
    }

    _studyCrossTurn(recent) {
      if (!Array.isArray(recent) || recent.length < 2) return false;
      for (let offset = 1; offset < Math.min(10, recent.length); offset++) {
        const i = (this.lastReplayIndex - offset + recent.length) % recent.length;
        const a = recent[i];
        const b = recent[(i + 1) % recent.length];
        if (!a || !b) continue;
        if (a.source === 'user' && b.source === 'model_response' && b.accepted !== false) {
          const result = this.observePair(a.text, b.text, { confidence: b.confidence });
          if (result.learned) {
            this.studyCycles++;
            return true;
          }
        }
      }
      return false;
    }

    backgroundTick(phase = 'WARM', epochAtStart = 0, getForegroundEpoch = null, overrideUnits = null) {
      const started = performance.now();
      const units = Number.isFinite(overrideUnits)
        ? Math.max(1, overrideUnits | 0)
        : phase === 'IDLE'
          ? this.options.idleStudyUnits
          : phase === 'WARM'
            ? this.options.warmStudyUnits
            : this.options.activeStudyUnits;
      const recent = this.memory?.recent?.(64) || [];
      let completed = 0;

      for (let i = 0; i < units; i++) {
        if (typeof getForegroundEpoch === 'function' && getForegroundEpoch() !== epochAtStart) break;
        if (performance.now() - started >= (overrideUnits == null ? this.options.maxTickMs : this.options.maxBurstMs)) break;
        let studied = false;

        const structuralEpisode = this.memory?.nextReplay?.();
        if (structuralEpisode) studied = this._studyEpisodeStructural(structuralEpisode) || studied;

        if (recent.length) {
          const index = this.lastReplayIndex % recent.length;
          this.lastReplayIndex = (this.lastReplayIndex + 1) >>> 0;
          studied = this._studyEpisodeCorrelation(recent[index]) || studied;
          if ((this.tickCount + i) % 3 === 0) studied = this._studyCrossTurn(recent) || studied;
        }

        if ((this.tickCount + i) % this.options.morphologyEvery === 0) {
          studied = this._studyMorphology() || studied;
        }
        if ((this.tickCount + i) % this.options.curiosityEvery === 0) {
          studied = this._studyCuriosityProbe() || studied;
        }
        if (studied) completed++;
      }

      this.lexical.wordState?.backgroundTick?.(phase);
      this.lexical.languageState?.backgroundTick?.(phase);
      this.tickCount++;
      this.lastStudyMs = performance.now() - started;
      return {
        phase,
        completed,
        studyMs: this.lastStudyMs,
        pairUpdates: this.lastPairUpdates,
        totalPairUpdates: this.pairUpdates,
        curiosityQueue: this.curiosityQueue.length,
        curiosityStudyCycles: this.curiosityStudyCycles
      };
    }

    studyBurst(units = 12, phase = 'WARM', epochAtStart = 0, getForegroundEpoch = null) {
      return this.backgroundTick(phase, epochAtStart, getForegroundEpoch, Math.max(1, units | 0));
    }

    exportState() {
      const take = (map, limit = 4096) => Array.from(map.entries()).slice(-limit);
      return {
        schemaVersion: 1,
        curiosityCounter: this.curiosityCounter,
        curiositySelections: this.curiositySelections,
        curiosityAlternativeSelections: this.curiosityAlternativeSelections,
        curiosityStudyCycles: this.curiosityStudyCycles,
        curiosityRejected: this.curiosityRejected,
        responseVisits: take(this.responseVisits),
        routeVisits: take(this.routeVisits),
        lastAnswerForPrompt: take(this.lastAnswerForPrompt, 2048)
      };
    }

    importState(state) {
      if (!state || Number(state.schemaVersion) !== 1) return false;
      const restore = rows => new Map(Array.isArray(rows) ? rows : []);
      this.responseVisits = restore(state.responseVisits);
      this.routeVisits = restore(state.routeVisits);
      this.lastAnswerForPrompt = restore(state.lastAnswerForPrompt);
      this.curiosityCounter = Math.max(0, Number(state.curiosityCounter) || 0);
      this.curiositySelections = Math.max(0, Number(state.curiositySelections) || 0);
      this.curiosityAlternativeSelections = Math.max(0, Number(state.curiosityAlternativeSelections) || 0);
      this.curiosityStudyCycles = Math.max(0, Number(state.curiosityStudyCycles) || 0);
      this.curiosityRejected = Math.max(0, Number(state.curiosityRejected) || 0);
      return true;
    }

    reset() {
      this.tickCount = 0;
      this.studyCycles = 0;
      this.replayCycles = 0;
      this.crossTurnCycles = 0;
      this.failureCycles = 0;
      this.morphologyCycles = 0;
      this.observedTexts = 0;
      this.observedPairs = 0;
      this.observedEpisodes = 0;
      this.tokensStudied = 0;
      this.pairUpdates = 0;
      this.lastPairUpdates = 0;
      this.lastStudyMs = 0;
      this.lastReplayIndex = 0;
      this.lastEpisodeId = 0;
      this.lastConfidence = 0;
      this.rootCursor = 0;
      this.curiosityQueue = [];
      this.curiosityQueuedKeys.clear();
      this.curiositySelections = 0;
      this.curiosityAlternativeSelections = 0;
      this.curiosityStudyCycles = 0;
      this.curiosityRejected = 0;
      this.curiosityCounter = 0;
      this.lastCuriosity = null;
      this.responseVisits.clear();
      this.routeVisits.clear();
      this.lastAnswerForPrompt.clear();
      this.refreshLexicalStudyIndex();
    }

    status() {
      return {
        role: 'production-unified-correlation-structural-curiosity-learning',
        tickCount: this.tickCount,
        studyCycles: this.studyCycles,
        replayCycles: this.replayCycles,
        crossTurnCycles: this.crossTurnCycles,
        failureCycles: this.failureCycles,
        morphologyCycles: this.morphologyCycles,
        observedTexts: this.observedTexts,
        observedPairs: this.observedPairs,
        observedEpisodes: this.observedEpisodes,
        tokensStudied: this.tokensStudied,
        pairUpdates: this.pairUpdates,
        lastPairUpdates: this.lastPairUpdates,
        lastStudyMs: this.lastStudyMs,
        lexicalRootFamiliesAvailable: this.rootKeys.length,
        posCurriculumSeed: this.posCurriculumSeed,
        lexicalPOSSeed: this.lexicalPOSSeed,
        curiosity: {
          enabled: this.options.curiosityEnabled,
          weight: this.options.curiosityWeight,
          maxConfidenceDrop: this.options.curiosityMaxConfidenceDrop,
          queued: this.curiosityQueue.length,
          selections: this.curiositySelections,
          alternativeSelections: this.curiosityAlternativeSelections,
          studyCycles: this.curiosityStudyCycles,
          rejected: this.curiosityRejected,
          last: this.lastCuriosity
        },
        keepsLearningWithoutPrompt: true,
        dynamicPOSPerOccurrence: true,
        learnsFromFailedResponsesWithoutBlindReinforcement: true,
        learnedWordState: Boolean(this.lexical.wordState),
        learnedPhraseState: Boolean(this.lexical.phraseState),
        learnedWordMatrix: Boolean(this.lexical.wordMatrix),
        generalLanguageState: Boolean(this.lexical.languageState),
        learnsTenseAspectMoodVoiceAgreement: Boolean(this.lexical.languageState),
        learnsUnknownWordGrammarWithoutInventingFacts: Boolean(this.lexical.languageState),
        storesAnswers: false,
        usesTrainingInfo: false,
        trainingInfoRequired: false
      };
    }
  }

  globalThis.VilotCorrelation = VilotCorrelation;
})();

/**
 * Correlation production research extension.
 *
 * Executable, data-driven numerical/statistical/graph utilities used by
 * diagnostics, background learning, calibration, matrix experiments and
 * future compute paths. No prompt-specific phrases or benchmark answers live
 * here. Methods stay out of the foreground hot path unless explicitly called.
 */
(() => {
  'use strict';

  class VilotCorrelationProductionLab {
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

    status() {
      return {
        module: 'Correlation',
        role: 'production-research-extension',
        calls: this.calls,
        methodCount: Object.getOwnPropertyNames(Object.getPrototypeOf(this)).length - 1,
        ageMs: (performance?.now?.() ?? Date.now()) - this.createdAt
      };
    }
  }

  globalThis.VilotCorrelationProductionLab = VilotCorrelationProductionLab;
})();
