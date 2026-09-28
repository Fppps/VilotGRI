/**
 * VilotNI 2.5 - LatencyGovernor.js
 *
 * Treats the 40ms response target as a performance mission and scheduling signal.
 *
 * Quality is not terminated just because the mission target is crossed. Hard
 * deadline behavior exists only as an explicit opt-in diagnostic mode.
 */
(() => {
  'use strict';

  class VilotLatencyGovernor {
    constructor(options = {}, profiler = null) {
      this.options = {
        targetMs: Math.max(10, Number(options.latencyTargetMs) || 40),
        safetyReserveMs: Math.max(4, Number(options.latencySafetyReserveMs) || 10),
        decoderFinishReserveMs: Math.max(2, Number(options.latencyDecoderFinishReserveMs) || 5),
        semanticReserveMs: Math.max(0.5, Number(options.latencySemanticReserveMs) || 3),
        coherenceReserveMs: Math.max(0.25, Number(options.latencyCoherenceReserveMs) || 1),
        minimumStageReserveMs: Math.max(0.25, Number(options.latencyMinimumStageReserveMs) || 1),
        ewmaAlpha: Math.max(0.05, Math.min(0.65, Number(options.latencyEWMAAlpha) || 0.22)),
        hardStopWithoutEligible: options.latencyHardStopWithoutEligible === true,
        hardDeadlineMode: options.latencyHardDeadlineMode === true,
        tailPercentile: Math.max(0.50, Math.min(0.999, Number(options.latencyTailPercentile) || 0.95)),
        tailGuardFactor: Math.max(1, Number(options.latencyTailGuardFactor) || 1.05),
        tailMinSamples: Math.max(4, options.latencyTailMinSamples | 0 || 8)
      };

      this.profiler = profiler || null;

      this.estimates = new Map([
        ['rsl', 3.5],
        ['analysis', 2.5],
        ['evidence', 2.0],
        ['candidate', 10.0],
        ['interference', 2.0],
        ['semantic', 4.0],
        ['ideaFusion', 0.45],
        ['coherence', 1.2],
        ['rank', 0.8],
        ['finish', this.options.safetyReserveMs]
      ]);

      this.requests = 0;
      this.deadlineAvoidedStages = 0;
      this.deadlineOverruns = 0;
      this.last = null;
    }

    begin(started = performance.now()) {
      const missionTargetAt = started + this.options.targetMs;
      const responseDeadlineAt = this.options.hardDeadlineMode
        ? missionTargetAt
        : Infinity;
      const workDeadlineAt = this.options.hardDeadlineMode
        ? responseDeadlineAt - this.options.safetyReserveMs
        : Infinity;

      this.requests++;
      this.last = {
        started,
        targetMs: this.options.targetMs,
        workDeadlineAt,
        responseDeadlineAt,
        missionTargetAt,
        missionOnly: !this.options.hardDeadlineMode,
        observations: {}
      };

      return {
        started,
        targetMs: this.options.targetMs,
        safetyReserveMs: this.options.safetyReserveMs,
        workDeadlineAt,
        responseDeadlineAt,
        missionTargetAt,
        missionOnly: !this.options.hardDeadlineMode
      };
    }

    remaining(deadlineAt) {
      return Number.isFinite(deadlineAt)
        ? Math.max(0, deadlineAt - performance.now())
        : Infinity;
    }

    _ewmaEstimate(stage, fallback = 1) {
      const value = this.estimates.get(String(stage || ''));
      return Number.isFinite(value)
        ? value
        : Math.max(0, Number(fallback) || 0);
    }

    estimate(stage, fallback = 1) {
      const key = String(stage || '');
      const ewma = this._ewmaEstimate(key, fallback);
      const stats = this.profiler?.stats?.(key) || null;
      const samples = Number(stats?.n ?? stats?.sampleCount) || 0;

      if (samples < this.options.tailMinSamples) {
        return ewma;
      }

      const tail = Number(
        this.profiler?.percentile?.(
          key,
          this.options.tailPercentile,
          ewma
        )
      ) || ewma;

      return Math.max(
        ewma,
        tail * this.options.tailGuardFactor
      );
    }

    observe(stage, ms) {
      const key = String(stage || '');
      const value = Math.max(0, Number(ms) || 0);
      const old = this._ewmaEstimate(key, value || 1);
      const a = this.options.ewmaAlpha;
      const next = old * (1 - a) + value * a;
      this.estimates.set(key, next);
      this.profiler?.observe?.(key, value);

      if (this.last) {
        this.last.observations[key] = value;
      }

      return next;
    }

    canStart(stage, deadlineAt, extraReserveMs = 0, fallbackEstimate = 1) {
      if (!Number.isFinite(deadlineAt)) return true;

      const remaining = this.remaining(deadlineAt);
      const needed =
        this.estimate(stage, fallbackEstimate) +
        Math.max(this.options.minimumStageReserveMs, Number(extraReserveMs) || 0);

      const allowed = remaining > needed;
      if (!allowed) this.deadlineAvoidedStages++;
      return allowed;
    }

    shouldSkipSemantic(deadlineAt) {
      return !this.canStart('semantic', deadlineAt, this.options.semanticReserveMs, 4);
    }

    shouldSkipCoherence(deadlineAt) {
      return !this.canStart('coherence', deadlineAt, this.options.coherenceReserveMs, 1.2);
    }

    decoderShouldStop(deadlineAt) {
      if (!Number.isFinite(deadlineAt)) return false;
      return this.remaining(deadlineAt) <= this.options.decoderFinishReserveMs;
    }

    finish(started, details = {}) {
      const elapsed = Math.max(0, performance.now() - Number(started || 0));
      const over = elapsed >= this.options.targetMs;
      if (over) this.deadlineOverruns++;

      this.last = {
        ...(this.last || {}),
        elapsed,
        over,
        ...details
      };

      this.profiler?.recordRequest?.(
        elapsed,
        this.last
      );

      return this.last;
    }

    status() {
      return {
        ready: true,
        role: 'quality-preserving-latency-mission-governor',
        targetMs: this.options.targetMs,
        safetyReserveMs: this.options.safetyReserveMs,
        hardStopWithoutEligible: this.options.hardStopWithoutEligible,
        hardDeadlineMode: this.options.hardDeadlineMode,
        tailAwareScheduling: true,
        tailPercentile: this.options.tailPercentile,
        tailGuardFactor: this.options.tailGuardFactor,
        requests: this.requests,
        deadlineAvoidedStages: this.deadlineAvoidedStages,
        deadlineOverruns: this.deadlineOverruns,
        estimates: Object.fromEntries(
          Array.from(this.estimates.entries()).map(([key, value]) => [
            key,
            Number(value.toFixed(4))
          ])
        ),
        profiler: this.profiler?.status?.() || null,
        last: this.last,
        mission:
          'Respond as fast as possible while preserving full reasoning breadth and the confidence gate.'
      };
    }
  }

  globalThis.VilotLatencyGovernor = VilotLatencyGovernor;
})();
