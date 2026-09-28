/**
 * VilotNI 2.5 - LatencyProfiler.js
 *
 * Internal latency telemetry. Keeps bounded rolling stage samples and exposes
 * tail percentiles for scheduling. Nothing here is user-visible inference.
 */
(() => {
  'use strict';

  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));

  class VilotLatencyProfiler {
    constructor(options = {}) {
      this.options = {
        sampleCapacity: Math.max(24, Math.min(256, options.latencyProfilerSampleCapacity | 0 || 96)),
        tailPercentile: clamp(options.latencyTailPercentile ?? 0.95, 0.50, 0.999),
        maxStages: Math.max(16, Math.min(128, options.latencyProfilerMaxStages | 0 || 64))
      };
      this.samples = new Map();
      this.counts = new Map();
      this.last = null;
      this.requests = 0;
    }

    _bucket(stage) {
      const key = String(stage || 'unknown');
      let bucket = this.samples.get(key);
      if (!bucket) {
        if (this.samples.size >= this.options.maxStages) {
          const oldest = this.samples.keys().next();
          if (!oldest.done) {
            this.samples.delete(oldest.value);
            this.counts.delete(oldest.value);
          }
        }
        bucket = [];
        this.samples.set(key, bucket);
      }
      return [key, bucket];
    }

    observe(stage, ms) {
      const value = Math.max(0, Number(ms) || 0);
      const [key, bucket] = this._bucket(stage);
      bucket.push(value);
      if (bucket.length > this.options.sampleCapacity) {
        bucket.splice(0, bucket.length - this.options.sampleCapacity);
      }
      this.counts.set(key, (this.counts.get(key) || 0) + 1);
      this.last = { stage: key, ms: value, at: Date.now() };
      return value;
    }

    percentile(stage, p = this.options.tailPercentile, fallback = 0) {
      const bucket = this.samples.get(String(stage || ''));
      if (!bucket?.length) return Math.max(0, Number(fallback) || 0);
      const sorted = bucket.slice().sort((a, b) => a - b);
      const q = clamp(p, 0, 1);
      const index = Math.min(sorted.length - 1, Math.max(0, Math.ceil(q * sorted.length) - 1));
      return sorted[index] || 0;
    }

    mean(stage, fallback = 0) {
      const bucket = this.samples.get(String(stage || ''));
      if (!bucket?.length) return Math.max(0, Number(fallback) || 0);
      return bucket.reduce((sum, value) => sum + value, 0) / bucket.length;
    }

    stats(stage) {
      const key = String(stage || '');
      const bucket = this.samples.get(key) || [];
      if (!bucket.length) return null;
      return {
        stage: key,
        n: bucket.length,
        totalObservations: this.counts.get(key) || bucket.length,
        meanMs: Number(this.mean(key).toFixed(4)),
        p50Ms: Number(this.percentile(key, 0.50).toFixed(4)),
        p90Ms: Number(this.percentile(key, 0.90).toFixed(4)),
        p95Ms: Number(this.percentile(key, 0.95).toFixed(4)),
        p99Ms: Number(this.percentile(key, 0.99).toFixed(4)),
        maxMs: Number(Math.max(...bucket).toFixed(4))
      };
    }

    recordRequest(totalMs, metadata = {}) {
      this.requests++;
      this.observe('roundTripInternal', totalMs);
      this.last = {
        stage: 'request',
        ms: Math.max(0, Number(totalMs) || 0),
        at: Date.now(),
        ...metadata
      };
      return this.last;
    }

    status() {
      const stages = {};
      for (const key of this.samples.keys()) {
        stages[key] = this.stats(key);
      }
      return {
        ready: true,
        role: 'bounded-tail-latency-profiler',
        requests: this.requests,
        tailPercentile: this.options.tailPercentile,
        sampleCapacity: this.options.sampleCapacity,
        stages,
        last: this.last
      };
    }
  }

  globalThis.VilotLatencyProfiler = VilotLatencyProfiler;
})();
