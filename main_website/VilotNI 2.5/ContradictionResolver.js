/**
 * VilotNI 2.5 - ContradictionResolver.js
 *
 * Detects contested learned relations and mutually opposing relation types.
 * It marks knowledge as contested instead of silently letting incompatible
 * beliefs coexist at full confidence.
 */
(() => {
  'use strict';

  class VilotContradictionResolver {
    constructor(graph, hypotheses, options = {}) {
      this.graph = graph;
      this.hypotheses = hypotheses;

      this.options = {
        scanLimit: Math.max(16, options.learningContradictionScanLimit | 0 || 96),
        minConfidence: Math.max(0.25, Math.min(0.95, Number(options.learningContradictionMinConfidence) || 0.55))
      };

      this.opposites = new Map([
        ['supports', 'contradicts'],
        ['contradicts', 'supports'],
        ['capability', 'limitation'],
        ['limitation', 'capability'],
        ['increases', 'decreases'],
        ['decreases', 'increases'],
        ['causes', 'prevents'],
        ['prevents', 'causes'],
        ['before', 'after'],
        ['after', 'before'],
        ['enables', 'blocks'],
        ['blocks', 'enables'],
        ['is', 'is-not'],
        ['is-not', 'is']
      ]);

      this.scans = 0;
      this.contested = 0;
      this.last = null;
    }

    run() {
      const rows =
        this.graph?.topEdges?.(
          this.options.scanLimit
        ) || [];

      let contested = 0;
      const reasons = [];

      for (const edge of rows) {
        if (
          edge.contradictionPressure >=
          this.graph.options.contestedThreshold
        ) {
          if (this.graph.markContested(edge, 'opposing-evidence')) {
            this.hypotheses?.markContested?.(edge.key, 'opposing-evidence');
            contested++;
            reasons.push({
              key: edge.key,
              reason: 'opposing-evidence'
            });
          }
        }

        const opposite =
          this.opposites.get(
            String(edge.relation || '')
          );

        if (!opposite) continue;

        const other =
          this.graph?.getEdge?.(
            edge.from,
            opposite,
            edge.to
          );

        if (
          other &&
          edge.confidence >= this.options.minConfidence &&
          other.confidence >= this.options.minConfidence
        ) {
          this.graph.markContested(edge, `opposite:${opposite}`);
          this.graph.markContested(other, `opposite:${edge.relation}`);
          this.hypotheses?.markContested?.(edge.key, `opposite:${opposite}`);
          this.hypotheses?.markContested?.(other.key, `opposite:${edge.relation}`);

          contested += 2;

          reasons.push({
            key: edge.key,
            other: other.key,
            reason: 'opposite-relations'
          });
        }
      }

      this.scans++;
      this.contested += contested;

      this.last = {
        at: Date.now(),
        scanned: rows.length,
        contested,
        reasons: reasons.slice(0, 24)
      };

      return this.last;
    }

    assessPropositionGraph(graph = {}) {
      const rows = Array.isArray(graph?.propositions) ? graph.propositions : [];
      const conflicts = [];
      const seen = new Set();
      const normalize = value => String(value || '').toLowerCase().replace(/[^a-z0-9'+.\- ]+/g, ' ').replace(/\s+/g, ' ').trim();
      for (let i = 0; i < rows.length; i++) {
        const a = rows[i] || {};
        for (let j = i + 1; j < rows.length; j++) {
          const b = rows[j] || {};
          const sameSubject = normalize(a.subject) && normalize(a.subject) === normalize(b.subject);
          const sameObject = normalize(a.object) && normalize(a.object) === normalize(b.object);
          if (!sameSubject || !sameObject) continue;
          const ar = normalize(a.relation), br = normalize(b.relation);
          const opposite = this.opposites.get(ar);
          const polarityConflict = Number(a.polarity || 1) !== Number(b.polarity || 1) && ar === br;
          const relationConflict = Boolean(opposite && opposite === br);
          if (!polarityConflict && !relationConflict) continue;
          const key = [normalize(a.subject), ar, br, normalize(a.object)].join('|');
          if (seen.has(key)) continue;
          seen.add(key);
          conflicts.push({
            a: a.id || `p-${i}`,
            b: b.id || `p-${j}`,
            subject: normalize(a.subject),
            object: normalize(a.object),
            relationA: ar,
            relationB: br,
            type: polarityConflict ? 'polarity-conflict' : 'opposite-relation',
            severity: Math.min(1, ((Number(a.certainty) || .5) + (Number(b.certainty) || .5)) / 2)
          });
        }
      }
      return { count: conflicts.length, conflicts: conflicts.slice(0, 32), safe: conflicts.length === 0 };
    }

    reset() {
      this.scans = 0;
      this.contested = 0;
      this.last = null;
      return true;
    }

    exportState() {
      return {
        schemaVersion: 1,
        scans: this.scans,
        contested: this.contested,
        last: this.last
      };
    }

    importState(state) {
      if (!state || Number(state.schemaVersion) !== 1) return false;

      this.scans = Math.max(0, Number(state.scans) || 0);
      this.contested = Math.max(0, Number(state.contested) || 0);
      this.last = state.last || null;
      return true;
    }

    status() {
      return {
        ready: true,
        role: 'learned-relation-contradiction-resolver',
        scans: this.scans,
        contestedRelations: this.contested,
        last: this.last
      };
    }
  }

  globalThis.VilotContradictionResolver = VilotContradictionResolver;
})();
