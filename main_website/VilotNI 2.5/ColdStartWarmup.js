/**
 * VilotNI 2.5 - ColdStartWarmup.js
 *
 * Moves first-use costs out of the first real user request and into BOOTING.
 *
 * The first real request was measurably much slower than the immediately
 * following request. The runtime has several large functions and a WebGPU
 * foreground path that browsers may optimize/compile lazily on first execution.
 *
 * Warmup does NOT observe a fake user turn, does NOT learn transitions, does
 * NOT write conversation memory, and does NOT create a user-visible response.
 * It only exercises read-only analysis/retrieval/generation paths and a GPU
 * no-op dispatch.
 */
(() => {
  'use strict';

  class VilotColdStartWarmup {
    constructor(deps = {}, options = {}) {
      this.compute =
        deps.compute ||
        null;

      this.processor =
        deps.processor ||
        null;

      this.languageState =
        deps.languageState ||
        null;

      this.queryPlanner =
        deps.queryPlanner ||
        null;

      this.evidenceSearch =
        deps.evidenceSearch ||
        null;

      this.candidateRanker =
        deps.candidateRanker ||
        null;

      this.decoder =
        deps.decoder ||
        null;

      this.trainingKnowledge =
        deps.trainingKnowledge ||
        null;

      this.options = {
        enabled:
          options.coldStartWarmupEnabled !== false,

        gpu:
          options.coldStartWarmupGPU !== false,

        model:
          options.coldStartWarmupModel !== false,

        evidenceDepth:
          Math.max(
            1,
            Math.min(
              2,
              options.coldStartWarmupEvidenceDepth | 0 ||
              1
            )
          ),

        trainingLimit:
          Math.max(
            4,
            Math.min(
              8,
              options.coldStartWarmupTrainingLimit | 0 ||
              6
            )
          )
      };

      this.done = false;
      this.last = null;
    }

    _genericPrompt() {
      // Generic system/process language intentionally exercises definition,
      // mechanism, multi-idea, punctuation, semantic collapse, and lexical
      // realization without specializing warmup around a user topic.
      return (
        'How does a system process information and produce an output?'
      );
    }

    async run() {
      if (
        this.done ||
        !this.options.enabled
      ) {
        return (
          this.last || {
            ready: true,
            enabled:
              this.options.enabled,
            skipped: true,
            reason:
              this.done
                ? 'already-warmed'
                : 'disabled'
          }
        );
      }

      const started =
        performance.now();

      const diagnostics = {
        ready: false,
        enabled: true,
        gpu: null,
        analysisMs: 0,
        planningMs: 0,
        evidenceMs: 0,
        trainingSearchMs: 0,
        decodeMs: 0,
        rankMs: 0,
        totalMs: 0,
        candidateCount: 0,
        errors: []
      };

      if (
        this.options.gpu &&
        this.compute
          ?.warmupForegroundPipeline
      ) {
        try {
          diagnostics.gpu =
            await this.compute
              .warmupForegroundPipeline();
        } catch (error) {
          diagnostics.errors.push({
            stage: 'gpu',
            message:
              String(
                error?.message ||
                error
              )
          });
        }
      }

      if (
        this.options.model &&
        this.processor &&
        this.decoder
      ) {
        try {
          const prompt =
            this._genericPrompt();

          let stage =
            performance.now();

          const baseAnalysis =
            this.processor
              .analyzePrompt(
                prompt
              );

          const analysis =
            this.languageState
              ?.enrichAnalysis?.(
                baseAnalysis
              ) ||
            baseAnalysis;

          diagnostics.analysisMs =
            performance.now() -
            stage;

          stage =
            performance.now();

          const plan =
            this.queryPlanner
              ?.plan?.(
                prompt,
                analysis
              ) || {
                intent:
                  analysis?.intent ||
                  'content',
                requestedRole:
                  analysis
                    ?.requestedSlot ||
                  'content',
                focusSeeds:
                  analysis
                    ?.contentWords ||
                  [],
                search: {
                  initialHops: 1
                }
              };

          diagnostics.planningMs =
            performance.now() -
            stage;

          stage =
            performance.now();

          const evidence =
            this.evidenceSearch
              ?.search?.(
                plan,
                analysis,
                {
                  depth:
                    this.options
                      .evidenceDepth
                }
              ) || {
                staticPool: [],
                diagnostics: null
              };

          diagnostics.evidenceMs =
            performance.now() -
            stage;

          // Exercise the TrainingInfo index separately once so its search path
          // is JIT-hot before the decoder requests it.
          if (
            this.trainingKnowledge
              ?.search
          ) {
            stage =
              performance.now();

            this.trainingKnowledge
              .search(
                prompt,
                analysis,
                plan,
                {
                  limit:
                    this.options
                      .trainingLimit
                }
              );

            diagnostics.trainingSearchMs =
              performance.now() -
              stage;
          }

          stage =
            performance.now();

          const batch =
            await this.decoder
              .generateBatch(
                prompt,
                analysis,
                {
                  batchSize: 1,
                  variantBase: 0,
                  batchIndex: 0,
                  effort: 'normal',
                  explorationLevel: 0,
                  recoveryMode: false,
                  focusedFastPath: true,
                  queryPlan: plan,
                  staticPool:
                    evidence
                      ?.staticPool ||
                    undefined,

                  // Boot warmup itself is outside the user's latency budget.
                  deadlineAt: Infinity,
                  responseDeadlineAt:
                    Infinity,

                  internalWarmup: true
                }
              );

          diagnostics.decodeMs =
            performance.now() -
            stage;

          const candidates =
            batch?.candidates ||
            [];

          diagnostics.candidateCount =
            candidates.length;

          if (
            this.candidateRanker
              ?.rank &&
            candidates.length
          ) {
            stage =
              performance.now();

            this.candidateRanker
              .rank(
                candidates,
                plan,
                evidence,
                80
              );

            diagnostics.rankMs =
              performance.now() -
              stage;
          }
        } catch (error) {
          diagnostics.errors.push({
            stage: 'model',
            message:
              String(
                error?.message ||
                error
              ),
            stack:
              String(
                error?.stack ||
                ''
              ).slice(
                0,
                1600
              )
          });
        }
      }

      diagnostics.totalMs =
        performance.now() -
        started;

      diagnostics.ready =
        diagnostics.errors.length === 0;

      this.done = true;
      this.last =
        diagnostics;

      return diagnostics;
    }

    status() {
      return {
        ready:
          this.done,
        role:
          'boot-time-cold-start-prewarm',
        enabled:
          this.options.enabled,
        gpuWarmup:
          this.options.gpu,
        modelWarmup:
          this.options.model,
        learnsFromWarmup:
          false,
        createsDialogueTurn:
          false,
        emitsUserResponse:
          false,
        last:
          this.last
      };
    }
  }

  globalThis.VilotColdStartWarmup =
    VilotColdStartWarmup;
})();
