/**
 * VilotNI 2.5 - MixedSlotPlanner.js
 *
 * Global coherence planner for the four projected word paths.
 *
 * The old mixer chose each sentence slot largely independently. That can create
 * one-word branch islands and broken phrase boundaries even when every local
 * choice looked reasonable.
 *
 * This planner performs a tiny Viterbi-style search across the four branches:
 *   - the existing 2 percentage-point confidence window is preserved;
 *   - the StateProcessor's preferred local collapse is retained as a signal;
 *   - staying on a coherent branch span is rewarded;
 *   - branch switches require transition/semantic support;
 *   - function-word attachment is protected;
 *   - isolated one-slot branch islands are repaired after decoding.
 *
 * Complexity is O(slots * 4^2), so it does not add another decoder pass.
 */
(() => {

  'use strict';


  const clamp = (x, lo = 0, hi = 1) =>
    Math.max(lo, Math.min(hi, Number.isFinite(Number(x)) ? Number(x) : 0));


  class VilotMixedSlotPlanner {

    constructor(options = {
}
) {

      this.options = {

        confidenceWindowPct:
          Math.max(
            0.1,
            Number(options.interferenceWordConfidenceWindowPct) || 2
          ),

        branchStayReward:
          Math.max(
            0,
            Number(options.mixedSlotBranchStayReward) || 0.10
          ),

        branchSwitchPenalty:
          Math.max(
            0,
            Number(options.mixedSlotBranchSwitchPenalty) || 0.075
          ),

        functionSwitchPenalty:
          Math.max(
            0,
            Number(options.mixedSlotFunctionSwitchPenalty) || 0.12
          ),

        transitionWeight:
          Math.max(
            0,
            Number(options.mixedSlotTransitionWeight) || 0.16
          ),

        compatibilityWeight:
          Math.max(
            0,
            Number(options.mixedSlotCompatibilityWeight) || 0.14
          ),

        preferredWeight:
          Math.max(
            0,
            Number(options.mixedSlotPreferredWeight) || 0.10
          ),

        knowledgeWeight:
          Math.max(
            0,
            Number(options.mixedSlotKnowledgeWeight) || 0.05
          ),

        focusWeight:
          Math.max(
            0,
            Number(options.mixedSlotFocusWeight) || 0.05
          ),

        maxSwitchDensity:
          clamp(
            options.mixedSlotMaxSwitchDensity ?? 0.34,
            0.05,
            0.8
          ),

        islandRepair:
          options.mixedSlotIslandRepair !== false
      
}
;


      this.plans = 0;

      this.lastDiagnostics = null;

    
}


    _isFunctionPOS(pos) {

      return new Set([
        'Determiner',
        'Prep',
        'Aux',
        'Modal',
        'Conj',
        'Pronoun',
        'Particle'
      ]).has(String(pos || ''));

    
}


    _candidateKey(candidate) {

      return [
        Number(candidate?.branch) || 0,
        Number(candidate?.step) || 0,
        String(candidate?.lower || candidate?.word || '')
      ].join('|');

    
}


    _pruneSlot(candidates, baseCandidate = null) {

      const rows =
        (candidates || [])
          .filter(Boolean);


      if (!rows.length) {

        return [];

      
}


      const maxConfidence =
        Math.max(
          ...rows.map(
            row =>
              Number(row?.confidence) || 0
          )
        );


      const window =
        this.options.confidenceWindowPct;


      const filtered =
        rows.filter(
          row =>
            (
              maxConfidence -
              (Number(row?.confidence) || 0)
            ) <=
              window +
              1e-9 ||
            (
              baseCandidate &&
              this._candidateKey(row) ===
              this._candidateKey(baseCandidate)
            )
        );


      return filtered.length
        ? filtered
        : [rows[0]];

    
}


    _emission(
      candidate,
      preferred,
      knowledgeWords
    ) {

      const confidence =
        clamp(
          (Number(candidate?.confidence) || 0) /
          100
        );


      const candidateConfidence =
        clamp(
          (Number(candidate?.candidateConfidence) || 0) /
          100
        );


      const focus =
        clamp(
          candidate?.row?.focusSupport ??
          0.5
        );


      const knowledge =
        knowledgeWords?.has?.(
          candidate?.lower
        )
          ? 1
          : 0;


      const preferredMatch =
        preferred &&
        this._candidateKey(candidate) ===
          this._candidateKey(preferred)
          ? 1
          : 0;


      return (
        confidence * 0.52 +
        candidateConfidence * 0.18 +
        focus *
          this.options.focusWeight +
        knowledge *
          this.options.knowledgeWeight +
        preferredMatch *
          this.options.preferredWeight
      );

    
}


    _transition(
      previous,
      current,
      callbacks
    ) {

      if (!previous || !current) {

        return 0;

      
}


      const sameBranch =
        Number(previous?.branch) ===
        Number(current?.branch);


      const transition =
        clamp(
          callbacks?.transitionScore?.(
            previous,
            current
          ) ?? 0.5
        );


      const compatibility =
        clamp(
          callbacks?.compatibilityScore?.(
            previous,
            current
          ) ?? 0.5
        );


      let score =
        transition *
          this.options.transitionWeight +
        compatibility *
          this.options.compatibilityWeight;


      if (sameBranch) {

        score +=
          this.options
            .branchStayReward;

      
}
 else {

        score -=
          this.options
            .branchSwitchPenalty;


        if (
          this._isFunctionPOS(
            previous?.pos
          ) ||
          this._isFunctionPOS(
            current?.pos
          )
        ) {

          score -=
            this.options
              .functionSwitchPenalty;

        
}


        // A branch change with weak continuity is usually where word salad
        // starts. Penalize it heavily rather than forbidding all switching.
        if (
          transition < 0.28 &&
          compatibility < 0.28
        ) {

          score -= 0.18;

        
}

      
}


      return score;

    
}


    _repairIslands(path, slots, callbacks) {

      if (
        !this.options.islandRepair ||
        path.length < 3
      ) {

        return {

          path,
          repairs: 0
        
}
;

      
}


      const repaired =
        path.slice();


      let repairs = 0;


      for (
        let i = 1;

        i < repaired.length - 1;

        i++
      ) {

        const left =
          repaired[i - 1];


        const current =
          repaired[i];


        const right =
          repaired[i + 1];


        if (
          !left ||
          !current ||
          !right
        ) {

          continue;

        
}


        const leftBranch =
          Number(left.branch);


        const currentBranch =
          Number(current.branch);


        const rightBranch =
          Number(right.branch);


        if (
          leftBranch !==
            rightBranch ||
          currentBranch ===
            leftBranch
        ) {

          continue;

        
}


        const replacement =
          (slots[i] || [])
            .find(
              candidate =>
                Number(candidate?.branch) ===
                leftBranch
            );


        if (!replacement) {

          continue;

        
}


        const leftTransition =
          clamp(
            callbacks?.transitionScore?.(
              left,
              replacement
            ) ?? 0
          );


        const rightTransition =
          clamp(
            callbacks?.transitionScore?.(
              replacement,
              right
            ) ?? 0
          );


        const leftCompatibility =
          clamp(
            callbacks?.compatibilityScore?.(
              left,
              replacement
            ) ?? 0
          );


        const rightCompatibility =
          clamp(
            callbacks?.compatibilityScore?.(
              replacement,
              right
            ) ?? 0
          );


        const confidenceGap =
          Math.abs(
            (Number(replacement?.confidence) || 0) -
            (Number(current?.confidence) || 0)
          );


        if (
          confidenceGap <=
            this.options
              .confidenceWindowPct &&
          (
            leftTransition +
            rightTransition +
            leftCompatibility +
            rightCompatibility
          ) >=
            0.82
        ) {

          repaired[i] =
            replacement;


          repairs++;

        
}

      
}


      return {

        path:
          repaired,
        repairs
      
}
;

    
}


    _switchStats(path) {

      let switches = 0;


      for (
        let i = 1;

        i < path.length;

        i++
      ) {

        if (
          Number(path[i]?.branch) !==
          Number(path[i - 1]?.branch)
        ) {

          switches++;

        
}

      
}


      const opportunities =
        Math.max(
          1,
          path.length - 1
        );


      return {

        switches,
        switchDensity:
          switches /
          opportunities
      
}
;

    
}


    _limitSwitchDensity(path, slots, callbacks) {

      let result =
        path.slice();


      let stats =
        this._switchStats(
          result
        );


      if (
        stats.switchDensity <=
        this.options
          .maxSwitchDensity
      ) {

        return {

          path: result,
          limited: false,
          replacements: 0,
          ...stats
        
}
;

      
}


      // Prefer the dominant branch if the path became too fragmented. Replace
      // the weakest switches first, but only where that branch has a legal
      // candidate inside the same local confidence window.
      const counts =
        new Map();


      for (
        const candidate of result
      ) {

        const branch =
          Number(
            candidate?.branch
          ) || 0;


        counts.set(
          branch,
          (
            counts.get(branch) ||
            0
          ) + 1
        );

      
}


      const dominant =
        Array.from(
          counts.entries()
        )
          .sort(
            (a, b) =>
              b[1] - a[1]
          )[0]?.[0];


      let replacements = 0;


      if (
        Number.isFinite(
          dominant
        )
      ) {

        for (
          let i = 0;

          i < result.length;

          i++
        ) {

          if (
            Number(
              result[i]?.branch
            ) === dominant
          ) {

            continue;

          
}


          const replacement =
            (slots[i] || [])
              .find(
                candidate =>
                  Number(
                    candidate?.branch
                  ) === dominant
              );


          if (!replacement) {

            continue;

          
}


          const confidenceGap =
            Math.abs(
              (
                Number(
                  replacement
                    ?.confidence
                ) || 0
              ) -
              (
                Number(
                  result[i]
                    ?.confidence
                ) || 0
              )
            );


          if (
            confidenceGap >
            this.options
              .confidenceWindowPct
          ) {

            continue;

          
}


          const previous =
            result[i - 1] ||
            null;


          const next =
            result[i + 1] ||
            null;


          const continuity =
            (
              previous
                ? clamp(
                    callbacks
                      ?.transitionScore?.(
                        previous,
                        replacement
                      ) ?? 0.5
                  )
                : 0.5
            ) +
            (
              next
                ? clamp(
                    callbacks
                      ?.transitionScore?.(
                        replacement,
                        next
                      ) ?? 0.5
                  )
                : 0.5
            );


          if (
            continuity >=
            0.72
          ) {

            result[i] =
              replacement;


            replacements++;


            stats =
              this._switchStats(
                result
              );


            if (
              stats.switchDensity <=
              this.options
                .maxSwitchDensity
            ) {

              break;

            
}

          
}

        
}

      
}


      stats =
        this._switchStats(
          result
        );


      return {

        path: result,
        limited: true,
        replacements,
        ...stats
      
}
;

    
}


    plan(
      slots,
      baseRows,
      preferredRows,
      knowledgeWords,
      callbacks = {
}

    ) {

      const started =
        performance.now();


      const pruned =
        (slots || [])
          .map(
            (rows, index) =>
              this._pruneSlot(
                rows,
                baseRows?.[index] ||
                null
              )
          );


      if (
        !pruned.length ||
        pruned.some(
          rows =>
            !rows.length
        )
      ) {

        return {

          path:
            baseRows ||
            [],

          diagnostics: {

            plannerMs:
              performance.now() -
              started,

            fallback:
              'empty-slot'
          
}

        
}
;

      
}


      const scores = [];

      const parents = [];


      for (
        let slot = 0;

        slot < pruned.length;

        slot++
      ) {

        scores[slot] =
          new Array(
            pruned[slot].length
          ).fill(
            -Infinity
          );


        parents[slot] =
          new Array(
            pruned[slot].length
          ).fill(
            -1
          );


        for (
          let j = 0;

          j < pruned[slot].length;

          j++
        ) {

          const current =
            pruned[slot][j];


          const emission =
            this._emission(
              current,
              preferredRows?.[slot] ||
              null,
              knowledgeWords
            );


          if (slot === 0) {

            scores[slot][j] =
              emission;

            continue;

          
}


          let bestScore =
            -Infinity;


          let bestParent =
            -1;


          for (
            let k = 0;

            k <
              pruned[slot - 1]
                .length;

            k++
          ) {

            const previous =
              pruned[slot - 1][k];


            const total =
              scores[slot - 1][k] +
              emission +
              this._transition(
                previous,
                current,
                callbacks
              );


            if (
              total >
              bestScore
            ) {

              bestScore =
                total;

              bestParent =
                k;

            
}

          
}


          scores[slot][j] =
            bestScore;


          parents[slot][j] =
            bestParent;

        
}

      
}


      let finalIndex = 0;


      const lastScores =
        scores[
          scores.length - 1
        ];


      for (
        let i = 1;

        i < lastScores.length;

        i++
      ) {

        if (
          lastScores[i] >
          lastScores[finalIndex]
        ) {

          finalIndex = i;

        
}

      
}


      const path =
        new Array(
          pruned.length
        );


      let cursor =
        finalIndex;


      for (
        let slot =
          pruned.length - 1;

        slot >= 0;

        slot--
      ) {

        path[slot] =
          pruned[slot][cursor];


        cursor =
          parents[slot][cursor];


        if (
          slot > 0 &&
          cursor < 0
        ) {

          cursor = 0;

        
}

      
}


      const island =
        this._repairIslands(
          path,
          pruned,
          callbacks
        );


      const limited =
        this._limitSwitchDensity(
          island.path,
          pruned,
          callbacks
        );


      const branchContributions =
        Array.from(
          new Set(
            limited.path
              .map(
                row =>
                  Number(
                    row?.branch
                  ) || 0
              )
          )
        ).sort();


      const diagnostics = {

        plannerMs:
          performance.now() -
          started,

        slotCount:
          pruned.length,

        branchContributions,

        switchCount:
          limited.switches,

        switchDensity:
          Number(
            limited
              .switchDensity
              .toFixed(4)
          ),

        islandRepairs:
          island.repairs,

        switchDensityLimited:
          limited.limited,

        densityReplacements:
          limited.replacements,

        maxSwitchDensity:
          this.options
            .maxSwitchDensity,

        confidenceWindowPct:
          this.options
            .confidenceWindowPct,

        dynamicProgramming:
          true
      
}
;


      this.plans++;

      this.lastDiagnostics =
        diagnostics;


      return {

        path:
          limited.path,

        diagnostics
      
}
;

    
}


    status() {

      return {

        ready: true,
        role:
          'four-path-global-mixed-slot-coherence-planner',
        plans:
          this.plans,
        lastDiagnostics:
          this.lastDiagnostics,
        architecture: {

          globalSlotPath:
            true,
          viterbiStyle:
            true,
          maxBranches:
            4,
          confidenceWindowPreserved:
            true,
          extraDecoderPasses:
            0
        
}

      
}
;

    
}

  
}


  globalThis.VilotMixedSlotPlanner =
    VilotMixedSlotPlanner;

}
)();

