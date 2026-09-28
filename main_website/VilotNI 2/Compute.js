/**
 * VilotNI 2 - Production WebGPU compute engine.
 *
 * The six-million-neuron state stays fully addressable and persistent. Normal
 * language recurrence is sparse-active: only neuron slots that have ever received
 * drive are dispatched. This is mathematically exact for untouched slots because
 * zero with zero drive remains zero under the recurrent/homeostasis equations.
 */
(() => {
  'use strict';

  class VilotCompute {
    constructor(arena, trace, options = {}) {
      if (!arena) throw new Error('VilotCompute requires a VilotArena.');
      this.arena = arena;
      this.trace = trace || null;
      this.options = options;

      this.mode = 'uninitialized';
      this.adapter = null;
      this.device = null;
      this.deviceLost = false;
      this.deviceInfo = null;

      this.shaderModule = null;
      this.recurrentPipeline = null;
      this.recurrentBatchPipeline = null;
      this.recurrentDensePipeline = null;
      this.homeostasisActivePipeline = null;

      this.gpuStateA = null;
      this.gpuStateB = null;
      this.gpuParams = null;
      this.gpuBatchIds = null;
      this.gpuBatchWeights = null;
      this.gpuDenseInput = null;
      this.gpuActiveSlots = null;
      this.gpuActiveCapacityBytes = 0;
      this.readback = null;

      this.bindRecurrentAB = null;
      this.bindRecurrentBA = null;
      this.bindBatchStateAB = null;
      this.bindBatchStateBA = null;
      this.bindBatchData = null;
      this.bindDenseStateAB = null;
      this.bindDenseStateBA = null;
      this.bindDenseData = null;
      this.bindHomeostasisStateAB = null;
      this.bindHomeostasisStateBA = null;
      this.bindHomeostasisData = null;
      this.gpuReadIsA = true;

      this.workgroupSize = 256;
      this.maxBatchSteps = Math.max(8, Math.min(256, options.rslMaxLexicalStepsPerTurn | 0 || 96));
      this.batchSlotsPerStep = 6;
      this.batchIdsScratch = new Uint32Array(this.maxBatchSteps * this.batchSlotsPerStep);
      this.batchWeightsScratch = new Float32Array(this.maxBatchSteps * this.batchSlotsPerStep);

      // 96-byte uniform shared by all kernels.
      this.paramBuffer = new ArrayBuffer(96);
      this.paramU32 = new Uint32Array(this.paramBuffer);
      this.paramF32 = new Float32Array(this.paramBuffer);

      this.submissions = 0;
      this.foregroundPasses = 0;
      this.backgroundPasses = 0;
      this.batchPasses = 0;
      this.batchedLexicalSteps = 0;
      this.lastBatchSize = 0;
      this.lastActiveDispatch = 0;
      this.activeBufferGrowths = 0;
    }

    async init(shaderURL = './Shaders.wgsl') {
      const start = performance.now();
      if (!globalThis.navigator?.gpu) {
        this.mode = 'cpu-fallback';
        return this.info();
      }

      try {
        this.adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!this.adapter) {
          this.mode = 'cpu-fallback';
          return this.info();
        }

        this.device = await this.adapter.requestDevice();
        this.deviceLost = false;
        this.device.lost.then(info => {
          this.deviceLost = true;
          this.mode = 'cpu-fallback';
          this.trace?.push?.(globalThis.VilotTraceEvent?.GPU_LOST || 4, 0);
          console.warn('[VilotNI 2] WebGPU device lost:', info);
        }).catch(() => {});

        try { this.deviceInfo = this.adapter.info || null; } catch (_) { this.deviceInfo = null; }

        const response = await fetch(shaderURL, { cache: 'no-store' });
        if (!response.ok) throw new Error(`Shader load failed (${response.status}).`);
        this.shaderModule = this.device.createShaderModule({ code: await response.text() });

        const descriptors = [
          { key: 'recurrentPipeline', entryPoint: 'recurrent_update' },
          { key: 'recurrentBatchPipeline', entryPoint: 'recurrent_batch_active' },
          { key: 'recurrentDensePipeline', entryPoint: 'recurrent_dense' },
          { key: 'homeostasisActivePipeline', entryPoint: 'homeostasis_active' }
        ];
        if (typeof this.device.createComputePipelineAsync === 'function') {
          const pipelines = await Promise.all(descriptors.map(d => this.device.createComputePipelineAsync({
            layout: 'auto', compute: { module: this.shaderModule, entryPoint: d.entryPoint }
          })));
          descriptors.forEach((d, i) => { this[d.key] = pipelines[i]; });
        } else {
          for (const d of descriptors) {
            this[d.key] = this.device.createComputePipeline({
              layout: 'auto', compute: { module: this.shaderModule, entryPoint: d.entryPoint }
            });
          }
        }

        this.allocatePersistentBuffers();
        this.mode = 'webgpu';
        this.trace?.push?.(globalThis.VilotTraceEvent?.GPU_INIT || 3, performance.now() - start, this.arena.stateDim);
      } catch (error) {
        console.warn('[VilotNI 2] WebGPU initialization failed, using CPU fallback.', error);
        this.mode = 'cpu-fallback';
      }
      return this.info();
    }

    allocatePersistentBuffers() {
      const device = this.device;
      const bytes = this.arena.stateDim * 4;
      const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      this.gpuStateA = device.createBuffer({ size: bytes, usage: storageUsage });
      this.gpuStateB = device.createBuffer({ size: bytes, usage: storageUsage });
      this.gpuParams = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      this.gpuBatchIds = device.createBuffer({
        size: this.batchIdsScratch.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this.gpuBatchWeights = device.createBuffer({
        size: this.batchWeightsScratch.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this._createActiveGPUBuffer(Math.max(4, this.arena.activeSlots.byteLength));

      device.queue.writeBuffer(this.gpuStateA, 0, this.arena.readState);
      device.queue.writeBuffer(this.gpuStateB, 0, this.arena.readState);

      const makeStateBind = (pipeline, read, write) => device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.gpuParams } },
          { binding: 1, resource: { buffer: read } },
          { binding: 2, resource: { buffer: write } }
        ]
      });

      this.bindRecurrentAB = makeStateBind(this.recurrentPipeline, this.gpuStateA, this.gpuStateB);
      this.bindRecurrentBA = makeStateBind(this.recurrentPipeline, this.gpuStateB, this.gpuStateA);
      this.bindBatchStateAB = makeStateBind(this.recurrentBatchPipeline, this.gpuStateA, this.gpuStateB);
      this.bindBatchStateBA = makeStateBind(this.recurrentBatchPipeline, this.gpuStateB, this.gpuStateA);
      this.bindDenseStateAB = makeStateBind(this.recurrentDensePipeline, this.gpuStateA, this.gpuStateB);
      this.bindDenseStateBA = makeStateBind(this.recurrentDensePipeline, this.gpuStateB, this.gpuStateA);
      this.bindHomeostasisStateAB = makeStateBind(this.homeostasisActivePipeline, this.gpuStateA, this.gpuStateB);
      this.bindHomeostasisStateBA = makeStateBind(this.homeostasisActivePipeline, this.gpuStateB, this.gpuStateA);
      this._rebuildActiveBindings();
      this.gpuReadIsA = true;
    }

    _createActiveGPUBuffer(requiredBytes) {
      if (!this.device) return null;
      const target = Math.max(4, requiredBytes | 0);
      try { this.gpuActiveSlots?.destroy?.(); } catch (_) {}
      this.gpuActiveSlots = this.device.createBuffer({
        size: target,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this.gpuActiveCapacityBytes = target;
      this.activeBufferGrowths++;
      return this.gpuActiveSlots;
    }

    _rebuildActiveBindings() {
      if (!this.device || !this.gpuActiveSlots || !this.recurrentBatchPipeline || !this.homeostasisActivePipeline) return;
      this.bindBatchData = this.device.createBindGroup({
        layout: this.recurrentBatchPipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: this.gpuBatchIds } },
          { binding: 1, resource: { buffer: this.gpuBatchWeights } },
          { binding: 3, resource: { buffer: this.gpuActiveSlots } }
        ]
      });
      this.bindHomeostasisData = this.device.createBindGroup({
        layout: this.homeostasisActivePipeline.getBindGroupLayout(1),
        entries: [{ binding: 3, resource: { buffer: this.gpuActiveSlots } }]
      });
    }

    _ensureGPUActiveCapacity() {
      if (!this.device) return null;
      const required = Math.max(4, this.arena.activeCount * 4);
      if (required <= this.gpuActiveCapacityBytes && this.gpuActiveSlots) return this.gpuActiveSlots;
      let next = Math.max(4096, this.gpuActiveCapacityBytes || 4096);
      while (next < required) next *= 2;
      next = Math.min(this.arena.stateDim * 4, Math.max(required, next));
      this._createActiveGPUBuffer(next);
      this._rebuildActiveBindings();
      return this.gpuActiveSlots;
    }

    _uploadActiveSlots() {
      if (!this.device || !this.arena.activeCount) return 0;
      const buffer = this._ensureGPUActiveCapacity();
      const view = this.arena.activeView();
      this.device.queue.writeBuffer(buffer, 0, view.buffer, view.byteOffset, view.byteLength);
      this.lastActiveDispatch = this.arena.activeCount;
      return this.arena.activeCount;
    }

    _dispatchShape(totalItems = this.arena.stateDim) {
      const totalGroups = Math.ceil(Math.max(1, totalItems) / this.workgroupSize);
      const groupsX = Math.min(65535, Math.max(1, totalGroups));
      return { groupsX, groupsY: Math.ceil(totalGroups / groupsX), invocationStride: groupsX * this.workgroupSize };
    }

    writeParams(decay, inputGain, inhibition, baselinePull, tick = 0, count = 0, sparseIds = null, sparseWeights = null) {
      const shape = this._dispatchShape(this.arena.stateDim);
      this.paramU32.fill(0);
      this.paramU32[0] = this.arena.stateDim >>> 0;
      this.paramU32[1] = tick >>> 0;
      this.paramU32[2] = shape.invocationStride >>> 0;
      this.paramU32[3] = Math.max(0, count | 0) >>> 0;
      this.paramF32[4] = decay;
      this.paramF32[5] = inputGain;
      this.paramF32[6] = inhibition;
      this.paramF32[7] = baselinePull;
      const sparseCount = Math.min(8, sparseIds?.length || 0, sparseWeights?.length || 0, count | 0);
      for (let j = 0; j < sparseCount; j++) {
        this.paramU32[8 + j] = sparseIds[j] >>> 0;
        this.paramF32[16 + j] = Number.isFinite(sparseWeights[j]) ? sparseWeights[j] : 0;
      }
      return shape;
    }

    // Full scan is retained for the explicit dense-input API and diagnostics.
    runCPU(inputEnabled, decay, inputGain, inhibition, baselinePull, sparseIds = null, sparseWeights = null) {
      const read = this.arena.readState;
      const write = this.arena.writeState;
      const denseInput = this.arena.input;
      const sparseCount = Math.min(8, sparseIds?.length || 0, sparseWeights?.length || 0);
      if (inputEnabled && sparseCount) this.arena.markActiveMany(sparseIds, sparseCount);
      for (let i = 0; i < read.length; i++) {
        const x = read[i];
        let drive = inputEnabled && denseInput ? (denseInput[i] || 0) * inputGain : 0;
        if (inputEnabled && sparseCount) {
          for (let j = 0; j < sparseCount; j++) {
            if ((sparseIds[j] >>> 0) === i) drive += (Number(sparseWeights[j]) || 0) * inputGain;
          }
        }
        const inhibitory = inhibition * x * Math.abs(x);
        const baseline = -baselinePull * x;
        let y = x * decay + drive - inhibitory + baseline;
        if (y > 4) y = 4; else if (y < -4) y = -4;
        write[i] = y;
      }
      this.arena.swapState();
    }

    runCPUBatch(ids, weights, count, decay, inputGain, inhibition, baselinePull) {
      const read = this.arena.readState;
      const write = this.arena.writeState;
      const steps = Math.max(0, Math.min(this.maxBatchSteps, count | 0));
      const elements = steps * this.batchSlotsPerStep;
      this.arena.markActiveMany(ids, elements);
      const active = this.arena.activeView();

      for (let k = 0; k < active.length; k++) {
        const i = active[k];
        let x = read[i];
        for (let step = 0; step < steps; step++) {
          const base = step * this.batchSlotsPerStep;
          let drive = 0;
          for (let j = 0; j < this.batchSlotsPerStep; j++) {
            if ((ids[base + j] >>> 0) === i) drive += Number(weights[base + j]) || 0;
          }
          const inhibitory = inhibition * x * Math.abs(x);
          const baseline = -baselinePull * x;
          x = x * decay + drive * inputGain - inhibitory + baseline;
          if (x > 4) x = 4; else if (x < -4) x = -4;
        }
        write[i] = x;
      }
      this.arena.swapState();
      if (steps > 1) {
        this.arena.revision += steps - 1;
        this.arena.swapCount += steps - 1;
      }
      this.lastActiveDispatch = active.length;
    }

    runCPUHomeostasis(decay, inhibition, baselinePull) {
      const read = this.arena.readState;
      const write = this.arena.writeState;
      const active = this.arena.activeView();
      if (!active.length) return false;
      for (let k = 0; k < active.length; k++) {
        const i = active[k];
        const x = read[i];
        const inhibitory = inhibition * x * Math.abs(x);
        const baseline = -baselinePull * x;
        let y = x * decay - inhibitory + baseline;
        if (y > 4) y = 4; else if (y < -4) y = -4;
        write[i] = y;
      }
      this.arena.swapState();
      this.lastActiveDispatch = active.length;
      return true;
    }

    dispatchGPU(pipeline, bindAB, bindBA, inputEnabled, decay, inputGain, inhibition, baselinePull, tick, sparseIds = null, sparseWeights = null) {
      if (!this.device || this.deviceLost) return false;
      const sparseCount = inputEnabled ? Math.min(8, sparseIds?.length || 0, sparseWeights?.length || 0) : 0;
      if (sparseCount) this.arena.markActiveMany(sparseIds, sparseCount);
      const shape = this.writeParams(decay, inputGain, inhibition, baselinePull, tick, sparseCount, sparseIds, sparseWeights);
      this.device.queue.writeBuffer(this.gpuParams, 0, this.paramBuffer);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.gpuReadIsA ? bindAB : bindBA);
      pass.dispatchWorkgroups(shape.groupsX, shape.groupsY);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      this.gpuReadIsA = !this.gpuReadIsA;
      this.arena.revision++;
      this.arena.swapCount++;
      this.submissions++;
      return true;
    }

    _ensureDenseGPUInput() {
      if (this.gpuDenseInput || !this.device) return this.gpuDenseInput;
      this.gpuDenseInput = this.device.createBuffer({
        size: this.arena.stateDim * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });
      this.bindDenseData = this.device.createBindGroup({
        layout: this.recurrentDensePipeline.getBindGroupLayout(1),
        entries: [{ binding: 2, resource: { buffer: this.gpuDenseInput } }]
      });
      return this.gpuDenseInput;
    }

    _dispatchDenseGPU(input, decay, inputGain, inhibition, baselinePull, tick) {
      if (!this.device || this.deviceLost || !input) return false;
      const buffer = this._ensureDenseGPUInput();
      this.device.queue.writeBuffer(buffer, 0, input.buffer, input.byteOffset || 0, Math.min(input.byteLength, this.arena.stateDim * 4));
      const shape = this.writeParams(decay, inputGain, inhibition, baselinePull, tick, 0);
      this.device.queue.writeBuffer(this.gpuParams, 0, this.paramBuffer);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.recurrentDensePipeline);
      pass.setBindGroup(0, this.gpuReadIsA ? this.bindDenseStateAB : this.bindDenseStateBA);
      pass.setBindGroup(1, this.bindDenseData);
      pass.dispatchWorkgroups(shape.groupsX, shape.groupsY);
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      this.gpuReadIsA = !this.gpuReadIsA;
      this.arena.revision++;
      this.arena.swapCount++;
      this.submissions++;
      return true;
    }

    recurrent(options = {}) {
      const decay = Number.isFinite(options.decay) ? options.decay : 0.985;
      const inputGain = Number.isFinite(options.inputGain) ? options.inputGain : 0.36;
      const inhibition = Number.isFinite(options.inhibition) ? options.inhibition : 0.018;
      const baselinePull = Number.isFinite(options.baselinePull) ? options.baselinePull : 0.002;
      const tick = options.tick | 0;
      const sparseIds = options.sparseIds || null;
      const sparseWeights = options.sparseWeights || null;

      if (this.mode === 'webgpu') {
        const ok = options.denseInput
          ? this._dispatchDenseGPU(options.denseInput, decay, inputGain, inhibition, baselinePull, tick)
          : this.dispatchGPU(this.recurrentPipeline, this.bindRecurrentAB, this.bindRecurrentBA, true,
              decay, inputGain, inhibition, baselinePull, tick, sparseIds, sparseWeights);
        if (!ok) this.runCPU(true, decay, inputGain, inhibition, baselinePull, sparseIds, sparseWeights);
      } else {
        this.runCPU(true, decay, inputGain, inhibition, baselinePull, sparseIds, sparseWeights);
      }
      this.foregroundPasses++;
    }

    recurrentBatch(options = {}) {
      const count = Math.max(0, Math.min(this.maxBatchSteps, options.count | 0));
      if (!count) return false;
      const ids = options.ids;
      const weights = options.weights;
      if (!ids || !weights) return false;
      const elements = count * this.batchSlotsPerStep;
      const decay = Number.isFinite(options.decay) ? options.decay : 0.985;
      const inputGain = Number.isFinite(options.inputGain) ? options.inputGain : 0.36;
      const inhibition = Number.isFinite(options.inhibition) ? options.inhibition : 0.018;
      const baselinePull = Number.isFinite(options.baselinePull) ? options.baselinePull : 0.002;

      this.arena.markActiveMany(ids, elements);

      if (this.mode === 'webgpu' && this.device && !this.deviceLost) {
        this.batchIdsScratch.set(ids.subarray ? ids.subarray(0, elements) : Array.from(ids).slice(0, elements), 0);
        this.batchWeightsScratch.set(weights.subarray ? weights.subarray(0, elements) : Array.from(weights).slice(0, elements), 0);
        this.device.queue.writeBuffer(this.gpuBatchIds, 0, this.batchIdsScratch.buffer, 0, elements * 4);
        this.device.queue.writeBuffer(this.gpuBatchWeights, 0, this.batchWeightsScratch.buffer, 0, elements * 4);
        const activeCount = this._uploadActiveSlots();
        if (!activeCount) return false;

        this.writeParams(decay, inputGain, inhibition, baselinePull, activeCount, count);
        this.device.queue.writeBuffer(this.gpuParams, 0, this.paramBuffer);
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.recurrentBatchPipeline);
        pass.setBindGroup(0, this.gpuReadIsA ? this.bindBatchStateAB : this.bindBatchStateBA);
        pass.setBindGroup(1, this.bindBatchData);
        pass.dispatchWorkgroups(Math.ceil(activeCount / this.workgroupSize));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        this.gpuReadIsA = !this.gpuReadIsA;
        this.arena.revision += count;
        this.arena.swapCount += count;
        this.submissions++;
      } else {
        this.runCPUBatch(ids, weights, count, decay, inputGain, inhibition, baselinePull);
      }

      this.foregroundPasses += count;
      this.batchPasses++;
      this.batchedLexicalSteps += count;
      this.lastBatchSize = count;
      return true;
    }

    _dispatchActiveHomeostasisGPU(decay, inhibition, baselinePull) {
      if (!this.device || this.deviceLost) return false;
      const activeCount = this._uploadActiveSlots();
      if (!activeCount) return false;
      this.writeParams(decay, 0, inhibition, baselinePull, activeCount, 0);
      this.device.queue.writeBuffer(this.gpuParams, 0, this.paramBuffer);
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.homeostasisActivePipeline);
      pass.setBindGroup(0, this.gpuReadIsA ? this.bindHomeostasisStateAB : this.bindHomeostasisStateBA);
      pass.setBindGroup(1, this.bindHomeostasisData);
      pass.dispatchWorkgroups(Math.ceil(activeCount / this.workgroupSize));
      pass.end();
      this.device.queue.submit([encoder.finish()]);
      this.gpuReadIsA = !this.gpuReadIsA;
      this.arena.revision++;
      this.arena.swapCount++;
      this.submissions++;
      return true;
    }

    homeostasis(options = {}) {
      const decay = Number.isFinite(options.decay) ? options.decay : 0.997;
      const inhibition = Number.isFinite(options.inhibition) ? options.inhibition : 0.006;
      const baselinePull = Number.isFinite(options.baselinePull) ? options.baselinePull : 0.001;
      let changed = false;
      if (this.mode === 'webgpu') changed = this._dispatchActiveHomeostasisGPU(decay, inhibition, baselinePull);
      else changed = this.runCPUHomeostasis(decay, inhibition, baselinePull);
      if (changed) this.backgroundPasses++;
      return changed;
    }

    _ensureReadback() {
      if (this.readback || !this.device) return this.readback;
      this.readback = this.device.createBuffer({
        size: this.arena.stateDim * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });
      return this.readback;
    }

    async syncStateToCPU() {
      if (this.mode !== 'webgpu' || !this.device || this.deviceLost) return this.arena.stateEnergy();
      const readback = this._ensureReadback();
      if (readback.mapState === 'mapped') readback.unmap();
      const source = this.gpuReadIsA ? this.gpuStateA : this.gpuStateB;
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(source, 0, readback, 0, this.arena.stateDim * 4);
      this.device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const view = new Float32Array(readback.getMappedRange());
      this.arena.readState.set(view);
      this.arena.writeState.set(view);
      readback.unmap();
      this.arena.revision++;
      return this.arena.stateEnergy();
    }

    uploadCPUState() {
      if (this.mode !== 'webgpu' || !this.device || this.deviceLost) return;
      this.device.queue.writeBuffer(this.gpuStateA, 0, this.arena.readState);
      this.device.queue.writeBuffer(this.gpuStateB, 0, this.arena.readState);
      this.gpuReadIsA = true;
      // Active-slot data is uploaded lazily on the next active dispatch.
    }

    info() {
      return {
        mode: this.mode,
        deviceLost: this.deviceLost,
        stateDim: this.arena.stateDim,
        activeNeurons: this.arena.activeCount,
        activeRatio: this.arena.activeCount / Math.max(1, this.arena.stateDim),
        lastActiveDispatch: this.lastActiveDispatch,
        activeBufferGrowths: this.activeBufferGrowths,
        submissions: this.submissions,
        foregroundPasses: this.foregroundPasses,
        backgroundPasses: this.backgroundPasses,
        batchPasses: this.batchPasses,
        batchedLexicalSteps: this.batchedLexicalSteps,
        lastBatchSize: this.lastBatchSize,
        maxBatchSteps: this.maxBatchSteps,
        workgroupSize: this.workgroupSize,
        sparseActiveExecution: true,
        lazyReadbackAllocated: Boolean(this.readback),
        denseInputGPUAllocated: Boolean(this.gpuDenseInput),
        adapter: this.deviceInfo ? {
          vendor: this.deviceInfo.vendor || '',
          architecture: this.deviceInfo.architecture || '',
          device: this.deviceInfo.device || '',
          description: this.deviceInfo.description || ''
        } : null
      };
    }

    destroy() {
      for (const buffer of [
        this.gpuStateA, this.gpuStateB, this.gpuParams, this.gpuBatchIds,
        this.gpuBatchWeights, this.gpuActiveSlots, this.gpuDenseInput, this.readback
      ]) {
        try { buffer?.destroy?.(); } catch (_) {}
      }
      this.mode = 'destroyed';
    }
  }

  globalThis.VilotCompute = VilotCompute;
})();

/**
 * Compute production research extension.
 *
 * Executable, data-driven numerical/statistical/graph utilities used by
 * diagnostics, background learning, calibration, matrix experiments and
 * future compute paths. No prompt-specific phrases or benchmark answers live
 * here. Methods stay out of the foreground hot path unless explicitly called.
 */
(() => {
  'use strict';

  class VilotComputeProductionLab {
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

    matVec(matrix, rows, cols, vector) {
      const r=Math.max(0,rows|0), c=Math.max(0,cols|0);
      const out=new Float32Array(r);
      for (let i=0;i<r;i++) {
        let sum=0;
        const base=i*c;
        for (let j=0;j<c;j++) sum+=(Number(matrix[base+j])||0)*(Number(vector[j])||0);
        out[i]=sum;
      }
      return out;
    }

    outerUpdate(matrix, rows, cols, a, b, rate = 0.01) {
      const r=Math.min(rows|0,a?.length||0), c=Math.min(cols|0,b?.length||0);
      const eta=Number(rate)||0;
      for (let i=0;i<r;i++) {
        const av=Number(a[i])||0;
        const base=i*cols;
        for (let j=0;j<c;j++) matrix[base+j]+=eta*av*(Number(b[j])||0);
      }
      return matrix;
    }

    rowNormalize(matrix, rows, cols, epsilon = 1e-9) {
      for (let i=0;i<(rows|0);i++) {
        const base=i*cols;
        let norm2=0;
        for (let j=0;j<(cols|0);j++) { const v=Number(matrix[base+j])||0; norm2+=v*v; }
        const inv=1/(Math.sqrt(norm2)+epsilon);
        for (let j=0;j<(cols|0);j++) matrix[base+j]=(Number(matrix[base+j])||0)*inv;
      }
      return matrix;
    }

    vectorAdd(a, b, scale = 1) {
      const n=Math.max(a?.length||0,b?.length||0);
      const out=new Float32Array(n);
      const s=Number(scale)||0;
      for (let i=0;i<n;i++) out[i]=(Number(a?.[i])||0)+(Number(b?.[i])||0)*s;
      return out;
    }

    vectorLerp(a, b, t = 0.5) {
      const n=Math.max(a?.length||0,b?.length||0);
      const out=new Float32Array(n);
      const k=Math.max(0,Math.min(1,Number(t)||0));
      for (let i=0;i<n;i++) out[i]=(Number(a?.[i])||0)*(1-k)+(Number(b?.[i])||0)*k;
      return out;
    }

    orthogonalize(vector, basis) {
      const out=Float32Array.from(vector||[]);
      for (const b of basis||[]) {
        const denom=this.dot(b,b)||1;
        const scale=this.dot(out,b)/denom;
        for (let i=0;i<Math.min(out.length,b.length);i++) out[i]-=scale*(Number(b[i])||0);
      }
      return Float32Array.from(this.normalize(out));
    }

    powerIteration(matrix, n, iterations = 16) {
      const size=Math.max(1,n|0);
      let v=new Float32Array(size).fill(1/Math.sqrt(size));
      for (let step=0;step<Math.max(1,iterations|0);step++) {
        const next=this.matVec(matrix,size,size,v);
        v=Float32Array.from(this.normalize(next));
      }
      return v;
    }

    calibrationBins(pairs, bins = 10) {
      const count=Math.max(2,bins|0);
      const out=Array.from({length:count},(_,i)=>({lo:i/count,hi:(i+1)/count,n:0,confidence:0,accuracy:0}));
      for (const pair of pairs||[]) {
        const c=Math.max(0,Math.min(1,Number(pair.confidence)||0));
        const y=Number(pair.correct)||0;
        const idx=Math.min(count-1,Math.floor(c*count));
        const bin=out[idx];
        bin.n++; bin.confidence+=c; bin.accuracy+=y;
      }
      for (const bin of out) if (bin.n) { bin.confidence/=bin.n; bin.accuracy/=bin.n; }
      return out;
    }

    expectedCalibrationError(pairs, bins = 10) {
      const grouped=this.calibrationBins(pairs,bins);
      const total=grouped.reduce((s,b)=>s+b.n,0)||1;
      return grouped.reduce((s,b)=>s+(b.n/total)*Math.abs(b.accuracy-b.confidence),0);
    }

    brierScore(pairs) {
      let total=0,count=0;
      for (const pair of pairs||[]) {
        const p=Math.max(0,Math.min(1,Number(pair.confidence)||0));
        const y=Number(pair.correct)?1:0;
        const d=p-y;
        total+=d*d; count++;
      }
      return count?total/count:0;
    }

    novelty(candidate, history, similarity = (a,b) => a === b ? 1 : 0) {
      let best=0;
      for (const prior of history||[]) best=Math.max(best,Number(similarity(candidate,prior))||0);
      return Math.max(0,1-best);
    }

    diversity(items, similarity) {
      const input=Array.from(items||[]);
      if (input.length<2) return input.length?1:0;
      let total=0,count=0;
      for (let i=0;i<input.length;i++) for (let j=i+1;j<input.length;j++) {
        total+=1-Math.max(0,Math.min(1,Number(similarity(input[i],input[j]))||0));
        count++;
      }
      return count?total/count:0;
    }

    dedupeCandidates(items, key = x => String(x?.text ?? x)) {
      const best=new Map();
      for (const item of items||[]) {
        const k=key(item);
        const old=best.get(k);
        const score=Number(item?.confidence ?? item?.score)||0;
        const oldScore=Number(old?.confidence ?? old?.score)||-Infinity;
        if (!old||score>oldScore) best.set(k,item);
      }
      return Array.from(best.values());
    }

    agreement(scores) {
      const input=Array.from(scores||[],Number).filter(Number.isFinite);
      if (!input.length) return 0;
      const sd=this.stddev(input);
      return 1/(1+sd);
    }

    adaptiveRate(confidence, visits, base = 0.05) {
      const c=Math.max(0,Math.min(1,Number(confidence)||0));
      const v=Math.max(0,Number(visits)||0);
      return Math.max(1e-5,(Number(base)||0.05)*(0.25+0.75*(1-c))/Math.sqrt(1+v*0.05));
    }

    boundedUpdate(oldValue, observation, rate = 0.05, lo = -1, hi = 1) {
      const old=Number(oldValue)||0;
      const obs=Number(observation)||0;
      const eta=Math.max(0,Math.min(1,Number(rate)||0));
      return Math.max(lo,Math.min(hi,old+(obs-old)*eta));
    }

    temporalWeight(ageMs, halfLifeMs = 60000) {
      const age=Math.max(0,Number(ageMs)||0);
      const half=Math.max(1,Number(halfLifeMs)||1);
      return Math.pow(0.5,age/half);
    }

    reservoirPush(array, item, seen, capacity, random = Math.random) {
      const cap=Math.max(1,capacity|0);
      if (array.length<cap) { array.push(item); return true; }
      const j=Math.floor((random?.()??Math.random())*Math.max(1,seen));
      if (j<cap) { array[j]=item; return true; }
      return false;
    }

    chunk(items, size = 64) {
      const input=Array.from(items||[]), n=Math.max(1,size|0), out=[];
      for (let i=0;i<input.length;i+=n) out.push(input.slice(i,i+n));
      return out;
    }

    stableMerge(left, right, key = x => String(x)) {
      const out=[], seen=new Set();
      for (const item of [...(left||[]),...(right||[])]) {
        const k=key(item);
        if (seen.has(k)) continue;
        seen.add(k); out.push(item);
      }
      return out;
    }

    mapObject(object, fn) {
      const out={};
      for (const [key,value] of Object.entries(object||{})) out[key]=fn(value,key);
      return out;
    }

    serializeMap(map) {
      return Array.from(map instanceof Map ? map.entries() : []);
    }

    deserializeMap(entries) {
      return new Map(Array.isArray(entries)?entries:[]);
    }

    deepClone(value) {
      if (value == null || typeof value !== 'object') return value;
      if (ArrayBuffer.isView(value)) return value.slice ? value.slice() : new value.constructor(value);
      if (Array.isArray(value)) return value.map(item=>this.deepClone(item));
      if (value instanceof Map) return new Map(Array.from(value,([k,v])=>[k,this.deepClone(v)]));
      const out={};
      for (const [k,v] of Object.entries(value)) out[k]=this.deepClone(v);
      return out;
    }

    minMax(values) {
      let min=Infinity,max=-Infinity;
      for (const raw of values||[]) {
        const value=Number(raw);
        if (!Number.isFinite(value)) continue;
        if (value<min) min=value;
        if (value>max) max=value;
      }
      return {min:Number.isFinite(min)?min:0,max:Number.isFinite(max)?max:0};
    }

    histogram(values, bins = 16) {
      const input=Array.from(values||[],Number).filter(Number.isFinite);
      const count=Math.max(2,bins|0), out=new Uint32Array(count);
      if (!input.length) return {bins:out,min:0,max:0};
      const {min,max}=this.minMax(input), span=max-min||1;
      for (const value of input) out[Math.min(count-1,Math.floor(((value-min)/span)*count))]++;
      return {bins:out,min,max};
    }

    rank(values) {
      const pairs=Array.from(values||[],(value,index)=>({value:Number(value)||0,index})).sort((a,b)=>a.value-b.value);
      const out=new Float64Array(pairs.length);
      for (let i=0;i<pairs.length;i++) out[pairs[i].index]=i;
      return out;
    }

    status() {
      return {
        module: 'Compute',
        role: 'production-research-extension',
        calls: this.calls,
        methodCount: Object.getOwnPropertyNames(Object.getPrototypeOf(this)).length - 1,
        ageMs: (performance?.now?.() ?? Date.now()) - this.createdAt
      };
    }
  }

  globalThis.VilotComputeProductionLab = VilotComputeProductionLab;
})();
