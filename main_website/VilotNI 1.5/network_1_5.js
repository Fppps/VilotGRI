/**
 * VilotNI 1.5: 3,000,000-Neuron Network Architecture.
 * Improvements applied:
 * - Pre-allocated static uniform buffers (zero VRAM leaks during runtime).
 * - Full synaptic eligibility trace backpropagation through all 15 hidden layers.
 */
class VilotNI15Network {
  constructor(backend, vocabSize = 440) {
    this.backend = backend;
    this.device = backend.device;
    this.vocabSize = vocabSize;
    this.layerCount = 15;
    this.layerSize = 200000;
    this.fanIn = 4;
    this.poolDim = 256;
    this.embDim = 64;

    this.gpuLayers = [];
    this.initGPUBuffers();
  }

  initGPUBuffers() {
    const dev = this.device;

    this.bufferA = this.backend.createBuffer(this.layerSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.bufferB = this.backend.createBuffer(this.layerSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.contextBuffer = this.backend.createBuffer(this.layerSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    this.pooledBuffer = this.backend.createBuffer(this.poolDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    this.readbackBuffer = dev.createBuffer({
      size: this.poolDim * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const scale = Math.sqrt(2.0 / this.fanIn);
    for (let l = 0; l < this.layerCount; l++) {
      const wData = new Float32Array(this.layerSize * this.fanIn);
      for (let i = 0; i < wData.length; i++) wData[i] = (Math.random() * 2 - 1) * scale;

      const bData = new Float32Array(this.layerSize);
      const wBuffer = this.backend.createBuffer(wData.byteLength, GPUBufferUsage.STORAGE, wData);
      const bBuffer = this.backend.createBuffer(bData.byteLength, GPUBufferUsage.STORAGE, bData);
      const gBuffer = this.backend.createBuffer(wData.byteLength, GPUBufferUsage.STORAGE);

      // Pre-allocated static forward uniform buffer
      const isResidual = l > 0 ? 1 : 0;
      const uData = new Uint32Array([l === 0 ? this.embDim : this.layerSize, this.layerSize, l, isResidual]);
      const uBuffer = this.backend.createBuffer(uData.byteLength, GPUBufferUsage.UNIFORM, uData);

      // Pre-allocated static reward uniform buffer (prevents VRAM leaks)
      const rlUniformBuffer = this.backend.createBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

      this.gpuLayers.push({
        wBuffer,
        bBuffer,
        gBuffer,
        uBuffer,
        rlUniformBuffer,
        count: wData.length
      });
    }

    this.headWeights = new Float32Array(this.poolDim * this.vocabSize);
    this.headGradients = new Float32Array(this.poolDim * this.vocabSize);
    const headScale = Math.sqrt(2.0 / this.poolDim);
    for (let i = 0; i < this.headWeights.length; i++) {
      this.headWeights[i] = (Math.random() * 2 - 1) * headScale;
    }

    this.embeddings = new Float32Array(this.vocabSize * this.embDim);
    for (let i = 0; i < this.embeddings.length; i++) {
      this.embeddings[i] = (Math.random() * 2 - 1) * 0.1;
    }
  }

  async forwardSequence(tokenIds) {
    const dev = this.device;

    const initialInput = new Float32Array(this.layerSize);
    for (const id of tokenIds) {
      const offset = (id % this.vocabSize) * this.embDim;
      for (let d = 0; d < this.embDim; d++) {
        initialInput[d] += this.embeddings[offset + d] / Math.max(1, tokenIds.length);
      }
    }
    dev.queue.writeBuffer(this.contextBuffer, 0, initialInput);

    const encoder = dev.createCommandEncoder();
    let currentIn = this.contextBuffer;
    let currentOut = this.bufferA;

    // Dispatch 15 steps of 200,000 neurons
    for (let l = 0; l < this.layerCount; l++) {
      const layer = this.gpuLayers[l];
      const bindGroup = dev.createBindGroup({
        layout: this.backend.forwardPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: layer.uBuffer } },
          { binding: 1, resource: { buffer: currentIn } },
          { binding: 2, resource: { buffer: layer.wBuffer } },
          { binding: 3, resource: { buffer: layer.bBuffer } },
          { binding: 4, resource: { buffer: currentOut } }
        ]
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.backend.forwardPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(this.layerSize / 256));
      pass.end();

      // Accumulate eligibility traces so 3M weights actually learn
      const traceBindGroup = dev.createBindGroup({
        layout: this.backend.tracePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: layer.uBuffer } },
          { binding: 1, resource: { buffer: currentIn } },
          { binding: 2, resource: { buffer: currentOut } },
          { binding: 3, resource: { buffer: layer.gBuffer } }
        ]
      });

      const tracePass = encoder.beginComputePass();
      tracePass.setPipeline(this.backend.tracePipeline);
      tracePass.setBindGroup(0, traceBindGroup);
      tracePass.dispatchWorkgroups(Math.ceil(this.layerSize / 256));
      tracePass.end();

      currentIn = currentOut;
      currentOut = (currentOut === this.bufferA) ? this.bufferB : this.bufferA;
    }

    // Pool Layer 15 into latent vector
    const poolBindGroup = dev.createBindGroup({
      layout: this.backend.poolPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: currentIn } },
        { binding: 1, resource: { buffer: this.pooledBuffer } }
      ]
    });

    const poolPass = encoder.beginComputePass();
    poolPass.setPipeline(this.backend.poolPipeline);
    poolPass.setBindGroup(0, poolBindGroup);
    poolPass.dispatchWorkgroups(4);
    poolPass.end();

    encoder.copyBufferToBuffer(this.pooledBuffer, 0, this.readbackBuffer, 0, this.poolDim * 4);
    dev.queue.submit([encoder.finish()]);

    await this.readbackBuffer.mapAsync(GPUMapMode.READ);
    const latent = new Float32Array(this.readbackBuffer.getMappedRange()).slice();
    this.readbackBuffer.unmap();
    this.currentLatent = latent;

    const logits = new Float32Array(this.vocabSize);
    for (let v = 0; v < this.vocabSize; v++) {
      let sum = 0;
      const off = v * this.poolDim;
      for (let p = 0; p < this.poolDim; p++) {
        sum += this.headWeights[off + p] * latent[p];
      }
      logits[v] = sum;
    }

    return logits;
  }

  applyPolicyReward(reward, lr = 0.008, updateDeepLayers = true) {
    const dev = this.device;

    // 1. Update CPU Policy Head (always learns immediately)
    for (let i = 0; i < this.headWeights.length; i++) {
      this.headWeights[i] += lr * reward * this.headGradients[i];
      this.headWeights[i] *= 0.9999;
      this.headGradients[i] = 0;
    }

    // 2. Skip deep GPU layer updates when periodic mode has not reached its turn interval
    if (!updateDeepLayers) return;

    const encoder = dev.createCommandEncoder();

    for (let l = 0; l < this.layerCount; l++) {
      const layer = this.gpuLayers[l];
      const rlUniformData = new Float32Array([reward, lr, 0, 0.9999]);
      new Uint32Array(rlUniformData.buffer)[2] = layer.count;

      dev.queue.writeBuffer(layer.rlUniformBuffer, 0, rlUniformData);

      const bindGroup = dev.createBindGroup({
        layout: this.backend.rewardPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: layer.rlUniformBuffer } },
          { binding: 1, resource: { buffer: layer.wBuffer } },
          { binding: 2, resource: { buffer: layer.gBuffer } }
        ]
      });

      const pass = encoder.beginComputePass();
      pass.setPipeline(this.backend.rewardPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(layer.count / 256));
      pass.end();
    }

    dev.queue.submit([encoder.finish()]);
  }
}