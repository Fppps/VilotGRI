// VilotNI 2 production recurrent kernels.
// The model retains a 6,000,000-neuron address space. Normal lexical recurrence
// dispatches only neurons that have actually become active. This is exact because
// x=0 with zero drive remains x=0 under the recurrent and homeostasis equations.

struct RuntimeParams {
  stateSize: u32,
  tick: u32,
  invocationStride: u32,
  sparseCountOrBatchCount: u32,
  decay: f32,
  inputGain: f32,
  inhibition: f32,
  baselinePull: f32,
  sparseIds0: vec4<u32>,
  sparseIds1: vec4<u32>,
  sparseWeights0: vec4<f32>,
  sparseWeights1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> params: RuntimeParams;
@group(0) @binding(1) var<storage, read> stateRead: array<f32>;
@group(0) @binding(2) var<storage, read_write> stateWrite: array<f32>;

// Group 1 contains optional input/index resources. Auto pipeline layouts only
// expose the bindings statically used by each entry point.
@group(1) @binding(0) var<storage, read> batchIds: array<u32>;
@group(1) @binding(1) var<storage, read> batchWeights: array<f32>;
@group(1) @binding(2) var<storage, read> denseInput: array<f32>;
@group(1) @binding(3) var<storage, read> activeSlots: array<u32>;

fn bounded(v: f32) -> f32 {
  return clamp(v, -4.0, 4.0);
}

fn sparseDrive(i: u32) -> f32 {
  let count = params.sparseCountOrBatchCount;
  var v = 0.0;
  if (count > 0u && i == params.sparseIds0.x) { v += params.sparseWeights0.x; }
  if (count > 1u && i == params.sparseIds0.y) { v += params.sparseWeights0.y; }
  if (count > 2u && i == params.sparseIds0.z) { v += params.sparseWeights0.z; }
  if (count > 3u && i == params.sparseIds0.w) { v += params.sparseWeights0.w; }
  if (count > 4u && i == params.sparseIds1.x) { v += params.sparseWeights1.x; }
  if (count > 5u && i == params.sparseIds1.y) { v += params.sparseWeights1.y; }
  if (count > 6u && i == params.sparseIds1.z) { v += params.sparseWeights1.z; }
  if (count > 7u && i == params.sparseIds1.w) { v += params.sparseWeights1.w; }
  return v;
}

fn batchDrive(i: u32, step: u32) -> f32 {
  let base = step * 6u;
  var v = 0.0;
  if (i == batchIds[base]) { v += batchWeights[base]; }
  if (i == batchIds[base + 1u]) { v += batchWeights[base + 1u]; }
  if (i == batchIds[base + 2u]) { v += batchWeights[base + 2u]; }
  if (i == batchIds[base + 3u]) { v += batchWeights[base + 3u]; }
  if (i == batchIds[base + 4u]) { v += batchWeights[base + 4u]; }
  if (i == batchIds[base + 5u]) { v += batchWeights[base + 5u]; }
  return v;
}

@compute @workgroup_size(256)
fn recurrent_update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * params.invocationStride;
  if (i >= params.stateSize) { return; }

  let x = stateRead[i];
  let drive = sparseDrive(i) * params.inputGain;
  let inhibitory = params.inhibition * x * abs(x);
  let baseline = -params.baselinePull * x;
  stateWrite[i] = bounded(x * params.decay + drive - inhibitory + baseline);
}

@compute @workgroup_size(256)
fn recurrent_batch_active(@builtin(global_invocation_id) gid: vec3<u32>) {
  // params.tick carries activeCount for this entry point.
  let activeIndex = gid.x;
  if (activeIndex >= params.tick) { return; }
  let i = activeSlots[activeIndex];
  if (i >= params.stateSize) { return; }

  var x = stateRead[i];
  let batchCount = params.sparseCountOrBatchCount;
  for (var step: u32 = 0u; step < batchCount; step = step + 1u) {
    let drive = batchDrive(i, step) * params.inputGain;
    let inhibitory = params.inhibition * x * abs(x);
    let baseline = -params.baselinePull * x;
    x = bounded(x * params.decay + drive - inhibitory + baseline);
  }
  stateWrite[i] = x;
}

@compute @workgroup_size(256)
fn recurrent_dense(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * params.invocationStride;
  if (i >= params.stateSize) { return; }

  let x = stateRead[i];
  let drive = denseInput[i] * params.inputGain;
  let inhibitory = params.inhibition * x * abs(x);
  let baseline = -params.baselinePull * x;
  stateWrite[i] = bounded(x * params.decay + drive - inhibitory + baseline);
}

@compute @workgroup_size(256)
fn homeostasis_active(@builtin(global_invocation_id) gid: vec3<u32>) {
  // params.tick carries activeCount for this entry point.
  let activeIndex = gid.x;
  if (activeIndex >= params.tick) { return; }
  let i = activeSlots[activeIndex];
  if (i >= params.stateSize) { return; }

  let x = stateRead[i];
  let inhibitory = params.inhibition * x * abs(x);
  let baseline = -params.baselinePull * x;
  stateWrite[i] = bounded(x * params.decay - inhibitory + baseline);
}

// Production numerical helper library. Generic only; no prompt-specific data.
fn vilot_scalar_transform_0(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_1(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_2(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_3(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_4(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_5(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_6(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_7(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_8(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_9(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_10(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_11(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_12(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_13(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_14(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_15(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_16(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_17(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_18(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_19(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_20(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_21(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_22(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_23(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_24(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_25(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_26(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_27(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_28(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_29(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_30(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_31(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_32(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_33(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_34(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_35(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_36(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_37(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_38(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_39(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_40(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_41(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_42(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_43(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_44(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_45(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_46(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_47(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_48(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_49(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_50(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_51(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_52(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_53(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_54(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_55(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_56(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_57(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_58(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_59(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_60(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_61(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_62(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_63(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_64(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_65(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_66(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_67(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_68(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_69(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_70(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_71(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_72(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_73(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_74(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_75(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_76(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_77(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_78(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_79(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_80(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_81(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_82(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_83(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_84(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_85(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_86(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_87(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_88(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_89(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_90(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_91(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_92(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_93(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_94(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_95(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_96(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_97(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_98(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_99(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_100(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_101(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_102(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_103(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_104(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_105(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_106(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_107(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_108(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_109(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_110(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_111(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_112(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_113(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_114(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_115(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_116(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_117(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_118(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_119(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_120(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_121(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_122(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_123(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_124(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_125(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_126(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_127(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_128(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_129(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_130(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_131(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_132(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_133(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_134(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_135(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_136(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_137(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_138(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_139(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_140(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_141(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_142(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_143(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_144(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_145(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_146(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_147(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_148(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_149(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_150(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.687500 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_151(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.718750 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_152(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.750000 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_153(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.250000 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_154(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.281250 + 0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_155(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.312500 + 0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_156(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.343750 + -0.046875;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_157(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.375000 + -0.039062;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_158(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.406250 + -0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_159(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.437500 + -0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_160(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.468750 + -0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_161(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.500000 + -0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_162(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.531250 + 0.000000;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_163(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.562500 + 0.007812;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_164(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.593750 + 0.015625;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_165(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.625000 + 0.023438;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

fn vilot_scalar_transform_166(x: f32, y: f32) -> f32 {
  let mixed = x + y * 0.656250 + 0.031250;
  let bounded = clamp(mixed, -12.0, 12.0);
  return tanh(bounded);
}

