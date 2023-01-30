import { describe, test, expect, beforeEach } from 'vitest';
import { KernelSHAP, IrisLinearBinary } from '../../src/index';
import math from '../../src/utils/math-import';
import type { SHAPModel } from '../../src/my-types';

const SEED = 0.20071022;

interface LocalTestContext {
  model: SHAPModel;
  data: number[][];
}

/**
 * Initialize the fixture for all tests
 */
beforeEach<LocalTestContext>(context => {
  const coef = [-0.1991, 0.3426, 0.0478, 1.03745];
  const intercept = -1.6689;
  const model = new IrisLinearBinary(coef, intercept);
  context.model = (x: number[][]) => model.predictProba(x);
  context.data = [
    [5.8, 2.8, 5.1, 2.4],
    [5.8, 2.7, 5.1, 1.9],
    [7.2, 3.6, 6.1, 2.5],
    [6.2, 2.8, 4.8, 1.8],
    [4.9, 3.1, 1.5, 0.1]
  ];
});

test<LocalTestContext>('constructor()', ({ model, data }) => {
  const yPredProbaExp = [
    0.7045917, 0.57841617, 0.73422101, 0.53812833, 0.19671004
  ];
  const explainer = new KernelSHAP(model, data, SEED);

  for (const [i, pred] of explainer.predictions.entries()) {
    expect(pred).toBeCloseTo(yPredProbaExp[i], 6);
  }
});

test<LocalTestContext>('prepareSampling()', ({ model, data }) => {
  const explainer = new KernelSHAP(model, data, SEED);
  const nSamples = 14;
  explainer.prepareSampling(nSamples);

  // The sample data should be initialized to repeat x_test
  const sampledData = explainer.sampledData!;
  expect(sampledData.size()[0]).toBe(nSamples * data.length);
  expect(sampledData.subset(math.index(0, 0))).toBe(data[0][0]);
  expect(sampledData.subset(math.index(data.length, 1))).toBe(data[0][1]);
  expect(sampledData.subset(math.index(sampledData.size()[0] - 1, 2))).toBe(
    data[data.length - 1][2]
  );
});

test<LocalTestContext>('sampleFeatureCoalitions()', ({ model, data }) => {
  const explainer = new KernelSHAP(model, data, SEED);
  const x1 = [4.8, 3.8, 2.1, 5.4];

  explainer.sampleFeatureCoalitions(x1, 32);

  // The number of samples should be overridden
  const sampledData = explainer.sampledData!;
  expect(sampledData.size()[0]).toBe(14 * data.length);

  // Size = 1 should be fully sampled
  const maskMat = explainer.maskMat!;
  const maskStrings = new Set<string>();
  for (let i = 0; i < maskMat.size()[0]; i++) {
    const row = math.row(maskMat, i).toArray()[0] as number[];
    maskStrings.add(KernelSHAP.getMaskStr(row));
  }

  const size1Comb = ['1000', '0100', '0010', '0001'];
  const size1CombComp = ['0111', '1011', '1101', '1110'];

  for (const comb of size1Comb) {
    expect(maskStrings.has(comb)).toBeTruthy();
  }

  for (const comb of size1CombComp) {
    expect(maskStrings.has(comb)).toBeTruthy();
  }

  // Kernel weights should sum up to 1
  const weightSum = math.sum(explainer.kernelWeight!) as number;
  expect(weightSum).toBeCloseTo(1, 6);

  // The weights for the enumeration of sample=1 combination should be
  // 0.09090909
  for (let i = 0; i < 8; i++) {
    expect(explainer.kernelWeight!.get([i, 0])).toBeCloseTo(0.09090909, 8);
  }

  // Verify tracker variables
  expect(explainer.nSamplesAdded).toBe(14);
});

test<LocalTestContext>('addSample() basic', ({ model, data }) => {
  const explainer = new KernelSHAP(model, data, SEED);
  const nSamples = 14;

  // Initialize the sample data
  explainer.prepareSampling(nSamples);

  // Test adding a sample
  const x1 = [4.8, 3.8, 2.1, 5.4];
  const mask1 = [1.0, 0.0, 1.0, 0.0];
  const weight1 = 0.52;
  explainer.addSample(x1, mask1, weight1);
  const sampledData = explainer.sampledData!;

  // Only the first and their elements are changed from the background
  for (let i = 0; i < data.length; i++) {
    const row = math.row(sampledData, i).toArray()[0];
    const rowExp = [x1[0], data[i][1], x1[2], data[i][3]];
    expect(row).toEqual(rowExp);
  }

  // Test if all other repetitions of the background data remain the same
  for (let i = 1; i < nSamples; i++) {
    for (let j = 0; j < data.length; j++) {
      const row = math.row(sampledData, i * data.length + j).toArray()[0];
      expect(row).toEqual(data[j]);
    }
  }

  // Test tracking variables
  expect(explainer.kernelWeight!.get([0, 0])).toBe(weight1);
  expect(explainer.nSamplesAdded).toBe(1);
});

test<LocalTestContext>('addSample() more complex', ({ model, data }) => {
  const explainer = new KernelSHAP(model, data, SEED);
  const nSamples = 14;

  // Initialize the sample data
  explainer.prepareSampling(nSamples);

  // Test adding a sample
  const x1 = [4.8, 3.8, 2.1, 5.4];
  const mask1 = [1.0, 0.0, 1.0, 0.0];
  const weight1 = 0.52;
  explainer.addSample(x1, mask1, weight1);

  const x2 = [11.2, 11.2, 11.2, 11.2];
  const mask2 = [1.0, 1.0, 0.0, 1.0];
  const weight2 = 0.99;
  explainer.addSample(x2, mask2, weight2);

  const sampledData = explainer.sampledData!;

  // The first repetition should match x_1 and mask_1
  for (let i = 0; i < data.length; i++) {
    const row = math.row(sampledData, i).toArray()[0];
    const rowExp = [x1[0], data[i][1], x1[2], data[i][3]];
    expect(row).toEqual(rowExp);
  }

  // The second repetition should match x_2 and mask_2
  for (let i = 0; i < data.length; i++) {
    const r = data.length + i;
    const row = math.row(sampledData, r).toArray()[0];
    const rowExp = [x2[0], x2[1], data[i][2], x2[3]];
    expect(row).toEqual(rowExp);
  }

  // Test if all other repetitions of the background data remain the same
  for (let i = 2; i < nSamples; i++) {
    for (let j = 0; j < data.length; j++) {
      const row = math.row(sampledData, i * data.length + j).toArray()[0];
      expect(row).toEqual(data[j]);
    }
  }

  // Test tracking variables
  expect(explainer.kernelWeight!.get([1, 0])).toBe(weight2);
  expect(explainer.nSamplesAdded).toBe(2);
});

test<LocalTestContext>('inferenceFeatureCoalitions()', ({ model, data }) => {
  const explainer = new KernelSHAP(model, data, SEED);
  const nSamples = 32;
  const x1 = [4.8, 3.8, 2.1, 5.4];

  // Inference on the sampled feature coalitions
  explainer.sampleFeatureCoalitions(x1, nSamples);
  explainer.inferenceFeatureCoalitions();

  // The first 8 masks (40 samples) are deterministic, so we compare the
  // results of them with SHAP
  const expectedYMat8 = [
    0.74428491, 0.62606565, 0.81667565, 0.60624373, 0.19987513, 0.98494403,
    0.98494403, 0.98019991, 0.98371625, 0.98738284, 0.77062789, 0.66666396,
    0.74737578, 0.62138006, 0.23736781, 0.98266107, 0.98206758, 0.98676269,
    0.98266107, 0.9843281, 0.67389612, 0.54311144, 0.69528502, 0.50593722,
    0.20128136, 0.98926348, 0.98926348, 0.98975948, 0.9891101, 0.98727311,
    0.98168608, 0.98105986, 0.98244577, 0.9799177, 0.98356063, 0.78032477,
    0.6789248, 0.79759091, 0.65590315, 0.2462757
  ];
  for (let i = 0; i < expectedYMat8.length; i++) {
    expect(explainer.yMat!.get([i, 0]) as number).toBeCloseTo(
      expectedYMat8[i],
      8
    );
  }

  // Compare the expected y value
  const expectedEyMat8 = [
    0.59862901, 0.98423741, 0.6086831, 0.9836961, 0.52390223, 0.98893393,
    0.98173401, 0.63180387
  ];
  for (let i = 0; i < expectedEyMat8.length; i++) {
    expect(explainer.yExpMat!.get([i, 0]) as number).toBeCloseTo(
      expectedEyMat8[i],
      8
    );
  }
});

test<LocalTestContext>('explainOneInstance()', ({ model, data }) => {
  const explainer = new KernelSHAP(model, data, SEED);
  const nSamples = 32;
  const x1 = [4.8, 3.8, 2.1, 5.4];

  const values = explainer.explainOneInstance(x1, nSamples)[0];
  const valuesExp = [0.02968265, 0.03134839, -0.0162967, 0.39248069];

  for (const [i, value] of values.entries()) {
    expect(value).toBeCloseTo(valuesExp[i], 6);
  }
});
