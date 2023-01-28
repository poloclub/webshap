import { describe, test, expect, beforeEach } from 'vitest';
import { KernelSHAP } from '../../src/index';
import { IrisLinearBinary } from '../../src/index';
import type { SHAPModel } from '../../src/my-types';
import math from '../../src/utils/math-import';

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
  const explainer = new KernelSHAP(model, data, 0.20071022);

  for (const [i, pred] of explainer.predictions.entries()) {
    expect(pred).toBeCloseTo(yPredProbaExp[i], 6);
  }
});

test<LocalTestContext>('prepareSampling()', ({ model, data }) => {
  //
  const aa = math.matrix(math.zeros(3));
});
