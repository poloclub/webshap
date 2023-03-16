import { describe, test, expect, beforeEach } from 'vitest';
import { KernelSHAP, IrisLinearMultiClass } from '../../src/index';
import math from '../../src/utils/math-import';

const SEED = 0.20071022;

interface LocalTestContext {
  model: (x: number[][]) => Promise<number[][]>;
  data: number[][];
}

/**
 * Initialize the fixture for all tests
 */
beforeEach<LocalTestContext>(context => {
  const coefs = [
    [-0.39228899, 0.85674351, -2.23337115, -0.98440396],
    [0.55874815, -0.25246978, -0.11357907, -0.90513218],
    [-0.16645916, -0.60427374, 2.34695022, 1.88953615]
  ];
  const intercepts = [8.85065312, 1.21877625, -10.06942938];
  const model = new IrisLinearMultiClass(coefs, intercepts);

  // Wrap the model in a promise
  context.model = (x: number[][]) => {
    const promise = new Promise<number[][]>((resolve, reject) => {
      const prob = model.predictProba(x);
      resolve(prob);
    });
    return promise;
  };

  context.data = [
    [5.8, 2.8, 5.1, 2.4],
    [5.8, 2.7, 5.1, 1.9],
    [7.2, 3.6, 6.1, 2.5],
    [6.2, 2.8, 4.8, 1.8],
    [4.9, 3.1, 1.5, 0.1]
  ];
});

test<LocalTestContext>('constructor()', async ({ model, data }) => {
  const yPredProbaExp = [
    [1.81082686e-4, 5.85998237e-2, 9.41219094e-1],
    [5.62410593e-4, 1.95448232e-1, 8.03989357e-1],
    [3.42078623e-6, 1.44898582e-2, 9.85506721e-1],
    [2.01336494e-3, 4.81186435e-1, 5.168002e-1],
    [9.61680907e-1, 3.83188489e-2, 2.44545042e-7]
  ];

  const explainer = new KernelSHAP(model, data, SEED);
  await explainer.initializeModel();

  for (let i = 0; i < explainer.predictions.length; i++) {
    const curRow = explainer.predictions[i];
    for (let j = 0; j < curRow.length; j++) {
      expect(curRow[j]).toBeCloseTo(yPredProbaExp[i][j], 6);
    }
  }
});
