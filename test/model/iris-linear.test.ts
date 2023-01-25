import { beforeEach, test, expect } from 'vitest';
import { IrisLinearBinary } from '../../src/index';

interface LocalTestContext {
  model: IrisLinearBinary;
  xTest: number[][];
}

/**
 * Initialize the fixture
 */
beforeEach<LocalTestContext>(context => {
  const coef = [-0.1991, 0.3426, 0.0478, 1.03745];
  const intercept = -1.6689;
  context.model = new IrisLinearBinary(coef, intercept);

  context.xTest = [
    [5.8, 2.8, 5.1, 2.4],
    [5.8, 2.7, 5.1, 1.9],
    [7.2, 3.6, 6.1, 2.5],
    [6.2, 2.8, 4.8, 1.8],
    [4.9, 3.1, 1.5, 0.1]
  ];
});

test<LocalTestContext>('predictProba()', ({ model, xTest }) => {
  const yProbs = model.predictProba(xTest);
  const yProbsExp = [0.7045917, 0.57841617, 0.73422101, 0.53812833, 0.19671004];
  for (const [i, p] of yProbs.entries()) {
    expect(p).toBeCloseTo(yProbsExp[i], 5);
  }
});

test<LocalTestContext>('predict()', ({ model, xTest }) => {
  const yPred = model.predict(xTest);
  const yPredExp = [1, 1, 1, 1, 0];
  expect(yPred).toEqual(yPredExp);
});
