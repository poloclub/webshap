import { describe, test, expect } from 'vitest';
import { accuracyScore } from '../../src/index';

let yTrue: number[] = [];
let yPred: number[] = [];

test('basic', () => {
  yTrue = [1, 0, 1, 1, 1];
  yPred = [1, 0, 0, 1, 1];

  expect(accuracyScore(yTrue, yPred)).toEqual(4 / 5);
});

test('different length', () => {
  yTrue = [1, 0, 1, 1, 1];
  yPred = [1, 0, 0];

  expect(() => accuracyScore(yTrue, yPred)).toThrowError();
});
