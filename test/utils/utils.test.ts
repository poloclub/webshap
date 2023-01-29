import { describe, test, expect, beforeEach } from 'vitest';
import { comb, getCombinations } from '../../src/utils/utils';

test('comb()', () => {
  expect(comb(10, 1)).toBe(10);
  expect(comb(10, 0)).toBe(1);
  expect(comb(10, 0)).toBe(1);
  expect(comb(15, 3)).toBe(455);
  expect(comb(15, 3)).toBe(455);
  expect(comb(25, 18)).toBe(480700);
  expect(comb(25, 18)).toBe(480700);
  expect(comb(100, 5)).toBe(75287520);
  expect(comb(100, 5)).toBe(75287520);
});

test('getCombinations()', () => {
  const myArray = [1, 2, 3, 4];

  expect(getCombinations(myArray, 2)).toEqual([
    [1, 2],
    [1, 3],
    [1, 4],
    [2, 3],
    [2, 4],
    [3, 4]
  ]);

  expect(getCombinations(myArray, 3)).toEqual([
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 4],
    [2, 3, 4]
  ]);

  expect(getCombinations(myArray, 4)).toEqual([[1, 2, 3, 4]]);
});
