import { describe, test, expect, beforeEach } from 'vitest';
import { comb } from '../../src/utils/utils';

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
