/**
 * Compute n choose k
 * @param n n to choose from
 * @param k to choose k
 * @returns Result
 */
export const comb = (n: number, k: number): number => {
  const minK = Math.min(k, n - k);
  return Array.from(new Array(minK), (_, i) => i + 1).reduce(
    (a, b) => (a * (n + 1 - b)) / b,
    1
  );
};
