// Kernel SHAP

/**
 * Compute accuracy score for classification.
 * @param yTrue 1d array of ground-truth labels
 * @param yPred 1d array of predicted labels
 * @returns accuracy score
 */
export const accuracyScore = (yTrue: number[], yPred: number[]): number => {
  if (yTrue.length !== yPred.length) {
    throw 'yTrue and yPred need to have the same length.';
  }

  const correctNum = yTrue.reduce((a, b, i) => (yPred[i] === b ? a + 1 : a), 0);
  return correctNum / yTrue.length;
};
