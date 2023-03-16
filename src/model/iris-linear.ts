// A simple logistic regression model for Iris dataset

/**
 * A logistic regression model for Iris dataset.
 */
export class IrisLinearBinary {
  coef: number[];
  intercept: number;

  constructor(coef: number[], intercept: number) {
    this.coef = coef;
    this.intercept = intercept;
  }

  /**
   * Predict on one data point
   * @param xs Array of data points to predict
   * @returns Predicted classes
   */
  predict = (xs: number[][]) => {
    const yProbs = this.predictProba(xs);
    return yProbs.map(d => (d >= 0.5 ? 1 : 0));
  };

  /**
   * Predict on one data point
   * @param xs Array of data points to predict
   * @returns Predicted probabilities
   */
  predictProba = (xs: number[][]) => {
    const yProbs: number[] = [];

    for (const x of xs) {
      // Compute the logit score
      let logit = this.intercept;
      for (const [i, v] of x.entries()) {
        logit += this.coef[i] * v;
      }

      // Convert logit to a probability through sigmoid
      yProbs.push(sigmoid(logit));
    }

    return yProbs;
  };
}

const sigmoid = (x: number) => {
  return Math.exp(x) / (Math.exp(x) + 1);
};
