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
    return yProbs.map(d => (d[0] >= 0.5 ? 1 : 0));
  };

  /**
   * Predict on one data point
   * @param xs Array of data points to predict
   * @returns Predicted probabilities
   */
  predictProba = (xs: number[][]) => {
    const yProbs: number[][] = [];

    for (const x of xs) {
      // Compute the logit score
      let logit = this.intercept;
      for (const [i, v] of x.entries()) {
        logit += this.coef[i] * v;
      }

      // Convert logit to a probability through sigmoid
      yProbs.push([sigmoid(logit)]);
    }

    return yProbs;
  };
}

/**
 * A logistic regression model for Iris dataset.
 */
export class IrisLinearMultiClass {
  coefs: number[][];
  intercepts: number[];

  constructor(coefs: number[][], intercepts: number[]) {
    this.coefs = coefs;
    this.intercepts = intercepts;
  }

  /**
   * Predict on one data point
   * @param xs Array of data points to predict
   * @returns Predicted classes
   */
  predict = (xs: number[][]) => {
    const yProbs = this.predictProba(xs);
    const yPredicts: number[] = [];

    // Use the class with the largest probability as prediction output
    for (const row of yProbs) {
      let maxValue = -Infinity;
      let maxIndex = -1;
      for (let i = 0; i < row.length; i++) {
        if (row[i] > maxValue) {
          maxValue = row[i];
          maxIndex = i;
        }
      }
      yPredicts.push(maxIndex);
    }

    return yPredicts;
  };

  /**
   * Predict on one data point
   * @param xs Array of data points to predict
   * @returns Predicted probabilities
   */
  predictProba = (xs: number[][]) => {
    const yProbs: number[][] = [];

    for (const row of xs) {
      // Compute the logits
      const logits = [...this.intercepts];

      for (let t = 0; t < this.coefs.length; t++) {
        for (let c = 0; c < this.coefs[t].length; c++) {
          logits[t] += this.coefs[t][c] * row[c];
        }
      }

      // Convert logit to a probability through softmax
      yProbs.push(softmax(logits));
    }

    return yProbs;
  };
}

/**
 * Compute sigmoid on the given logit
 * @param logit Logit
 * @returns Probability value
 */
const sigmoid = (logit: number) => {
  return Math.exp(logit) / (Math.exp(logit) + 1);
};

/**
 * Compute softmax probabilities on the given logits
 * @param logits Logits
 * @returns Probabilities [nTargets]
 */
const softmax = (logits: number[]) => {
  const exps = logits.map(d => Math.exp(d));
  const expSum = exps.reduce((a, b) => a + b);
  return exps.map(d => d / expSum);
};
