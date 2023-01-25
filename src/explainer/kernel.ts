/**
 * Kernel SHAP
 * @author: Jay Wang (jay@zijie.wang)
 */

import { randomLcg, randomUniform } from 'd3-random';
import type { RandomUniform } from 'd3-random';

/**
 * A model that outputs a 1D vector (binary classification, regression)
 */
type SHAPModel = (x: number[][]) => number[];

/**
 * Kernel SHAP method to approximate Shapley attributions by solving s specially
 * weighted linear regression problem.
 */
export class KernelSHAP {
  /** Prediction model */
  model: SHAPModel;

  /** Background data */
  data: number[][];

  /** Expected prediction value */
  expectedValue: number;

  /** Model's prediction on the background ata */
  predictions: number[];

  /** Number of features */
  nFeatures: number;

  /** Dimension of the prediction output */
  nTargets: number;

  /** Number of coalition samples added */
  nSamplesAdded: number;

  /** Sampled data in a matrix form. It is initialized after the explain() call. */
  sampledData: number[][] | null = null;

  /** Matrix to store the feature masks */
  maskMat: number[][] | null = null;

  /** Kernel weights for each coalition sample */
  kernelWeight: number[][] | null = null;

  /** Model prediction outputs on the sampled data */
  yMat: number[][] | null = null;

  /** Expected model predictions on the sample data */
  yExpMat: number[][] | null = null;

  /** Mask used in the last run */
  lastMask: number[][] | null = null;

  /** */
  rng: RandomUniform;

  /**
   * Initialize a new KernelSHAP explainer.
   * @param model The trained model to explain
   * @param data The background data
   * @param seed Optional random seed in the range [0, 1)
   */
  constructor(model: SHAPModel, data: number[][], seed: number | null) {
    this.model = model;
    this.data = data;

    // Initialize the RNG
    if (seed) {
      let curSeed = seed;
      if (seed < 0 || seed > 1) {
        console.warn('Clipping random seed to range [0, 1)');
        curSeed = Math.abs(curSeed);
        curSeed = curSeed - Math.floor(curSeed);
      }

      this.rng = randomUniform.source(randomLcg(curSeed));
    } else {
      this.rng = randomUniform.source(randomLcg(0.20230101));
    }

    // Initialize the model values
    // Step 1: Compute the base value (expected values), which is the average
    // of the predictions on the background dataset
    this.predictions = this.model(this.data);
    this.expectedValue =
      this.predictions.reduce((a, b) => a + b) / this.predictions.length;

    // Step 2: Initialize data structures
    this.nFeatures = this.data[0].length;
    this.nTargets = 1;
    this.nSamplesAdded = 0;
  }
}
