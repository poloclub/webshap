/**
 * Kernel SHAP
 * @author: Jay Wang (jay@zijie.wang)
 */

import { randomLcg, randomUniform } from 'd3-random';
import type { RandomUniform } from 'd3-random';
import type { SHAPModel } from '../my-types';
import math from '../utils/math-import';

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
  sampledData: math.Matrix | null = null;

  /** Matrix to store the feature masks */
  maskMat: math.Matrix | null = null;

  /** Kernel weights for each coalition sample */
  kernelWeight: math.Matrix | null = null;

  /** Model prediction outputs on the sampled data */
  yMat: math.Matrix | null = null;

  /** Expected model predictions on the sample data */
  yExpMat: math.Matrix | null = null;

  /** Mask used in the last run */
  lastMask: math.Matrix | null = null;

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

  /**
   * Estimate SHAP values of the given sample x
   * @param x One data sample
   * @param nSamples Number of coalitions to samples (default to null which uses
   * a heuristic to determine a large sample size)
   */
  explainOneInstance = (x: number[], nSamples: number | null = null) => {
    // Validate the input
    if (x.length !== this.nFeatures) {
      throw new Error(
        'x has to have the same number of features as the background dataset.'
      );
    }

    // Create a copy of the given 1D x array in a 2D format
    const curX = structuredClone(x);

    // Find the current prediction f(x)
    // Return a matrix with only one item (y(x))
    const yPredProbMat = this.model([x]);

    // Generate sampled data
    const fractionEvaluated = this.sampleFeatureCoalitions(x, nSamples);
  };

  sampleFeatureCoalitions = (x: number[], nSamples: number | null) => {
    // Determine the number of feature coalitions to sample
    // If `n_samples` is not given, we use a simple heuristic to
    // determine number of samples to train the linear model
    // https://github.com/slundberg/shap/issues/97
    let curNSamples = nSamples ? nSamples : this.nFeatures * 2 + 2048;
    let nSamplesMax = Math.pow(2, 30);

    // If there are not too many features, we can enumerate all coalitions
    if (this.nFeatures <= 30) {
      // We subtract 2 here to discount the cases with all 1 and all 0,
      // which are not helpful to figure out feature attributions
      nSamplesMax = Math.pow(2, this.nFeatures) - 2;
      if (curNSamples > nSamplesMax) curNSamples = nSamplesMax;
    }

    // Prepare for the feature coalition sampling
    this.prepareSampling(curNSamples);
  };

  /**
   * Initialize data structures to prepare for the feature coalition sampling
   * @param nSamples Number of coalitions to sample
   */
  prepareSampling = (nSamples: number) => {
    // Store the sampled data
    // (number of background samples * n_samples, n_features)
    const nBackground = this.data.length;
    this.sampledData = math.matrix(
      math.zeros([nBackground * nSamples, this.nFeatures])
    );

    // Convert the background data from 2d vector to DMatrix
    const backgroundMat = math.matrix(this.data);

    // Initialize the sampled data by repeating the background samples
    for (let i = 0; i < nSamples; i++) {
      const rEnd = i * nBackground;
      this.sampledData.subset(
        math.index(math.range(i, rEnd), math.range(0, backgroundMat.size()[1])),
        backgroundMat
      );
    }

    // Initialize the mask matrix
    this.maskMat = math.matrix(math.zeros([nSamples, this.nFeatures]));

    // Initialize the kernel weight matrix
    this.kernelWeight = math.matrix(math.zeros([nSamples, 1]));

    // Matrix to store the model outputs and expected outputs
    this.yMat = math.matrix(
      math.zeros([nSamples * nBackground, this.nTargets])
    );
    this.yExpMat = math.matrix(math.zeros([nSamples, this.nTargets]));
    this.lastMask = math.matrix(math.zeros([nSamples]));
  };
}
