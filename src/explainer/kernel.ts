/**
 * Kernel SHAP
 * @author: Jay Wang (jay@zijie.wang)
 */

import { randomLcg, randomUniform, randomInt } from 'd3-random';
import type { RandomUniform, RandomInt } from 'd3-random';
import type { SHAPModel } from '../my-types';
import { comb, getCombinations } from '../utils/utils';
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

  /**
   * Sampled data in a matrix form.
   * It is initialized after the explain() call.
   */
  sampledData: math.Matrix | null = null;

  /**
   * Matrix to store the feature masks
   * [nSamples, this.nFeatures]
   */
  maskMat: math.Matrix | null = null;

  /**
   * Kernel weights for each coalition sample
   * [nSamples, 1]
   */
  kernelWeight: math.Matrix | null = null;

  /**
   * Model prediction outputs on the sampled data
   * [nSamples * nBackground, this.nTargets]
   */
  yMat: math.Matrix | null = null;

  /**
   * Expected model predictions on the sample data
   * [nSamples, this.nTargets]
   */
  yExpMat: math.Matrix | null = null;

  /**
   * Mask used in the last run
   * [nSamples]
   */
  lastMask: math.Matrix | null = null;

  /** Random seed */
  lcg: () => number;

  /** Uniform random number generator*/
  rng: RandomUniform;

  /** Uniform random integer generator */
  rngInt: RandomInt;

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
      this.lcg = randomLcg(curSeed);
    } else {
      this.lcg = randomLcg(0.20230101);
    }
    this.rng = randomUniform.source(this.lcg);
    this.rngInt = randomInt.source(this.lcg);

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
    const curX = [x.slice()];

    // Find the current prediction f(x)
    // Return a matrix with only one item (y(x))
    const yPredProbMat = math.matrix(this.model(curX));

    // Sample feature coalitions
    const fractionEvaluated = this.sampleFeatureCoalitions(x, nSamples);

    // Inference on the sampled feature coli
    this.inferenceFeatureCoalitions();
  };

  inferenceFeatureCoalitions = () => {
    if (this.sampledData === null) {
      console.error('sampledData is null.');
      return;
    }

    if (this.yExpMat === null) {
      console.error('yExpMat is null.');
      return;
    }

    if (this.yMat === null) {
      console.error('yMat is null.');
      return;
    }

    // Convert the sampled data from matrix to a 2D vec
    const sampledDataVec = this.sampledData.toArray() as number[][];

    // Get the model output on the sampled data and initialize self.y_mat
    const yPredProb = this.model(sampledDataVec);
    this.yMat.subset(
      math.index(math.range(0, this.yMat.size()[0]), 0),
      yPredProb
    );

    // Get the mean y value of samples having the same mask
    const nBackground = this.data.length;
    for (let i = 0; i < this.nSamplesAdded; i++) {
      const yMatSlice = this.yMat.subset(
        math.index(math.range(i * nBackground, (i + 1) * nBackground), 0)
      );
      this.yExpMat.subset(math.index(i, 0), math.mean(yMatSlice));
    }
  };

  /**
   * Enumerate/sample feature coalitions to approximate the shapley values
   * @param x Instance to explain
   * @param nSamples Number of coalitions to sample
   * @returns Sample rate (fraction of sampled feature coalitions)
   */
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

    if (this.kernelWeight === null) {
      console.error('kernelWeight is not initialized.');
      return;
    }

    // Search for feature coalitions to sample and give them SHAP kernel
    // weights: (M - 1) / (C(M, z) * z * (M - z)).

    // Sampling i features has the same weight as sampling (M - i) features
    // Here we sample feature coalitions and their complement at the same time
    const maxSampleSize = Math.ceil((this.nFeatures - 1) / 2);
    const maxPairedSampleSize = Math.floor((this.nFeatures - 1) / 2);

    // Initialize the weight vector with (M - 1) / (z * (M - z))
    const sampleWeights = new Array<number>(maxSampleSize).fill(0);
    for (let i = 1; i < maxSampleSize + 1; i++) {
      sampleWeights[i - 1] = (this.nFeatures - 1) / (i * (this.nFeatures - i));
    }

    // Normalize the weights so that they sum up to 1. Because the weights
    // for i and (M - i) are stored at the same index, we times 2 for paired
    // indexes before normalization
    for (let i = 1; i < maxPairedSampleSize + 1; i++) {
      sampleWeights[i - 1] *= 2;
    }
    const weightSum = sampleWeights.reduce((a, b) => a + b);
    for (let i = 1; i < maxSampleSize + 1; i++) {
      sampleWeights[i - 1] /= weightSum;
    }

    // Sample feature coalitions by iterating the sample size from two tails
    // (a lot of 1 or a lot of 0 in the mask array) to the middle
    // Track the number of sample size we use full samples
    let nFullSubsets = 0;
    let nSamplesLeft = curNSamples;
    let remainSampleWeights = sampleWeights.slice();

    for (let curSize = 1; curSize <= maxSampleSize; curSize++) {
      // Compute the number of samples with the current sample size
      let nSubsets = comb(this.nFeatures, curSize);

      // We sample from two tails if possible
      if (curSize <= maxPairedSampleSize) {
        nSubsets *= 2;
      }

      // If we have enough budget left to sample all coalitions with the
      // current subset size
      if (nSubsets < nSamplesLeft * remainSampleWeights[curSize - 1]) {
        nFullSubsets += 1;
        nSamplesLeft -= nSubsets;

        // Rescale the remaining weights to sum to 1
        if (remainSampleWeights[curSize - 1] < 1.0) {
          const scale = 1.0 - remainSampleWeights[curSize - 1];

          for (let i = 0; i < remainSampleWeights.length; i++) {
            remainSampleWeights[i] /= scale;
          }
        }

        // Add all coalitions with the current subset size
        let curWeight =
          sampleWeights[curSize - 1] / comb(this.nFeatures, curSize);

        // If there is complement pair, split the weight
        if (curSize <= maxPairedSampleSize) {
          curWeight /= 2.0;
        }

        // Add combinations into sampledData
        const rangeArray = Array.from(new Array(this.nFeatures), (_, i) => i);
        const combinations = getCombinations(rangeArray, curSize);

        for (const activeIndexes of combinations) {
          const mask = new Array<number>(this.nFeatures).fill(0.0);
          for (const i of activeIndexes) {
            mask[i] = 1.0;
          }
          this.addSample(x, mask, curWeight);

          // Add the complements combination if it is paired
          if (curSize <= maxPairedSampleSize) {
            const compMask = mask.map(x => (x === 0.0 ? 1.0 : 0.0));
            this.addSample(x, compMask, curWeight);
          }
        }
      } else {
        break;
      }
    }

    // Now there is no budge left to sample all combinations for the current
    // sample size. We randomly sample combinations until use up all budgets.
    const nFixedSamples = this.nSamplesAdded;
    nSamplesLeft = curNSamples - nFixedSamples;

    if (nFullSubsets !== maxSampleSize) {
      // Reinitialize the running weights from the initial weights
      remainSampleWeights = sampleWeights.slice();

      // If it has complementary sampling, we sample two combinations in
      // each iteration
      for (let i = 0; i < maxPairedSampleSize; i++) {
        remainSampleWeights[i] /= 2.0;
      }

      // Make the remaining weights sum to 1
      remainSampleWeights = remainSampleWeights.slice(nFullSubsets);
      const weightSum = remainSampleWeights.reduce((a, b) => a + b);
      for (let i = 0; i < remainSampleWeights.length; i++) {
        remainSampleWeights[i] /= weightSum;
      }

      // Randomly choose sample subset's size (*10 is arbitrary, we won't
      // iterate all of them.)
      // We use weighted uniform random
      const randomSubsetSizes: number[] = [];
      let randomSubsetSizesCursor = 0;
      const cdf = remainSampleWeights.map(
        (
          sum => value =>
            (sum += value)
        )(0)
      );

      for (let i = 0; i < 10 * nSamplesLeft; i++) {
        const curRandomNum = this.rng(0, 1)();
        // We can safely use length here because random's max is exclusive
        // (smaller than 1)
        const curSelectedIndex = cdf.filter(d => curRandomNum >= d).length;
        randomSubsetSizes.push(curSelectedIndex);
      }

      // Track the mask combinations we have used
      const usedMasks = new Map<string, number>();

      while (
        nSamplesLeft > 0 &&
        randomSubsetSizesCursor < randomSubsetSizes.length
      ) {
        // Gte a random sample subset size
        const curSize =
          randomSubsetSizes[randomSubsetSizesCursor] + nFullSubsets + 1;
        randomSubsetSizesCursor += 1;

        // Generate the current mask
        const mask = new Array<number>(this.nFeatures).fill(0);

        // Randomly sample curSize indexes
        const activeIndexes: number[] = [];
        const sampledIndexes = new Set<number>();

        while (activeIndexes.length < curSize) {
          const curRandomIndex = this.rngInt(this.nFeatures)();
          if (!sampledIndexes.has(curRandomIndex)) {
            sampledIndexes.add(curRandomIndex);
            activeIndexes.push(curRandomIndex);
          }
        }

        for (const i of activeIndexes) mask[i] = 1;

        // Add this sample if we have not used this mask yet, otherwise
        // we just increase the previous occurrence's weight
        const maskStr = KernelSHAP.getMaskStr(mask);
        if (usedMasks.has(maskStr)) {
          // If this mask has been used, update its weight
          const weightI = usedMasks.get(maskStr)!;
          this.kernelWeight.subset(
            math.index(weightI, 0),
            (this.kernelWeight.get([weightI, 0]) as number) + 1
          );
        } else {
          // Add a new sample
          usedMasks.set(maskStr, this.nSamplesAdded);
          nSamplesLeft -= 1;

          // The weight here is 1.0 because we used `remain_sample_weights`
          // to sample the subset sizes
          // https://github.com/slundberg/shap/issues/2615
          this.addSample(x, mask, 1.0);
        }

        // Also handle this mask's complementary mask
        if (nSamplesLeft > 0 && curSize <= maxPairedSampleSize) {
          const compMask = mask.map(x => (x === 0 ? 1 : 0));
          const compMaskStr = KernelSHAP.getMaskStr(compMask);

          if (usedMasks.has(compMaskStr)) {
            // If this mask has been used, update its weight
            const weightI = usedMasks.get(compMaskStr)!;
            this.kernelWeight.subset(
              math.index(weightI, 0),
              (this.kernelWeight.get([weightI, 0]) as number) + 1
            );
          } else {
            // Add a new sample
            usedMasks.set(compMaskStr, this.nSamplesAdded);
            nSamplesLeft -= 1;
            this.addSample(x, mask, 1.0);
          }
        }
      }

      // Override the kernel weights for random samples and make sure all
      // weights sum up to one
      const leftWeightSum = sampleWeights
        .slice(nFullSubsets)
        .reduce((a, b) => a + b);

      const curRandWeightSum = math.sum(
        this.kernelWeight.subset(
          math.index(math.range(nFixedSamples, this.kernelWeight.size()[0]), 0)
        )
      ) as number;
      const weightScale = leftWeightSum / curRandWeightSum;

      for (let i = nFixedSamples; i < this.kernelWeight.size()[0]; i++) {
        this.kernelWeight.subset(
          math.index(i, 0),
          this.kernelWeight.get([i, 0]) * weightScale
        );
      }
    }

    // Return the sample rate
    return curNSamples / nSamplesMax;
  };

  // Add a feature coalition sample into `self.sampled_data`
  addSample(x: number[], mask: number[], weight: number) {
    if (this.sampledData === null) {
      console.error('this.sampleData is null');
      return;
    }

    if (this.maskMat === null) {
      console.error('this.maskMat is null');
      return;
    }

    if (this.kernelWeight === null) {
      console.error('this.kernelWeight is null');
      return;
    }

    // (1) Find the current block in self.sampled_data to modify
    const backgroundDataLength = this.data.length;
    const rStart = this.nSamplesAdded * backgroundDataLength;
    const rEnd = rStart + backgroundDataLength;

    // (2) Fill columns with mask=1 to be corresponding value from the
    // explaining instance x
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] === 1) {
        const newColumn = new Array(backgroundDataLength).fill(x[i]);
        this.sampledData.subset(
          math.index(math.range(rStart, rEnd), i),
          newColumn
        );
      }
    }

    // (3) Update tracker variables
    // Record the mask in self.mask_mat
    this.maskMat.subset(
      math.index(this.nSamplesAdded, math.range(0, this.maskMat.size()[1])),
      mask
    );

    // Record the weight in self.kernel_weight
    this.kernelWeight.subset(math.index(this.nSamplesAdded, 0), weight);

    this.nSamplesAdded += 1;
  }

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
      const row = i * nBackground;
      this.sampledData.subset(
        math.index(
          math.range(row, row + nBackground),
          math.range(0, backgroundMat.size()[1])
        ),
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

  /**
   * Helper function to convert a mask array into a string
   * @param mask Binary mask array
   * @returns String version of the binary mask array
   */
  static getMaskStr = (mask: number[]) => {
    return mask.map(x => (x === 1.0 ? '1' : '0')).join('');
  };
}
