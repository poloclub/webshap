import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import { tick } from 'svelte';
import type {
  TabularData,
  TabularContFeature,
  TabularCatFeature,
  Size,
  Padding
} from '../../types/common-types';
import { KernelSHAP } from 'webshap';
import { round, timeit } from '../../utils/utils';

import * as ort from 'onnxruntime-web/dist/ort-web.min.js';
import wasm from 'onnxruntime-web/dist/ort-wasm.wasm?url';
import wasmThreaded from 'onnxruntime-web/dist/ort-wasm-threaded.wasm?url';
import wasmSimd from 'onnxruntime-web/dist/ort-wasm-simd.wasm?url';
import wasmSimdThreaded from 'onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';
import modelUrl from './../../../../models/lending-club-xgboost.onnx?url';

// Set up the correct WASM paths
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': wasm,
  'ort-wasm-threaded.wasm': wasmThreaded,
  'ort-wasm-simd.wasm': wasmSimd,
  'ort-wasm-simd-threaded.wasm': wasmSimdThreaded
};

const DEBUG = config.debug;
const LCG = d3.randomLcg(0.20230101);
const RANDOM_INT = d3.randomInt.source(LCG);
const RANDOM_UNIFORM = d3.randomUniform.source(LCG);

/**
 * Class for the Tabular WebSHAP demo
 */

export class Tabular {
  component: HTMLElement;
  tabularUpdated: () => void;

  // SVGs
  predBarSVG: d3.Selection<HTMLElement, unknown, null, undefined>;
  predBarSVGSize: Size;
  predBarSVGPadding: Padding;
  predBarScale: d3.ScaleLinear<number, number, never>;

  // Dataset
  data: TabularData | null = null;
  contFeatures: Map<string, TabularContFeature> | null = null;
  catFeatures: Map<string, TabularCatFeature> | null = null;
  curX: number[] | null = null;
  curY: number | null = null;
  curIndex = 0;

  // ONNX data
  message: string;
  curPred: number | null = null;

  // WebSHAP data
  backgroundData: number[][] = [];

  /**
   * @param args Named parameters
   * @param args.component The component
   * @param args.tabularUpdated A function to trigger updates
   */
  constructor({
    component,
    tabularUpdated
  }: {
    component: HTMLElement;
    tabularUpdated: () => void;
  }) {
    this.component = component;
    this.tabularUpdated = tabularUpdated;
    this.message = 'initialized';

    this.predBarSVG = d3
      .select<HTMLElement, unknown>(this.component)
      .select('svg.pred-bar-svg');
    this.predBarSVGSize = { width: 0, height: 0 };
    this.predBarSVGPadding = { top: 4, bottom: 4, left: 10, right: 10 };
    this.predBarScale = d3.scaleLinear();

    // Load the training and test dataset
    this.initData().then(() => {
      // Initialize the SVGs
      tick().then(() => {
        this.initPredBar();
      });
    });
  }

  initPredBar = () => {
    if (this.predBarSVG === null) throw Error('predBarSVG is null.');

    // Get the SVG size
    const svgBBox = this.predBarSVG.node()?.getBoundingClientRect();
    if (svgBBox !== undefined) {
      this.predBarSVGSize.width =
        svgBBox.width -
        this.predBarSVGPadding.left -
        this.predBarSVGPadding.right;
      this.predBarSVGSize.height =
        svgBBox.height -
        this.predBarSVGPadding.top -
        this.predBarSVGPadding.bottom;
    }

    const content = this.predBarSVG
      .append('g')
      .attr('class', 'content')
      .attr(
        'transform',
        `translate(${this.predBarSVGPadding.left}, ${this.predBarSVGPadding.top})`
      );

    // Create scales
    this.predBarScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([0, this.predBarSVGSize.width]);

    // Init rectangles
    content
      .append('rect')
      .attr('class', 'back-rect')
      .attr('rx', this.predBarSVGSize.height / 2)
      .attr('ry', this.predBarSVGSize.height / 2)
      .attr('width', this.predBarScale(1))
      .attr('height', this.predBarSVGSize.height);

    content
      .append('rect')
      .attr('class', 'top-rect')
      .classed('approval', this.curPred ? this.curPred >= 0.5 : true)
      .attr('rx', this.predBarSVGSize.height / 2)
      .attr('ry', this.predBarSVGSize.height / 2)
      .attr('width', this.predBarScale(this.curPred || 0))
      .attr('height', this.predBarSVGSize.height);

    // Add a threshold bar
    content
      .append('rect')
      .attr('class', 'threshold')
      .attr('x', this.predBarScale(0.5) - 1)
      .attr('width', 2)
      .attr('height', this.predBarSVGSize.height);
  };

  /**
   * Load the lending club dataset.
   */
  initData = async () => {
    this.data = (await d3.json(
      `${import.meta.env.BASE_URL}data/lending-club.json`
    )) as TabularData;

    // Load a random sample
    this.loadRandomSample();

    // Inference the model
    const x = this.getCurX();
    const result = await this.predict([x]);
    this.curPred = result[0];

    // Explain this instance
    // Create background data for SHAP
    this.backgroundData = [];

    // Take 10 random training data
    // const backgroundSize = 1;
    // const addedIndexes = new Set<number>();
    // while (this.backgroundData.length < backgroundSize) {
    //   const curRandomIndex = RANDOM_INT(this.data.xTrain.length)();
    //   if (!addedIndexes.has(curRandomIndex)) {
    //     this.backgroundData.push(this.data.xTrain[curRandomIndex]);
    //     addedIndexes.add(curRandomIndex);
    //   }
    // }

    // Take training data median as background data
    const curBackgroundData = [];
    for (let c = 0; c < this.data.xTrain[0].length; c++) {
      const curColumn: number[] = [];
      for (let r = 0; r < this.data.xTrain.length; r++) {
        curColumn.push(this.data.xTrain[r][c]);
      }
      const curMedian = d3.median(curColumn) || 0;
      curBackgroundData.push(curMedian);
    }
    this.backgroundData.push(curBackgroundData);

    const shapValues = await this.explain(x);
    // console.log('background', this.backgroundData);
    // console.log('shap', shapValues);
  };

  /**
   * Load a random sample from the test dataset.
   */
  loadRandomSample = () => {
    if (this.data === null) {
      throw Error('this.data is null');
    }

    // Get a random instance
    // RANDOM_INT is seeded, but d3.randomInt is not
    // const randomIndex = d3.randomInt(this.data.xTest.length)();
    const randomIndex = RANDOM_INT(this.data.xTest.length)();

    this.curX = this.data.xTest[randomIndex];
    this.curY = this.data.yTest[randomIndex];
    this.curIndex = randomIndex;

    // Convert the data into structured format
    this.contFeatures = new Map();
    this.catFeatures = new Map();
    const addedCatNames = new Set<string>();

    for (const [i, featureType] of this.data.featureTypes.entries()) {
      if (featureType === 'cont') {
        const curName = this.data.featureNames[i];
        this.contFeatures.set(curName, {
          name: curName,
          displayName: this.data.featureInfo[curName][0],
          desc: this.data.featureInfo[curName][1],
          value: this.data.featureRequiresLog.includes(curName)
            ? round(Math.pow(10, this.curX[i]), 0)
            : this.curX[i],
          requiresInt: this.data.featureRequireInt.includes(curName),
          requiresLog: this.data.featureRequiresLog.includes(curName)
        });
      } else {
        const curName = this.data.featureNames[i].replace(/(.+)-(.+)/, '$1');
        const curLevel = this.data.featureNames[i].replace(/(.+)-(.+)/, '$2');

        // Initialize this entry
        if (!addedCatNames.has(curName)) {
          const curLevelInfo = this.data.featureLevelInfo[curName];
          const allLevels = [];
          for (const key of Object.keys(curLevelInfo)) {
            allLevels.push({
              level: key,
              displayName: curLevelInfo[key][0]
            });
          }

          this.catFeatures.set(curName, {
            name: curName,
            displayName: this.data.featureInfo[curName][0],
            desc: this.data.featureInfo[curName][1],
            levelInfo: this.data.featureLevelInfo[curName],
            allLevels: allLevels,
            value: '0'
          });
          addedCatNames.add(curName);
        }

        // Handle one-hot encoding
        if (this.curX[i] == 1) {
          this.catFeatures.get(curName)!.value = curLevel;
        }
      }
    }

    this.tabularUpdated();
  };

  /**
   * Get the current x values from the user inputs
   */
  getCurX = () => {
    if (
      this.data === null ||
      this.catFeatures === null ||
      this.contFeatures === null ||
      this.curX === null ||
      this.curY === null
    ) {
      throw Error('Data or a random sample is not initialized');
    }

    const curX = new Array<number>(this.curX.length).fill(0);

    // Iterate through all features to get the current x values
    for (const [i, featureType] of this.data.featureTypes.entries()) {
      if (featureType === 'cont') {
        const curName = this.data.featureNames[i];
        const curContFeature = this.contFeatures.get(curName)!;
        if (curContFeature.requiresLog) {
          curX[i] = Math.log10(curContFeature.value);
        } else {
          curX[i] = curContFeature.value;
        }
      } else {
        const curName = this.data.featureNames[i].replace(/(.+)-(.+)/, '$1');
        const curLevel = this.data.featureNames[i].replace(/(.+)-(.+)/, '$2');

        // One-hot encoding: we only need to change the value of the active
        // one-hot column
        const curCatFeature = this.catFeatures.get(curName)!;
        if (curCatFeature.value === curLevel) {
          curX[i] = 1;
        }
      }
    }

    return curX;
  };

  /**
   * Run XGBoost on the given input data x
   * @param x Input data instances (n, k)
   * @returns Predicted positive label probabilities (n)
   */
  predict = async (x: number[][]) => {
    const posProbs = [];

    try {
      // Create a new session and load the LightGBM model
      const session = await ort.InferenceSession.create(modelUrl);

      // First need to flatten the x array
      const xFlat = Float32Array.from(x.flat());

      // Prepare feeds, use model input names as keys.
      const xTensor = new ort.Tensor('float32', xFlat, [x.length, x[0].length]);
      const feeds = { float_input: xTensor };

      // Feed inputs and run
      const results = await session.run(feeds);

      // Read from results, probs has shape (n * 2) => (n, 2)
      const probs = results.probabilities.data as Float32Array;

      for (const [i, p] of probs.entries()) {
        // Positive label prob is always at the odd index
        if (i % 2 === 1) {
          posProbs.push(p);
        }
      }

      this.message = `Success: ${round(posProbs[0], 4)}`;
    } catch (e) {
      this.message = `Failed: ${e}.`;
    }

    this.tabularUpdated();
    return posProbs;
  };

  /**
   * Run WebSHAP to explain the given input data x
   * @param x Input data instance
   */
  explain = async (x: number[]) => {
    const explainer = new KernelSHAP(
      (x: number[][]) => this.predict(x),
      this.backgroundData,
      0.2022
    );

    timeit('Explain', DEBUG);
    const shapValues = await explainer.explainOneInstance(x, 32);
    // const shapValues = await explainer.explainOneInstance(x);
    timeit('Explain', DEBUG);
    return shapValues;
  };

  /**
   * Event handler for the sample button clicking.
   */
  sampleClicked = async () => {
    this.loadRandomSample();
    const curX = this.getCurX();
    const result = await this.predict([curX]);
    this.curPred = result[0];
    this.updatePred();
    this.tabularUpdated();
  };

  /**
   * Event handler for the sample button clicking.
   */
  inputChanged = async () => {
    const curX = this.getCurX();
    const result = await this.predict([curX]);
    this.curPred = result[0];
    this.updatePred();
    this.tabularUpdated();
  };

  /**
   * Helper function to update the view with the new prediction result
   */
  updatePred = () => {
    if (this.curPred === null) return;

    // Update the bar
    const content = this.predBarSVG.select('g.content');
    content
      .select('rect.top-rect')
      .classed('approval', this.curPred >= 0.5)
      .attr('width', this.predBarScale(this.curPred));
  };
}
