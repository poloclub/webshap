import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import { tick } from 'svelte';
import type {
  TabularData,
  TabularContFeature,
  TabularCatFeature,
  Size,
  Padding,
  SHAPRow
} from '../../types/common-types';
import { KernelSHAP } from 'webshap';
import { round, timeit } from '../../utils/utils';
import { getLatoTextWidth } from '../../utils/text-width';

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

// SVG constants
const GAP = 20;
const K = 10;
const ROW_HEIGHT = 28;
const FORMAT_2 = d3.format('.4f');
const BAR_HEIGHT = ROW_HEIGHT - 8;

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

  shapSVG: d3.Selection<HTMLElement, unknown, null, undefined>;
  shapSVGSize: Size;
  shapSVGPadding: Padding;
  shapScale: d3.ScaleLinear<number, number, never>;
  maxTextWidth = 200;
  maxBarWidth = 200;

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
  curShapValues: number[] | null = null;

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

    this.shapSVG = d3
      .select<HTMLElement, unknown>(this.component)
      .select('svg.shap-svg');
    this.shapSVGSize = { width: 0, height: 0 };
    this.shapSVGPadding = { top: 1, bottom: 1, left: 10, right: 25 };
    this.shapScale = d3.scaleLinear();

    // Load the training and test dataset
    this.initData().then(() => {
      // Initialize the SVGs
      tick().then(() => {
        this.initPredBar();
        this.initShapPlot();
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

  initShapPlot = () => {
    if (this.shapSVG === null) throw Error('shapSVG is null.');
    if (this.curShapValues === null) throw Error('curShapValues is null.');
    if (this.data === null) throw Error('data is null.');
    if (this.contFeatures === null) throw Error('contFeatures is null.');
    if (this.catFeatures === null) throw Error('catFeatures is null.');
    if (this.curX === null) throw Error('curX is null.');

    // Get the SVG size
    const svgBBox = this.shapSVG.node()?.getBoundingClientRect();
    if (svgBBox !== undefined) {
      this.shapSVGSize.width =
        svgBBox.width - this.shapSVGPadding.left - this.shapSVGPadding.right;
      this.shapSVGSize.height =
        svgBBox.height - this.shapSVGPadding.top - this.shapSVGPadding.bottom;
    }

    const content = this.shapSVG
      .append('g')
      .attr('class', 'content')
      .attr(
        'transform',
        `translate(${this.shapSVGPadding.left}, ${this.shapSVGPadding.top})`
      );

    // Decide the text and bar widths
    let maxTextWidth = 200;
    let maxBarWidth = 200;

    if (this.shapSVGSize.width - 260 - GAP > 200) {
      maxTextWidth = 260;
      maxBarWidth = this.shapSVGSize.width - GAP - maxTextWidth;
    } else {
      maxBarWidth = 220;
      maxTextWidth = this.shapSVGSize.width - GAP - maxBarWidth;
    }

    this.maxTextWidth = maxTextWidth;
    this.maxBarWidth = maxBarWidth;

    // Create scales
    const absValues = this.curShapValues.map(x => Math.abs(x));
    const maxAbs = d3.max(absValues)!;
    this.shapScale = d3
      .scaleLinear()
      .domain([0, maxAbs])
      .range([0, maxBarWidth / 2]);

    const shapValueScale = d3
      .scaleLinear()
      .domain([-maxAbs, maxAbs])
      .range([0, maxBarWidth]);

    // Organize all shap values
    const allShaps: SHAPRow[] = [];

    for (let i = 0; i < this.data.featureNames.length; i++) {
      const curFeatureType = this.data.featureTypes[i];
      const curFeatureName = this.data.featureNames[i];

      // Get the display name
      let displayName = '';
      let fullName = '';

      if (curFeatureType === 'cont') {
        displayName = this.contFeatures.get(curFeatureName)!.displayName;
        fullName = displayName;
      } else {
        const curName = curFeatureName.replace(/(.+)-(.+)/, '$1');
        const curLevel = curFeatureName.replace(/(.+)-(.+)/, '$2');
        const catInfo = this.catFeatures.get(curName)!;
        const dummy = this.curX[i] === 1 ? 'T' : 'F';
        const dummyFull = this.curX[i] === 1 ? 'True' : 'False';
        displayName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummy})`;
        fullName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummyFull})`;
      }

      // Truncate displayName until it fits the limit
      let nameWidth = getLatoTextWidth(displayName, 15);

      while (nameWidth > maxTextWidth) {
        displayName = displayName.replace('...', '');
        displayName = displayName
          .slice(0, displayName.length - 1)
          .concat('...');
        nameWidth = getLatoTextWidth(displayName, 15);
      }

      allShaps.push({
        index: i,
        shap: this.curShapValues[i],
        name: displayName,
        fullName: fullName
      });
    }

    // Sort all shaps based on their absolute shap values
    allShaps.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

    const rowContent = content
      .append('g')
      .attr('class', 'row-content')
      .attr('transform', 'translate(0, 20)');

    // Draw the background grid
    rowContent
      .append('rect')
      .attr('class', 'grid-rect')
      .attr('x', maxTextWidth + GAP + maxBarWidth / 2)
      .attr('y', -BAR_HEIGHT / 2)
      .attr('width', 0.2)
      .attr('height', 10 * ROW_HEIGHT + 5);

    /**
     * Helper function to add a SHAP row
     * @param shap SHAP value
     * @param y Y of this row
     * @param opacity The initial opacity value
     */
    const addShapRow = (shap: SHAPRow, y: number, opacity: number) => {
      const row = rowContent
        .append('g')
        .attr('class', `row row-${shap.index}`)
        .attr('transform', `translate(0, ${y})`)
        .style('opacity', opacity);

      // Add background grid
      row
        .append('line')
        .attr('class', 'grid-line')
        .attr('x1', maxTextWidth + GAP / 2)
        .attr('y1', 0)
        .attr('x2', maxTextWidth + GAP + maxBarWidth + GAP / 2)
        .attr('y2', 0);

      row
        .append('text')
        .attr('class', 'feature-name')
        .attr('x', maxTextWidth)
        .text(shap.name)
        .append('title')
        .text(shap.fullName);

      // Add the rectangle
      const rect = row.append('rect').attr('class', 'shap-bar');
      const curRectWidth = this.shapScale(Math.abs(shap.shap));
      if (shap.shap < 0) {
        rect
          .classed('negative', true)
          .attr('x', maxTextWidth + GAP + maxBarWidth / 2 - curRectWidth)
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth)
          .attr('height', BAR_HEIGHT);
      } else {
        rect
          .attr('x', maxTextWidth + GAP + maxBarWidth / 2)
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth)
          .attr('height', BAR_HEIGHT);
      }

      // Add the shap number
      row
        .append('text')
        .attr('class', 'shap-number')
        .classed('negative', shap.shap < 0)
        .text(FORMAT_2(shap.shap))
        .attr(
          'x',
          shap.shap < 0
            ? maxTextWidth + GAP + maxBarWidth / 2 + 5
            : maxTextWidth + GAP + maxBarWidth / 2 - 5
        );
    };

    // Add the top K in a list
    for (let i = 0; i < K; i++) {
      const shap = allShaps[i];
      addShapRow(shap, i * ROW_HEIGHT, 1);
    }

    // Draw the rest shap values off the screen
    for (let i = K; i < this.data.featureNames.length; i++) {
      const shap = allShaps[i];
      addShapRow(shap, this.shapSVGSize.height + 5, 0);
    }

    // Draw the axis
    const axisGroup = content
      .append('g')
      .attr('class', 'axis-group')
      .attr(
        'transform',
        `translate(${maxTextWidth + GAP}, ${this.shapSVGSize.height - 20})`
      );
    const axis = d3.axisBottom(shapValueScale).tickValues([-maxAbs, 0, maxAbs]);
    axisGroup.call(axis);
  };

  updateShapPlot = () => {
    if (this.shapSVG === null) throw Error('shapSVG is null.');
    if (this.curShapValues === null) throw Error('curShapValues is null.');
    if (this.data === null) throw Error('data is null.');
    if (this.contFeatures === null) throw Error('contFeatures is null.');
    if (this.catFeatures === null) throw Error('catFeatures is null.');

    const curX = this.getCurX();

    // Create scales
    const absValues = this.curShapValues.map(x => Math.abs(x));
    const maxAbs = d3.max(absValues)!;
    this.shapScale = d3
      .scaleLinear()
      .domain([0, maxAbs])
      .range([0, this.maxBarWidth / 2]);

    const shapValueScale = d3
      .scaleLinear()
      .domain([-maxAbs, maxAbs])
      .range([0, this.maxBarWidth]);

    // Organize all shap values
    const allShaps: SHAPRow[] = [];

    for (let i = 0; i < this.data.featureNames.length; i++) {
      const curFeatureType = this.data.featureTypes[i];
      const curFeatureName = this.data.featureNames[i];

      // Get the display name
      let displayName = '';
      let fullName = '';

      if (curFeatureType === 'cont') {
        displayName = this.contFeatures.get(curFeatureName)!.displayName;
        fullName = displayName;
      } else {
        const curName = curFeatureName.replace(/(.+)-(.+)/, '$1');
        const curLevel = curFeatureName.replace(/(.+)-(.+)/, '$2');
        const catInfo = this.catFeatures.get(curName)!;
        const dummy = curX[i] === 1 ? 'T' : 'F';
        const dummyFull = curX[i] === 1 ? 'True' : 'False';
        displayName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummy})`;
        fullName = `${catInfo.displayName} (${catInfo.levelInfo[curLevel][0]}=${dummyFull})`;
      }

      // Truncate displayName until it fits the limit
      let nameWidth = getLatoTextWidth(displayName, 15);

      while (nameWidth > this.maxTextWidth) {
        displayName = displayName.replace('...', '');
        displayName = displayName
          .slice(0, displayName.length - 1)
          .concat('...');
        nameWidth = getLatoTextWidth(displayName, 15);
      }

      allShaps.push({
        index: i,
        shap: this.curShapValues[i],
        name: displayName,
        fullName: fullName
      });
    }

    // Sort all shaps based on their absolute shap values
    allShaps.sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

    const content = this.shapSVG.select('g.content');
    const rowContent = this.shapSVG.select('g.row-content');

    const trans = d3.transition('update').duration(300).ease(d3.easeCubicInOut);

    const updateShapRow = (shap: SHAPRow, y: number, opacity: number) => {
      const row = rowContent.select(`g.row-${shap.index}`);

      row
        .transition(trans)
        .attr('transform', `translate(0, ${y})`)
        .style('opacity', opacity);

      // Update the feature name
      row
        .select('text.feature-name')
        .attr('x', this.maxTextWidth)
        .text(shap.name)
        .select('title')
        .text(shap.fullName);

      // Update the rectangle
      const rect = row.select('rect.shap-bar');
      const curRectWidth = this.shapScale(Math.abs(shap.shap));
      if (shap.shap < 0) {
        rect
          .classed('negative', true)
          .transition(trans)
          .attr(
            'x',
            this.maxTextWidth + GAP + this.maxBarWidth / 2 - curRectWidth
          )
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth);
      } else {
        rect
          .classed('negative', false)
          .transition(trans)
          .attr('x', this.maxTextWidth + GAP + this.maxBarWidth / 2)
          .attr('y', -BAR_HEIGHT / 2)
          .attr('width', curRectWidth);
      }

      // Update the shap number
      row
        .select('text.shap-number')
        .classed('negative', shap.shap < 0)
        .text(FORMAT_2(shap.shap))
        .transition(trans)
        .attr(
          'x',
          shap.shap < 0
            ? this.maxTextWidth + GAP + this.maxBarWidth / 2 + 5
            : this.maxTextWidth + GAP + this.maxBarWidth / 2 - 5
        );
    };

    // Update the top 10 features first
    for (let i = 0; i < K; i++) {
      const shap = allShaps[i];
      updateShapRow(shap, i * ROW_HEIGHT, 1);
    }

    // Draw the rest shap values off the screen
    for (let i = K; i < this.data.featureNames.length; i++) {
      const shap = allShaps[i];
      updateShapRow(shap, this.shapSVGSize.height + 5, 0);
    }

    // Update the axis
    const axisGroup = content.select<SVGGElement>('g.axis-group');
    const axis = d3.axisBottom(shapValueScale).tickValues([-maxAbs, 0, maxAbs]);
    axisGroup.call(axis);
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
    // const backgroundSize = 10;
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
    this.curShapValues = shapValues[0];
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
    const shapValues = await explainer.explainOneInstance(x, 512);
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

    // Predict this example
    const result = await this.predict([curX]);
    if (result.length == 0) {
      console.error('ONNX returns empty result.');
      return;
    }

    this.curPred = result[0];
    this.updatePred();
    this.updateShapPlot();

    // Explain the prediction
    const shapValues = await this.explain(curX);
    this.curShapValues = shapValues[0];
    this.tabularUpdated();
    this.updateShapPlot();
  };

  /**
   * Event handler for the sample button clicking.
   */
  inputChanged = async () => {
    const curX = this.getCurX();
    const result = await this.predict([curX]);
    if (result.length == 0) {
      console.error('ONNX returns empty result.');
      return;
    }

    this.curPred = result[0];
    this.updatePred();
    this.tabularUpdated();

    // Explain the prediction
    const shapValues = await this.explain(curX);
    this.curShapValues = shapValues[0];
    this.tabularUpdated();
    this.updateShapPlot();
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
