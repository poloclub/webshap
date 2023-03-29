import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import { tick } from 'svelte';
import type {
  TabularData,
  TabularContFeature,
  TabularCatFeature,
  Size,
  Padding,
  SHAPRow,
  TabularWorkerMessage
} from '../../types/common-types';
import { round, timeit, downloadJSON } from '../../utils/utils';
import { getLatoTextWidth } from '../../utils/text-width';
import TabularWorker from './tabular-worker?worker';
import modelUrl from './../../../../models/lending-club-xgboost.onnx?url';

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
  tabularWorker: Worker;

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
  shapPlotInitialized = false;

  // Dataset
  data: TabularData | null = null;
  contFeatures: Map<string, TabularContFeature> | null = null;
  catFeatures: Map<string, TabularCatFeature> | null = null;
  curX: number[] | null = null;
  curY: number | null = null;
  curIndex = 0;

  // Model information
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

    // Workers
    this.tabularWorker = new TabularWorker();
    this.tabularWorker.onmessage = (e: MessageEvent<TabularWorkerMessage>) => {
      this.tabularWorkerMessageHandler(e);
    };

    // Load ONNX model
    const message: TabularWorkerMessage = {
      command: 'startLoadModel',
      payload: {
        url: modelUrl
      }
    };
    this.tabularWorker.postMessage(message);
    this.updateModelLoader(true, true);

    // SVGs
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
      });
    });
  }

  /**
   * Flip the loading spinner for the data model arrow
   * @param isLoading If the model is loading
   */
  updateModelLoader = (isLoading: boolean, controlCircle = true) => {
    const lineLoader = this.component.querySelector(
      '.data-model-arrow .line-loader'
    ) as HTMLElement;

    const circleLoader = this.component.querySelector(
      '.data-model-arrow .loader-container'
    ) as HTMLElement;

    if (isLoading) {
      lineLoader.classList.remove('hidden');

      if (controlCircle) {
        circleLoader.classList.remove('hidden');
      }
    } else {
      lineLoader.classList.add('hidden');

      if (controlCircle) {
        circleLoader.classList.add('hidden');
      }
    }
  };

  /**
   * Flip the loading spinner for the explain loaders
   * @param isLoading If the model is loading
   */
  updateExplainLoader = (isLoading: boolean) => {
    const circleLoader = this.component.querySelector(
      '.model-explain-arrow .loader-container'
    ) as HTMLElement;

    const explainBoxLoader = this.component.querySelector(
      '.explain-box .loader-container'
    ) as HTMLElement;

    const lineLoader = this.component.querySelector(
      '.model-explain-arrow .line-loader'
    ) as HTMLElement;

    if (isLoading) {
      lineLoader.classList.remove('hidden');

      if (!this.shapPlotInitialized) {
        explainBoxLoader.classList.remove('hidden');
      } else {
        circleLoader.classList.remove('hidden');
      }
    } else {
      lineLoader.classList.add('hidden');
      circleLoader.classList.add('hidden');
      explainBoxLoader.classList.add('hidden');
    }
  };

  /**
   * Handling worker messages
   * @param e Message event
   */
  tabularWorkerMessageHandler = (e: MessageEvent<TabularWorkerMessage>) => {
    switch (e.data.command) {
      case 'finishLoadModel': {
        this.updateModelLoader(false, true);
        break;
      }

      case 'finishPredict': {
        const posProbs = e.data.payload.posProbs;
        this.curPred = posProbs[0][0];
        this.updatePred();
        this.updateModelLoader(false, false);

        break;
      }

      case 'finishExplain': {
        const shapValues = e.data.payload.shapValues;
        this.curShapValues = shapValues[0];

        if (this.shapPlotInitialized) {
          this.updateShapPlot();
        } else {
          this.initShapPlot();
        }
        this.updateExplainLoader(false);

        break;
      }

      default: {
        console.error('Worker: unknown message', e.data.command);
        break;
      }
    }
  };

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

    // Init with 0 as default pred score
    const curPred = 0;

    content
      .append('rect')
      .attr('class', 'top-rect')
      .classed('approval', curPred ? curPred >= 0.5 : true)
      .attr('rx', this.predBarSVGSize.height / 2)
      .attr('ry', this.predBarSVGSize.height / 2)
      .attr('width', this.predBarScale(curPred || 0))
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
    if (this.shapPlotInitialized)
      throw Error('shap plot is already initailized.');

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

    this.shapPlotInitialized = true;
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

    this.updateExplainLoader(true);
    this.updateModelLoader(true, false);

    // Load a random sample
    this.loadRandomSample();

    // Inference the model
    const x = this.getCurX();
    const predictMessage: TabularWorkerMessage = {
      command: 'startPredict',
      payload: {
        x: [x]
      }
    };
    this.tabularWorker.postMessage(predictMessage);

    // Explain this instance
    // Create background data for SHAP
    this.backgroundData = [];

    // Take 10 random training data
    // const backgroundSize = 50;
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

    // Explain the prediction
    const explainMessage: TabularWorkerMessage = {
      command: 'startExplain',
      payload: {
        x,
        backgroundData: this.backgroundData
      }
    };
    this.tabularWorker.postMessage(explainMessage);
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
    const randomIndex = d3.randomInt(this.data.xTest.length)();
    // const randomIndex = RANDOM_INT(this.data.xTest.length)();

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
   * Event handler for the sample button clicking.
   */
  sampleClicked = () => {
    this.loadRandomSample();

    // Predict this example
    this.getNewExplanation();
  };

  /**
   * Event handler for the sample button clicking.
   */
  inputChanged = () => {
    this.curX = this.getCurX();

    // Predict this example
    this.getNewExplanation();
  };

  getNewExplanation = () => {
    if (this.curX === null) {
      throw new Error('curX is null');
    }

    this.updateModelLoader(true, false);
    this.updateExplainLoader(true);

    const predictMessage: TabularWorkerMessage = {
      command: 'startPredict',
      payload: {
        x: [this.curX]
      }
    };
    this.tabularWorker.postMessage(predictMessage);
    // Explain the prediction
    const explainMessage: TabularWorkerMessage = {
      command: 'startExplain',
      payload: {
        x: this.curX,
        backgroundData: this.backgroundData
      }
    };
    this.tabularWorker.postMessage(explainMessage);
  };

  /**
   * Helper function to update the view with the new prediction result
   */
  updatePred = () => {
    if (this.curPred === null) {
      throw Error('curPred is null');
    }

    // Update the bar
    const content = this.predBarSVG.select('g.content');
    content
      .select('rect.top-rect')
      .classed('approval', this.curPred >= 0.5)
      .attr('width', this.predBarScale(this.curPred));

    this.tabularUpdated();
  };
}
