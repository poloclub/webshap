import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import type {
  TabularData,
  TabularContFeature,
  TabularCatFeature
} from '../../types/common-types';
import { round } from '../../utils/utils';

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

/**
 * Class for the Embedding view
 */

export class Tabular {
  component: HTMLElement;
  tabularUpdated: () => void;

  // Dataset
  data: TabularData | null = null;
  contFeatures: Map<string, TabularContFeature> | null = null;
  catFeatures: Map<string, TabularCatFeature> | null = null;
  curX: number[] | null = null;
  curY: number | null = null;
  curIndex = 0;

  // ONNX data
  message: string;

  /**
   *
   * @param args Named parameters
   * @param args.component The component
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

    // Load the training and test dataset
    this.initData();
  }

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
    this.inference(x);
  };

  /**
   * Load a random sample from the test dataset.
   */
  loadRandomSample = () => {
    if (this.data === null) {
      throw Error('this.data is null');
    }

    // Get a random instance
    const randomIndex = d3.randomInt(this.data.xTest.length)();
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
   * Event handler for the sample button clicking.
   */
  sampleClicked = () => {
    this.loadRandomSample();
    const curX = this.getCurX();
    this.inference(curX);
    console.log(curX);
  };

  /**
   * Event handler for the sample button clicking.
   */
  inputChanged = () => {
    const curX = this.getCurX();
    console.log(curX);
    this.inference(curX);
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

  inference = async (x: number[]) => {
    try {
      // Create a new session and load the LightGBM model
      const session = await ort.InferenceSession.create(modelUrl);

      // Prepare feeds, use model input names as keys.
      const xTensor = new ort.Tensor('float32', Float32Array.from(x), [1, 31]);
      const feeds = { float_input: xTensor };

      // Feed inputs and run
      const results = await session.run(feeds);

      // Read from results
      const probs = results.probabilities.data as Float32Array;

      this.message = `Success: ${round(probs[1], 4)}`;
    } catch (e) {
      this.message = `Failed: ${e}.`;
    }

    this.tabularUpdated();
  };
}
