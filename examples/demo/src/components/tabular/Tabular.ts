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
  contFeatures: TabularContFeature[] | null = null;
  catFeatures: TabularCatFeature[] | null = null;

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

    // Inference the model
    this.inference();
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
  };

  loadRandomSample = () => {
    if (this.data === null) {
      throw Error('this.data is null');
    }

    const randomIndex = d3.randomInt(this.data.xTest.length)();
    const x = this.data.xTest[randomIndex];
    const y = this.data.yTest[randomIndex];
    console.log(x, y);

    // Convert the data into structured format
    this.contFeatures = [];
    this.catFeatures = [];
    const addedCatNames = new Set<string>();

    for (const [i, featureType] of this.data.featureTypes.entries()) {
      if (featureType === 'cont') {
        const curName = this.data.featureNames[i];
        this.contFeatures.push({
          name: curName,
          displayName: this.data.featureInfo[curName][0],
          desc: this.data.featureInfo[curName][1],
          value: x[i],
          requiresInt: this.data.featureRequireInt.includes(curName),
          requiresLog: this.data.featureRequiresLog.includes(curName)
        });
      } else {
        const curName = this.data.featureNames[i].replace(/(.+)-(.+)/, '$1');

        if (!addedCatNames.has(curName)) {
          this.catFeatures.push({
            name: curName,
            displayName: this.data.featureInfo[curName][0],
            desc: this.data.featureInfo[curName][1],
            levelInfo: this.data.featureLevelInfo[curName],
            value: '0'
          });
          addedCatNames.add(curName);
        }
      }
    }
  };

  inference = async () => {
    try {
      // Create a new session and load the LightGBM model
      const session = await ort.InferenceSession.create(modelUrl);

      const x = Float32Array.from(new Array(31).fill(0));

      // Prepare feeds, use model input names as keys.
      const xTensor = new ort.Tensor('float32', x, [1, 31]);
      const feeds = { float_input: xTensor };

      // Feed inputs and run
      const results = await session.run(feeds);

      // Read from results
      const probs = results.probabilities.data;

      this.message = `Success: ${round(probs[1], 4)}`;
    } catch (e) {
      this.message = `Failed: ${e}.`;
    }

    this.tabularUpdated();
  };
}
