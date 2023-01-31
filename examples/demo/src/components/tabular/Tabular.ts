import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import type {
  TabularData,
  TabularContFeature,
  TabularCatFeature
} from '../../types/common-types';

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
    const data = (await d3.json(
      `${import.meta.env.BASE_URL}data/lending-club.json`
    )) as TabularData;

    // Convert the data into structured format
    const contFeatures: TabularContFeature[] = [];
    const catFeatures: TabularCatFeature[] = [];
    const addedCatNames = new Set<string>();

    for (const [i, featureType] of data.featureTypes.entries()) {
      if (featureType === 'cont') {
        const curName = data.featureNames[i];
        contFeatures.push({
          name: curName,
          displayName: data.featureInfo[curName][0],
          desc: data.featureInfo[curName][1],
          value: 0
        });
      } else {
        const curName = data.featureNames[i].replace(/(.+)-(.+)/, '$1');

        if (!addedCatNames.has(curName)) {
          catFeatures.push({
            name: curName,
            displayName: data.featureInfo[curName][0],
            desc: data.featureInfo[curName][1],
            levelInfo: data.featureLevelInfo[curName],
            value: '0'
          });
          addedCatNames.add(curName);
        }
      }
    }
    console.log(contFeatures, catFeatures);
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

      this.message = `Success: ${probs}`;
    } catch (e) {
      this.message = `Failed: ${e}.`;
    }

    this.tabularUpdated();
  };
}
