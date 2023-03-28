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
import { KernelSHAP } from 'webshap';
import { round, timeit, downloadJSON } from '../../utils/utils';
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

let loadModel: Promise<ort.InferenceSession>;

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<TabularWorkerMessage>) => {
  switch (e.data.command) {
    case 'startLoadModel': {
      const modelUrl = e.data.payload.url;
      loadModel = startLoadModel(modelUrl);
      break;
    }

    case 'startPredict': {
      const x = e.data.payload.x;
      predict(x, true);
      break;
    }

    case 'startExplain': {
      const x = e.data.payload.x;
      const backgroundData = e.data.payload.backgroundData;
      explain(x, backgroundData);
      break;
    }

    default: {
      console.error('Worker: unknown message', e.data.command);
      break;
    }
  }
};

const startLoadModel = async (url: string) => {
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  };
  const session = await ort.InferenceSession.create(url, options);

  // Tell the main thread that we have finished loading
  const message: TabularWorkerMessage = {
    command: 'finishLoadModel',
    payload: {}
  };

  postMessage(message);

  return session;
};

/**
 * Run XGBoost on the given input data x
 * @param x Input data instances (n, k)
 * @param notifyMainThread True if it needs to send message to main thread
 * @returns Predicted positive label probabilities (n)
 */
const predict = async (x: number[][], notifyMainThread: boolean) => {
  const model = await loadModel;
  const posProbs: number[][] = [];

  try {
    // First need to flatten the x array
    const xFlat = Float32Array.from(x.flat());

    // Prepare feeds, use model input names as keys.
    const xTensor = new ort.Tensor('float32', xFlat, [x.length, x[0].length]);
    const feeds = { float_input: xTensor };

    // Feed inputs and run
    const results = await model.run(feeds);

    // Read from results, probs has shape (n * 2) => (n, 2)
    const probs = results.probabilities.data as Float32Array;

    for (const [i, p] of probs.entries()) {
      // Positive label prob is always at the odd index
      if (i % 2 === 1) {
        posProbs.push([p]);
      }
    }

    if (notifyMainThread) {
      const message: TabularWorkerMessage = {
        command: 'finishPredict',
        payload: {
          posProbs: posProbs
        }
      };
      postMessage(message);
    }
  } catch (e) {
    console.error(`Failed model prediction: ${e}.`);
  }

  return posProbs;
};

/**
 * Run WebSHAP to explain the given input data x
 * @param x Input data instance
 */
const explain = async (x: number[], backgroundData: number[][]) => {
  const explainer = new KernelSHAP(
    (x: number[][]) => predict(x, false),
    backgroundData,
    0.2022
  );

  timeit('Explain tabular', DEBUG);
  const shapValues = await explainer.explainOneInstance(x, 512);
  // const shapValues = await explainer.explainOneInstance(x);
  timeit('Explain tabular', DEBUG);

  const message: TabularWorkerMessage = {
    command: 'finishExplain',
    payload: {
      shapValues
    }
  };
  postMessage(message);
  return shapValues;
};
