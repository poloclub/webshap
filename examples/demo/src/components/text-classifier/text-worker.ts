import d3 from '../../utils/d3-import';
import type { TextWorkerMessage } from '../../types/common-types';
import { timeit } from '../../utils/utils';
import { config } from '../../config/config';

import * as ort from 'onnxruntime-web/dist/ort-web.min.js';
import wasm from 'onnxruntime-web/dist/ort-wasm.wasm?url';
import wasmThreaded from 'onnxruntime-web/dist/ort-wasm-threaded.wasm?url';
import wasmSimd from 'onnxruntime-web/dist/ort-wasm-simd.wasm?url';
import wasmSimdThreaded from 'onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';

// Set up the correct WASM paths
ort.env.wasm.wasmPaths = {
  'ort-wasm.wasm': wasm,
  'ort-wasm-threaded.wasm': wasmThreaded,
  'ort-wasm-simd.wasm': wasmSimd,
  'ort-wasm-simd-threaded.wasm': wasmSimdThreaded
};

const DEBUG = config.debug;
let session: ort.InferenceSession | null = null;

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = async (e: MessageEvent<TextWorkerMessage>) => {
  // Stream point data
  switch (e.data.command) {
    case 'startLoadModel': {
      const modelUrl = e.data.payload.url;
      // const modelUrl = '/models/text-classifier/lending-club-lightgbm.onnx';
      timeit('Load text model', DEBUG);
      await startLoadModel(modelUrl);
      timeit('Load text model', DEBUG);

      console.log(session);
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
  session = await ort.InferenceSession.create(url, options);
};
