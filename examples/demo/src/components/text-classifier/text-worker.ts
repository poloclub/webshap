import d3 from '../../utils/d3-import';
import type { TextWorkerMessage } from '../../types/common-types';
import { timeit, sleep } from '../../utils/utils';
import { config } from '../../config/config';
import { loadTokenizer } from './bert-tokenizer';
import type { BertTokenizer } from './bert-tokenizer';

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

interface LoadedModel {
  session: ort.InferenceSession;
  tokenizer: BertTokenizer;
}

const DEBUG = config.debug;
let loadModel: Promise<LoadedModel>;

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<TextWorkerMessage>) => {
  switch (e.data.command) {
    case 'startLoadModel': {
      const modelUrl = e.data.payload.url;
      loadModel = startLoadModel(modelUrl);
      getModelInput("Son, you're too young GPU vandalism.");
      break;
    }

    default: {
      console.error('Worker: unknown message', e.data.command);
      break;
    }
  }
};

const startLoadModel = async (url: string) => {
  timeit('Load text model', DEBUG);

  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  };
  const session = await ort.InferenceSession.create(url, options);
  const tokenizer = loadTokenizer();

  // Tell the main thread that we have finished loading
  const message: TextWorkerMessage = {
    command: 'finishLoadModel',
    payload: {}
  };
  timeit('Load text model', DEBUG);

  // await sleep(3000);
  postMessage(message);

  return {
    session,
    tokenizer
  };
};

const tokenize = (inputText: string) => {
  // Tokenize the input text
  return loadModel.then(model => {
    const tokens = model.tokenizer.tokenize(inputText);
    return tokens;
  });
};

const detokenize = (tokenIDs: number[]) => {
  return loadModel.then(model => {
    // TODO
  });
};

/**
 * Get the model's input (token IDs, attention mask, type ids) in Tensor format
 * @param inputText Input text
 */
const getModelInput = async (inputText: string) => {
  const [tokenIDs, tokenWords] = await tokenize(inputText);
  console.log(tokenWords);

  // Create big int arrays
  const inputIDs = new Array<bigint>(tokenIDs.length + 2);
  const attentionMasks = new Array<bigint>(tokenIDs.length + 2).fill(BigInt(1));
  const tokenTypeIDs = new Array<bigint>(tokenIDs.length + 2).fill(BigInt(0));

  // 101 is the [CLS] token
  inputIDs[0] = BigInt(101);

  for (let i = 0; i < tokenIDs.length; i++) {
    inputIDs[i + 1] = BigInt(tokenIDs[i]);
  }

  // 102 is the [SEP] token
  inputIDs[tokenIDs.length + 1] = BigInt(102);

  // Convert arrays into tensors
  const inputIDsTensor = new ort.Tensor('int64', BigInt64Array.from(inputIDs), [
    1,
    tokenIDs.length + 2
  ]);

  const attentionMasksTensor = new ort.Tensor(
    'int64',
    BigInt64Array.from(attentionMasks),
    [1, tokenIDs.length + 2]
  );

  const tokenTypeIDsTensor = new ort.Tensor(
    'int64',
    BigInt64Array.from(tokenTypeIDs),
    [1, tokenIDs.length + 2]
  );

  console.log(inputIDsTensor, attentionMasksTensor, tokenTypeIDsTensor);

  return {
    inputIDsTensor,
    attentionMasksTensor,
    tokenTypeIDsTensor
  };
};
