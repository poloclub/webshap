import d3 from '../../utils/d3-import';
import type {
  TextWorkerMessage,
  TextExplainResult
} from '../../types/common-types';
import { timeit, sleep } from '../../utils/utils';
import { config } from '../../config/config';
import { KernelSHAP } from 'webshap';
import { loadTokenizer } from './bert-tokenizer';
import type { BertTokenizer } from './bert-tokenizer';

import * as ort from 'onnxruntime-web/dist/ort-web.min.js';
import wasm from 'onnxruntime-web/dist/ort-wasm.wasm?url';
import wasmThreaded from 'onnxruntime-web/dist/ort-wasm-threaded.wasm?url';
import wasmSimd from 'onnxruntime-web/dist/ort-wasm-simd.wasm?url';
import wasmSimdThreaded from 'onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url';

// const NUM_SAMPLES = 512;
const NUM_SAMPLES = 128;

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
      break;
    }

    case 'startPredict': {
      const inputText = e.data.payload.inputText;
      predict(inputText);
      break;
    }

    case 'startExplain': {
      const inputText = e.data.payload.inputText;
      explainInputText(inputText);
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
    return model.tokenizer.tokenize(inputText);
  });
};

/**
 * Get the model's input (token IDs, attention mask, type ids) in Tensor format
 * @param inputText Input text
 */
const getONNXInput = (tokenIDs: number[]) => {
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

  // We need to use the exact input name defined in the onnx model
  const onnxInput = {
    input_ids: inputIDsTensor,
    attention_mask: attentionMasksTensor,
    token_type_ids: tokenTypeIDsTensor
  };

  return onnxInput;
};

const predict = async (inputText: string) => {
  const [tokenIDs, tokenWords] = await tokenize(inputText);
  const onnxInput = getONNXInput(tokenIDs);
  const model = await loadModel;
  const output = await model.session.run(onnxInput);

  // Convert logits into probabilities
  const logits = output['output_0'].data as Float32Array;
  const probs = softmax([...logits]);

  const result = {
    inputText,
    tokenWords,
    probs
  };

  // Send the result back to the main thread
  const message: TextWorkerMessage = {
    command: 'finishPredict',
    payload: {
      result
    }
  };
  postMessage(message);
};

/**
 * Explain the current input image
 */
const explainInputText = async (inputText: string) => {
  const model = await loadModel;

  const [tokenIDs, tokenWords] = model.tokenizer.tokenize(inputText);

  // We need to create the prediction function as a closure because the number
  // of tokenIDs can vary
  const maskedPredict = async (tokenIDMasks: number[][]) => {
    // Step 1: create the default inputs
    const inputIDs = new Array<bigint>(tokenIDs.length + 2);
    const tokenTypeIDs = new Array<bigint>(tokenIDs.length + 2).fill(BigInt(0));

    // 101 is the [CLS] token
    inputIDs[0] = BigInt(101);

    for (let i = 0; i < tokenIDs.length; i++) {
      inputIDs[i + 1] = BigInt(tokenIDs[i]);
    }

    // 102 is the [SEP] token
    inputIDs[tokenIDs.length + 1] = BigInt(102);

    // Step 2: create collision inputs by varying the attention masks
    // Create big int arrays
    const allInputIDs: bigint[] = [];
    const allAttentionMasks: bigint[] = [];
    const allTokenTypeIDs: bigint[] = [];

    for (let r = 0; r < tokenIDMasks.length; r++) {
      // We won't change token IDs or type IDs
      allInputIDs.push(...inputIDs);
      allTokenTypeIDs.push(...tokenTypeIDs);

      // We will vary attention masks based on the current perturbation
      const curAttentionMasks = new Array<bigint>(tokenIDs.length + 2).fill(
        BigInt(1)
      );

      for (let c = 0; c < tokenIDMasks[r].length; c++) {
        if (tokenIDMasks[r][c] === 0) {
          curAttentionMasks[c + 1] = BigInt(0);
        }
      }
      allAttentionMasks.push(...curAttentionMasks);
    }

    // Step 3: create batch tensors
    // Convert arrays into tensors
    const inputIDsTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(allInputIDs),
      [tokenIDMasks.length, tokenIDs.length + 2]
    );

    const attentionMasksTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(allAttentionMasks),
      [tokenIDMasks.length, tokenIDs.length + 2]
    );

    const tokenTypeIDsTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(allTokenTypeIDs),
      [tokenIDMasks.length, tokenIDs.length + 2]
    );

    // We need to use the exact input name defined in the onnx model
    const onnxInput = {
      input_ids: inputIDsTensor,
      attention_mask: attentionMasksTensor,
      token_type_ids: tokenTypeIDsTensor
    };

    // Step 4: run the model on this batch
    const output = await model.session.run(onnxInput);

    // Step 5: Convert logits into probabilities
    const logits = output['output_0'].data as Float32Array;
    const predictedProbs: number[][] = [];

    for (let i = 0; i < logits.length; i += 2) {
      const curLogits = [logits[i], logits[i + 1]];
      const probs = softmax(curLogits);
      predictedProbs.push([probs[1]]);
    }

    // Step 6: return a promise
    const promise = new Promise<number[][]>(resolve => {
      resolve(predictedProbs);
    });
    return promise;
  };

  // The background data would be all masks (no attended words!)
  const backgroundData = [new Array<number>(tokenIDs.length).fill(0)];
  const explainer = new KernelSHAP(maskedPredict, backgroundData, 0.2022);

  // To explain the prediction on this image, we provide the "feature" as
  // showing all segments
  timeit('Explain text', DEBUG);
  const allTokenData = new Array<number>(tokenIDs.length).fill(1);
  const shapValues = await explainer.explainOneInstance(
    allTokenData,
    NUM_SAMPLES
  );
  timeit('Explain text', DEBUG);

  const result: TextExplainResult = {
    inputText,
    tokenWords,
    shapValues: shapValues[0]
  };

  // Send the result to the main thread
  const message: TextWorkerMessage = {
    command: 'finishExplain',
    payload: {
      result
    }
  };
  postMessage(message);
};

/**
 * Helper function for softmax
 * @param logits Logits
 * @returns Class probabilities
 */
const softmax = (logits: number[]) => {
  const expSum = logits.reduce((a, b) => a + Math.exp(b), 0);
  return logits.map(d => Math.exp(d) / expSum);
};
