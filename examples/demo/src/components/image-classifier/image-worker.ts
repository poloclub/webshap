import { config } from '../../config/config';
import type {
  ImageSegmentation,
  ImageWorkerMessage
} from '../../types/common-types';
import { KernelSHAP } from 'webshap';
import { timeit } from '../../utils/utils';
import { fill, tensor3d, loadLayersModel, stack } from '@tensorflow/tfjs';
import type { Tensor3D, LayersModel, Tensor, Rank } from '@tensorflow/tfjs';

const DEBUG = config.debug;
const IMG_SRC_LENGTH = config.layout.imageSrcLength;
const IMG_LENGTH = config.layout.imageLength;

// const NUM_SAMPLES = 512;
const NUM_SAMPLES = 128;

let loadModel: Promise<LayersModel>;

/**
 * Handle message events from the main thread
 * @param e Message event
 */
self.onmessage = (e: MessageEvent<ImageWorkerMessage>) => {
  switch (e.data.command) {
    case 'startLoadModel': {
      const modelUrl = e.data.payload.url;
      loadModel = startLoadModel(modelUrl);
      break;
    }

    case 'startPredict': {
      const inputImageData = e.data.payload.inputImageData;
      modelInference(inputImageData);
      break;
    }

    case 'startExplain': {
      const inputImageData = e.data.payload.inputImageData;
      const inputImageSeg = e.data.payload.inputImageSeg;
      explainInputImage(inputImageData, inputImageSeg);
      break;
    }

    default: {
      console.error('Worker: unknown message', e.data.command);
      break;
    }
  }
};

/**
 * Initialize the trained Tiny VGG model.
 */
const startLoadModel = async (modelURL: string) => {
  const model = await loadLayersModel(modelURL);
  const message: ImageWorkerMessage = {
    command: 'finishLoadModel',
    payload: {}
  };
  postMessage(message);
  return model;
};

/**
 * Run the model on the current input image
 * @returns Class probabilities
 */
const modelInference = async (inputImage: ImageData) => {
  const model = await loadModel;

  const imageTensor = imageDataTo3DTensor(
    inputImage.data,
    inputImage.width,
    inputImage.height,
    true
  );

  const inputImageTensorBatch = stack([imageTensor]);
  const predictedProbTensor = model.call(
    inputImageTensorBatch,
    {}
  ) as Tensor<Rank>[];
  const predictedProb = predictedProbTensor[0].dataSync() as Float32Array;

  // Send back the prediction result
  const message: ImageWorkerMessage = {
    command: 'finishPredict',
    payload: {
      predictedProb
    }
  };
  postMessage(message);

  return predictedProb;
};

/**
 * Explain the current input image
 */
const explainInputImage = async (
  inputImageData: ImageData,
  inputImageSeg: ImageSegmentation
) => {
  const model = await loadModel;

  // We need to create the prediction function as a closure because the number
  // of segments can vary
  const predict = (segMasks: number[][]) => {
    // Step 1: convert segMasks into masked image tensors
    const maskedImageTensors: Tensor3D[] = [];
    for (const segMask of segMasks) {
      const curMaskedImageArray = getMaskedImageData(
        inputImageData.data,
        inputImageSeg.segData.data,
        segMask
      );

      const curMaskedTensor = imageDataTo3DTensor(
        curMaskedImageArray,
        IMG_SRC_LENGTH,
        IMG_SRC_LENGTH,
        true
      );

      maskedImageTensors.push(curMaskedTensor);
    }

    // Step 2: create a batch tensor (4D)
    const batchTensor = stack(maskedImageTensors);

    // Step 3: run the model on this batch
    const predictedProbTensor = model.call(batchTensor, {}) as Tensor<Rank>[];
    const predictedProb = predictedProbTensor[0].arraySync() as number[][];

    // Step 4: return a promise
    const promise = new Promise<number[][]>(resolve => {
      resolve(predictedProb);
    });
    return promise;
  };

  // The background data would be empty image (white color)
  // We represent "features" as a binary array of segSize elements
  // 0: use white color for the ith segment
  // 1: use the input image's segment for the ith segment
  const backgroundData = [new Array<number>(inputImageSeg.segSize).fill(0)];
  const explainer = new KernelSHAP(predict, backgroundData, 0.2022);

  // To explain the prediction on this image, we provide the "feature" as
  // showing all segments
  timeit('Explain image', DEBUG);
  const allSegData = new Array<number>(inputImageSeg.segSize).fill(1);
  const shapValues = await explainer.explainOneInstance(
    allSegData,
    NUM_SAMPLES
  );
  timeit('Explain image', DEBUG);

  const message: ImageWorkerMessage = {
    command: 'finishExplain',
    payload: {
      shapValues
    }
  };
  postMessage(message);

  return shapValues;
};

/**
 * Get the masked image array
 * @param imageArray Image array
 * @param segArray Segmentation array, R value is the segmentation index
 * @param segMask Binary array: 1 => show image segmentation, 0 => nothing
 * @param background Background color
 * @returns Masked image array
 */
const getMaskedImageData = (
  imageArray: Uint8ClampedArray,
  segArray: Uint8ClampedArray,
  segMask: number[],
  background = 255
) => {
  // Collect the segmentation index to show
  const segIndexes = new Set<number>();
  for (const [i, m] of segMask.entries()) {
    if (m !== 0) {
      segIndexes.add(i);
    }
  }

  // Fill the image array with the correct RGB values
  const output = new Array<number>(imageArray.length).fill(background);
  for (let i = 0; i < imageArray.length; i += 4) {
    if (segIndexes.has(segArray[i])) {
      for (let j = 0; j < 3; j++) {
        output[i + j] = imageArray[i + j];
      }
    }
  }

  return new Uint8ClampedArray(output);
};

/**
 * Crop the largest central square of size IMG_SRC_LENGTH x IMG_SRC_LENGTH x 3
 * of a 3d array.
 *
 * @param {number[][][]} arr array that requires cropping and padding (if a
 * IMG_SRC_LENGTH x IMG_SRC_LENGTH crop is not present)
 * @returns IMG_SRC_LENGTH x IMG_SRC_LENGTH x 3 array
 */
const cropCentralSquare = (arr: number[][][]) => {
  const width = arr.length;
  const height = arr[0].length;
  let croppedArray: number[][][] = [];

  if (width < IMG_SRC_LENGTH || height < IMG_SRC_LENGTH) {
    throw Error('Image size is smaller than the specified length.');
  }

  // Crop largest square from image
  const startXIdx = Math.floor(width / 2) - Math.floor(IMG_SRC_LENGTH / 2);
  const startYIdx = Math.floor(height / 2) - Math.floor(IMG_SRC_LENGTH / 2);
  croppedArray = arr
    .slice(startXIdx, startXIdx + IMG_SRC_LENGTH)
    .map(i => i.slice(startYIdx, startYIdx + IMG_SRC_LENGTH));
  return croppedArray;
};

/**
 * Convert canvas image data into a 3D tensor with dimension [height, width, 3].
 * Recall that tensorflow uses NHWC order (batch, height, width, channel).
 * Each pixel is in 0-255 scale.
 *
 * @param imageData Canvas image data
 * @param width Canvas image width
 * @param height Canvas image height
 */
const imageDataTo3DTensor = (
  imageData: Uint8ClampedArray,
  width: number,
  height: number,
  normalize = true
) => {
  // Create array placeholder for the 3d array
  let imageArray = fill([width, height, 3], 0).arraySync() as number[][][];

  // Iterate through the data to fill out channel arrays above
  for (let i = 0; i < imageData.length; i++) {
    const pixelIndex = Math.floor(i / 4),
      channelIndex = i % 4,
      row =
        width === height ? Math.floor(pixelIndex / width) : pixelIndex % width,
      column =
        width === height ? pixelIndex % width : Math.floor(pixelIndex / width);

    if (channelIndex < 3) {
      let curEntry = imageData[i];
      // Normalize the original pixel value from [0, 255] to [0, 1]
      if (normalize) {
        curEntry /= 255;
      }
      imageArray[row][column][channelIndex] = curEntry;
    }
  }

  // If the image is not 64x64, crop and or pad the image appropriately.
  if (width != IMG_SRC_LENGTH && height != IMG_SRC_LENGTH) {
    imageArray = cropCentralSquare(imageArray);
  }

  const tensor = tensor3d(imageArray);
  return tensor;
};
