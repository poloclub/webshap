import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import { SLIC } from './segmentation/slic';
import type { SuperPixelOptions } from './segmentation/slic';
import { tick } from 'svelte';
import type {
  LoadedImage,
  Size,
  Padding,
  ImageSegmentation
} from '../../types/common-types';
import { KernelSHAP } from 'webshap';
import { round, timeit, downloadJSON } from '../../utils/utils';
import { getLatoTextWidth } from '../../utils/text-width';
import { fill, tensor3d, loadLayersModel, stack } from '@tensorflow/tfjs';
import type { Tensor3D, LayersModel, Tensor, Rank } from '@tensorflow/tfjs';

const DEBUG = config.debug;
const LCG = d3.randomLcg(0.20230101);

// const NUM_SAMPLES = 512;
const NUM_SAMPLES = 128;
const NUM_CLASS = 4;
const DIVERGE_COLORS = [config.colors['pink-600'], config.colors['blue-700']];

const IMG_SRC_LENGTH = 64;
const IMG_LENGTH = 134;

/**
 * Class for the Image Classifier WebSHAP demo
 */

export class ImageClassifier {
  component: HTMLElement;
  imageClassifierUpdated: () => void;

  // Canvas elements
  inputCanvas: HTMLCanvasElement;
  segCanvas: HTMLCanvasElement;
  explainCanvases: HTMLCanvasElement[];
  inputBackCanvases: HTMLCanvasElement[];
  outputWrappers: d3.Selection<HTMLElement, unknown, null, undefined>[];

  // SVG selections
  colorScaleSVG: d3.Selection<HTMLElement, unknown, null, undefined>;

  // Visualizations
  colorScale: (t: number) => string;
  shapScale: d3.ScaleLinear<number, number, never>;
  shapLengthScale: d3.ScaleLinear<number, number, never>;
  colorLegendAxis: d3.Axis<d3.NumberValue>;
  predProbScale: d3.ScaleLinear<number, number, never>;

  // ML inference
  inputImage: LoadedImage | null = null;
  imageSeg: ImageSegmentation | null = null;
  model: LayersModel | null = null;

  /**
   * @param args Named parameters
   * @param args.component The component
   * @param args.imageClassifierUpdated A function to trigger updates
   */
  constructor({
    component,
    imageClassifierUpdated
  }: {
    component: HTMLElement;
    imageClassifierUpdated: () => void;
  }) {
    this.component = component;
    this.imageClassifierUpdated = imageClassifierUpdated;

    // Initialize canvas elements
    this.inputCanvas = initCanvasElement(
      this.component,
      '.input-image',
      IMG_LENGTH
    );
    this.segCanvas = initCanvasElement(
      this.component,
      '.seg-image',
      IMG_LENGTH
    );

    this.explainCanvases = [];
    this.inputBackCanvases = [];
    this.outputWrappers = [];

    for (let i = 0; i < NUM_CLASS; i++) {
      const explainCanvas = initCanvasElement(
        this.component,
        `.explain-wrapper-${i} .explain-image`,
        IMG_LENGTH
      );
      this.explainCanvases.push(explainCanvas);

      const inputCanvas = initCanvasElement(
        this.component,
        `.explain-wrapper-${i} .input-image-back`,
        IMG_LENGTH
      );
      this.inputBackCanvases.push(inputCanvas);

      const outputWrapper = d3
        .select(component)
        .select<HTMLElement>(`.output-wrapper-${i}`);
      this.outputWrappers.push(outputWrapper);
    }

    // Initialize the prediction rect scale
    const barBackBBox = this.outputWrappers[0]
      .select<HTMLElement>('.class-score-back')
      .node()!
      .getBoundingClientRect();

    this.predProbScale = d3
      .scaleLinear()
      .domain([0, 1])
      .range([0, barBackBBox.height]);

    // Initialize SVG elements
    this.colorScale = d3.piecewise(d3.interpolateHsl, [
      DIVERGE_COLORS[0],
      'white',
      DIVERGE_COLORS[1]
    ]) as (t: number) => string;

    this.shapScale = d3.scaleLinear().domain([-0.5, 0.5]);
    this.shapLengthScale = d3.scaleLinear().domain([-0.5555, 0.5555]);
    this.colorLegendAxis = d3.axisBottom(this.shapLengthScale);

    this.colorScaleSVG = d3
      .select(this.component)
      .select<HTMLElement>('svg.color-scale-svg');
    this.initSVGs();

    // Initialize the classifier and input image
    const modelPromise = this.initModel();
    const imagePromise = this.loadInputImage();

    Promise.all([modelPromise, imagePromise]).then(() => {
      this.updateVisualizations();
    });
  }

  /**
   * Update visualizations for the model prediction and explanations
   */
  updateVisualizations = async () => {
    // Get the model prediction
    const predictedProb = this.modelInference();
    const formatter = d3.format('.4f');

    // Update the bar chart
    for (const [i, output] of this.outputWrappers.entries()) {
      const frontRect = output.select<HTMLElement>('.class-score-front');
      frontRect.style('height', `${this.predProbScale(predictedProb[i])}px`);
      frontRect.select('.class-score-label').text(formatter(predictedProb[i]));
    }

    // Explain the model prediction
    const shapValues = await this.explainInputImage();

    // Show the explanations
    // Need to get the min and max of shap values across all classes
    const shapRange: [number, number] = [Infinity, -Infinity];
    for (const row of shapValues) {
      for (const value of row) {
        if (value < shapRange[0]) {
          shapRange[0] = value;
        }
        if (value > shapRange[1]) {
          shapRange[1] = value;
        }
      }
    }

    // Make the shap range symmetric around 0
    if (Math.abs(shapRange[1]) > Math.abs(shapRange[0])) {
      if (shapRange[1] > 0) {
        shapRange[0] = -shapRange[1];
      } else {
        shapRange[0] = shapRange[1];
        shapRange[1] = -shapRange[1];
      }
    } else {
      if (shapRange[0] < 0) {
        shapRange[1] = -shapRange[0];
      } else {
        shapRange[1] = shapRange[0];
        shapRange[0] = -shapRange[0];
      }
    }

    // Update the shap scales
    this.shapScale.domain(shapRange);
    this.shapLengthScale.domain(shapRange);

    // Update the color legend scale
    const scaleAxisGroup =
      this.colorScaleSVG.select<SVGGElement>('g.axis-group');
    this.colorLegendAxis = d3
      .axisBottom(this.shapLengthScale)
      .tickValues([
        this.shapLengthScale.domain()[0],
        0,
        this.shapLengthScale.domain()[1]
      ])
      .tickFormat(d3.format('.2f'));
    scaleAxisGroup.call(this.colorLegendAxis);
    scaleAxisGroup.attr('font-size', null);

    // Create a buffer context to resize image
    const hiddenCanvas = createBufferCanvas(IMG_SRC_LENGTH);
    const hiddenCtx = hiddenCanvas.getContext('2d')!;

    // Update explanation images
    for (let c = 0; c < NUM_CLASS; c++) {
      const shapImage = this.getExplanationImage(shapValues[c]);
      const explainCtx = this.explainCanvases[c].getContext('2d')!;

      // Draw the input image on screen
      hiddenCtx.clearRect(0, 0, IMG_SRC_LENGTH, IMG_SRC_LENGTH);
      hiddenCtx.putImageData(shapImage, 0, 0);

      explainCtx.drawImage(
        hiddenCanvas,
        0,
        0,
        IMG_SRC_LENGTH,
        IMG_SRC_LENGTH,
        0,
        0,
        IMG_LENGTH,
        IMG_LENGTH
      );
    }

    hiddenCanvas.remove();
  };

  /**
   * Initialize SVG elements
   */
  initSVGs = () => {
    // Initialize the color scale svg
    const bbox = this.colorScaleSVG.node()!.getBoundingClientRect();
    const svgSize: Size = {
      width: bbox.width,
      height: bbox.height
    };

    const svgPadding: Padding = {
      top: 0,
      left: 15,
      right: 15,
      bottom: 10
    };

    const rectHeight = 10;

    const contentGroup = this.colorScaleSVG
      .append('g')
      .attr('class', 'content');

    const axisGroup = contentGroup
      .append('g')
      .attr('class', 'axis-group')
      .attr(
        'transform',
        `translate(${svgPadding.left + 1}, ${rectHeight - 1})`
      );

    contentGroup
      .append('rect')
      .attr('class', 'scale-rect')
      .attr('x', svgPadding.left)
      .attr('width', svgSize.width - svgPadding.left - svgPadding.right)
      .attr('height', rectHeight)
      .attr('fill', 'url(#scale-gradient)');

    // Fill the rect with a diverging color gradient
    const gradients = this.colorScaleSVG
      .append('defs')
      .append('linearGradient')
      .attr('id', 'scale-gradient');

    const splits = 5;
    for (let i = 0; i < splits; i++) {
      const curStep = i / (splits - 1);
      gradients
        .append('stop')
        .attr('offset', `${curStep * 100}%`)
        .attr(
          'stop-color',
          `${d3.color(this.colorScale(curStep))!.formatHsl()}`
        );
    }

    // Add a legend below the color scale
    this.shapLengthScale.range([
      0,
      svgSize.width - svgPadding.left - svgPadding.right - 2
    ]);

    this.colorLegendAxis = d3
      .axisBottom(this.shapLengthScale)
      .tickValues([
        this.shapLengthScale.domain()[0],
        0,
        this.shapLengthScale.domain()[1]
      ])
      .tickFormat(d3.format('.2f'));
    axisGroup.call(this.colorLegendAxis);
    axisGroup.attr('font-size', null);
  };

  /**
   * Initialize the trained Tiny VGG model.
   */
  initModel = async () => {
    const modelFile = `${
      import.meta.env.BASE_URL
    }models/image-classifier/model.json`;
    this.model = await loadLayersModel(modelFile);
  };

  /**
   * Run the model on the current input image
   * @returns Class probabilities
   */
  modelInference = () => {
    if (this.model === null || this.inputImage === null) {
      throw Error('Model or input image is not initialized.');
    }

    // Need to feed the model with a batch
    const inputImageTensorBatch = stack([this.inputImage.imageTensor]);
    const predictedProbTensor = this.model.call(
      inputImageTensorBatch,
      {}
    ) as Tensor<Rank>[];
    const predictedProb = predictedProbTensor[0].dataSync();

    return predictedProb;
  };

  /**
   * Explain the current input image
   */
  explainInputImage = async () => {
    if (
      this.model === null ||
      this.imageSeg === null ||
      this.inputImage === null
    ) {
      throw Error('Model or data is not initialized.');
    }

    // We need to create the prediction function as a closure because the number
    // of segments can vary
    const predict = (segMasks: number[][]) => {
      // Step 1: convert segMasks into masked image tensors
      const maskedImageTensors: Tensor3D[] = [];
      for (const segMask of segMasks) {
        const curMaskedImageArray = getMaskedImageData(
          this.inputImage!.imageData.data,
          this.imageSeg!.segData.data,
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
      const predictedProbTensor = this.model!.call(
        batchTensor,
        {}
      ) as Tensor<Rank>[];
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
    const backgroundData = [new Array<number>(this.imageSeg.segSize).fill(0)];
    const explainer = new KernelSHAP(predict, backgroundData, 0.2022);

    // To explain the prediction on this image, we provide the "feature" as
    // showing all segments
    timeit('Explain image', DEBUG);
    const allSegData = new Array<number>(this.imageSeg.segSize).fill(1);
    const shapValues = await explainer.explainOneInstance(
      allSegData,
      NUM_SAMPLES
    );
    timeit('Explain image', DEBUG);

    return shapValues;
  };

  getExplanationImage = (shapValues: number[]) => {
    if (this.imageSeg === null || this.inputImage === null) {
      throw Error('Image is not initialized');
    }

    if (shapValues.length !== this.imageSeg.segSize) {
      throw Error('SHAP value length differs from segment count.');
    }

    // Create a diverging color scale
    const output: number[] = [];

    for (let i = 0; i < this.imageSeg.segData.data.length; i += 4) {
      const segIndex = this.imageSeg.segData.data[i];
      const segColorStr = this.colorScale(this.shapScale(shapValues[segIndex]));
      const segColor = d3.color(segColorStr)!.rgb();
      output.push(segColor.rgb().r);
      output.push(segColor.rgb().g);
      output.push(segColor.rgb().b);
      output.push(255);
    }

    const outputImage = new ImageData(
      new Uint8ClampedArray(output),
      IMG_SRC_LENGTH,
      IMG_SRC_LENGTH
    );

    return outputImage;
  };

  /**
   * Load the initial input image
   */
  loadInputImage = async () => {
    const inputCtx = this.inputCanvas.getContext('2d')!;

    // Create a buffer context to load image
    const hiddenCanvas = createBufferCanvas(IMG_SRC_LENGTH);
    const hiddenCtx = hiddenCanvas.getContext('2d')!;

    const imgFile = `${
      import.meta.env.BASE_URL
    }data/classifier-images/bug-2.jpeg`;
    this.inputImage = await getInputImageData(imgFile);

    // Draw the input image on screen
    hiddenCtx.clearRect(0, 0, IMG_SRC_LENGTH, IMG_SRC_LENGTH);
    hiddenCtx.putImageData(this.inputImage.imageData, 0, 0);

    inputCtx.clearRect(0, 0, IMG_LENGTH, IMG_LENGTH);
    inputCtx.drawImage(
      hiddenCanvas,
      0,
      0,
      IMG_SRC_LENGTH,
      IMG_SRC_LENGTH,
      0,
      0,
      IMG_LENGTH,
      IMG_LENGTH
    );

    // Also put the image in explain wrappers as a background
    for (let c = 0; c < NUM_CLASS; c++) {
      const curInputCtx = this.inputBackCanvases[c].getContext('2d')!;
      curInputCtx.clearRect(0, 0, IMG_LENGTH, IMG_LENGTH);
      curInputCtx.drawImage(
        hiddenCanvas,
        0,
        0,
        IMG_SRC_LENGTH,
        IMG_SRC_LENGTH,
        0,
        0,
        IMG_LENGTH,
        IMG_LENGTH
      );
    }

    // Get the segmentation
    this.createSegmentation(this.inputImage.imageData);

    // Destroy temp canvases
    hiddenCanvas.remove();
  };

  /**
   * Create and draw the segmentation of the input image
   * @param imageData Input image data
   */
  createSegmentation = (imageData: ImageData) => {
    const hiddenCanvas = createBufferCanvas(IMG_SRC_LENGTH);
    const hiddenCtx = hiddenCanvas.getContext('2d')!;

    const segCtx = this.segCanvas.getContext('2d')!;

    const options: SuperPixelOptions = {
      regionSize: 16,
      maxIterations: 10
    };

    const slic = new SLIC(imageData, options);

    // Create a segmentation image
    if (
      slic.result.width !== IMG_SRC_LENGTH ||
      slic.result.height !== IMG_SRC_LENGTH
    ) {
      throw Error(`SLIC result has bad shape ${slic.result}`);
    }

    const segRgba = new Array<number>(
      IMG_SRC_LENGTH * slic.result.height * 4
    ).fill(0);

    // Create a color map
    let indexColorMap: string[] = [];
    d3.schemeTableau10.forEach(d => indexColorMap.push(d));
    d3.schemePastel2.forEach(d => indexColorMap.push(d));
    indexColorMap = d3.shuffler(LCG)(indexColorMap);

    d3.schemeSet1.forEach(d => indexColorMap.push(d));
    d3.schemeSet2.forEach(d => indexColorMap.push(d));
    d3.schemeSet3.forEach(d => indexColorMap.push(d));

    for (let i = 0; i < slic.result.data.length; i += 4) {
      const segLabel = slic.result.data[i];

      // Get the RGB value of this segmentation index
      const segColor = indexColorMap[segLabel % indexColorMap.length];
      const segColorRgb = d3.color(segColor)!.rgb();

      // Add the [R, G, B, A] to the pixel array
      segRgba[i] = segColorRgb.r;
      segRgba[i + 1] = segColorRgb.g;
      segRgba[i + 2] = segColorRgb.b;
      segRgba[i + 3] = 255;
    }

    const imageSegRGB = new ImageData(
      new Uint8ClampedArray(segRgba),
      IMG_SRC_LENGTH,
      IMG_SRC_LENGTH
    );

    this.imageSeg = {
      segData: slic.result,
      segRGBData: imageSegRGB,
      segSize: slic.numSegments
    };

    hiddenCtx.clearRect(0, 0, IMG_SRC_LENGTH, IMG_SRC_LENGTH);
    hiddenCtx.putImageData(imageSegRGB, 0, 0);

    segCtx.drawImage(
      hiddenCanvas,
      0,
      0,
      IMG_SRC_LENGTH,
      IMG_SRC_LENGTH,
      0,
      0,
      IMG_LENGTH,
      IMG_LENGTH
    );

    // Update the segment info
    d3.select(this.component)
      .select('.segment-info')
      .text(`${this.imageSeg.segSize} segments`);

    hiddenCanvas.remove();
  };
}

/**
 * Initialize a canvas element
 * @param component Component HTML element
 * @param canvasName Class name of the canvas element
 * @param canvasLength Length of the canvas element
 * @returns Loaded canvas
 */
const initCanvasElement = (
  component: HTMLElement,
  canvasName: string,
  canvasLength: number
) => {
  const canvas = component.querySelector(`${canvasName}`) as HTMLCanvasElement;
  canvas.width = canvasLength;
  canvas.height = canvasLength;
  return canvas;
};

/**
 * Create a throw away canvas
 * @param length Canvas length (width and height of square canvas)
 * @returns Created canvas
 */
const createBufferCanvas = (length: number) => {
  const bufferCanvas = document.createElement('canvas');
  bufferCanvas.classList.add('hidden-canvas');
  bufferCanvas.width = length;
  bufferCanvas.height = length;
  return bufferCanvas;
};

/**
 * Get the 3D pixel value array of the given image file.
 *
 * @param {string} imgFile File path to the image file
 * @returns A promise with the corresponding 3D array
 */
const getInputImageData = (imgFile: string, normalize = true) => {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'display:none;';
  document.getElementsByTagName('body')[0].appendChild(canvas);
  const context = canvas.getContext('2d')!;

  return new Promise<LoadedImage>((resolve, reject) => {
    const inputImage = new Image();
    inputImage.crossOrigin = 'Anonymous';
    inputImage.src = imgFile;
    let imageData: ImageData;

    inputImage.onload = () => {
      canvas.width = inputImage.width;
      canvas.height = inputImage.height;
      // Resize the input image of the network if it is too large to simply crop
      // the center 64x64 portion in order to still provide a representative
      // input image into the network.
      if (
        inputImage.width > IMG_SRC_LENGTH ||
        inputImage.height > IMG_SRC_LENGTH
      ) {
        // Step 1 - Resize using smaller dimension to scale the image down.
        const resizeCanvas = document.createElement('canvas'),
          resizeContext = resizeCanvas.getContext('2d')!;
        const smallerDimension = Math.min(inputImage.width, inputImage.height);
        const resizeFactor = (IMG_SRC_LENGTH + 1) / smallerDimension;
        resizeCanvas.width = inputImage.width * resizeFactor;
        resizeCanvas.height = inputImage.height * resizeFactor;
        resizeContext.drawImage(
          inputImage,
          0,
          0,
          resizeCanvas.width,
          resizeCanvas.height
        );

        // Step 2 - Flip non-square images horizontally and rotate them 90deg since
        // non-square images are not stored upright.
        if (inputImage.width != inputImage.height) {
          context.translate(resizeCanvas.width, 0);
          context.scale(-1, 1);
          context.translate(resizeCanvas.width / 2, resizeCanvas.height / 2);
          context.rotate((90 * Math.PI) / 180);
        }

        // Step 3 - Draw resized image on original canvas.
        if (inputImage.width != inputImage.height) {
          context.drawImage(
            resizeCanvas,
            -resizeCanvas.width / 2,
            -resizeCanvas.height / 2
          );
        } else {
          context.drawImage(resizeCanvas, 0, 0);
        }
        imageData = context.getImageData(
          0,
          0,
          resizeCanvas.width,
          resizeCanvas.height
        );
      } else {
        context.drawImage(inputImage, 0, 0);
        imageData = context.getImageData(
          0,
          0,
          inputImage.width,
          inputImage.height
        );
      }
      // Get image data and convert it to a 3D array
      const imageArray = imageData.data;
      const imageWidth = imageData.width;
      const imageHeight = imageData.height;

      // Remove this newly created canvas element
      canvas.remove();

      const imageTensor = imageDataTo3DTensor(
        imageArray,
        imageWidth,
        imageHeight,
        normalize
      );

      resolve({ imageData, imageTensor });
    };
    inputImage.onerror = reject;
  });
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
