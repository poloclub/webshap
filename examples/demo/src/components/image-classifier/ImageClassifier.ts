import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import { SLIC } from './segmentation/slic';
import type { SuperPixelOptions } from './segmentation/slic';
import { tick } from 'svelte';
import type {
  LoadedImage,
  Size,
  Padding,
  ImageSegmentation,
  ImageWorkerMessage
} from '../../types/common-types';
import ImageWorker from './image-worker?worker';

const DEBUG = config.debug;
const LCG = d3.randomLcg(0.20230101);

const NUM_CLASS = 4;
const DIVERGE_COLORS = [config.colors['pink-600'], config.colors['blue-700']];

const IMG_SRC_LENGTH = config.layout.imageSrcLength;
const IMG_LENGTH = config.layout.imageLength;

const TOTAL_IMG_NUM = 200;

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

  // Workers
  imageWorker: Worker;

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

    // Initialize the worker
    this.imageWorker = new ImageWorker();
    this.imageWorker.onmessage = (e: MessageEvent<ImageWorkerMessage>) => {
      this.imageWorkerMessageHandler(e);
    };

    // Start to load the model
    const message: ImageWorkerMessage = {
      command: 'startLoadModel',
      payload: {
        url: `${import.meta.env.BASE_URL}models/image-classifier/model.json`
      }
    };
    this.imageWorker.postMessage(message);

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

    // Initialize the input image
    const imagePromise = this.loadRandomInputImage();

    Promise.all([imagePromise]).then(() => {
      // Inference and explain the input image
      this.startPredictInputImage();
      this.startExplainInputImage();
    });
  }

  /**
   * Tell the Web Worker to predict on the current input image
   */
  startPredictInputImage = () => {
    if (this.inputImage === null) {
      throw Error('Data is not initialized.');
    }

    const message: ImageWorkerMessage = {
      command: 'startPredict',
      payload: {
        inputImageData: this.inputImage!.imageData
      }
    };
    this.imageWorker.postMessage(message);
  };

  /**
   * Tell the Web Worker to explain the model on the current input image
   */
  startExplainInputImage = () => {
    if (this.imageSeg === null || this.inputImage === null) {
      throw Error('Data is not initialized.');
    }

    const message: ImageWorkerMessage = {
      command: 'startExplain',
      payload: {
        inputImageData: this.inputImage!.imageData,
        inputImageSeg: this.imageSeg
      }
    };
    this.imageWorker.postMessage(message);
  };

  /**
   * Handling worker messages
   * @param e Message event
   */
  imageWorkerMessageHandler = (e: MessageEvent<ImageWorkerMessage>) => {
    switch (e.data.command) {
      case 'finishLoadModel': {
        break;
      }

      case 'finishPredict': {
        const predictedProb = e.data.payload.predictedProb;
        this.updateBarChart(predictedProb);

        break;
      }

      case 'finishExplain': {
        const shapValues = e.data.payload.shapValues;
        this.updateExplanation(shapValues);
        break;
      }

      default: {
        console.error('Worker: unknown message', e.data.command);
        break;
      }
    }
  };

  /**
   * Update the probability bar chart
   * @param predictedProb New predicted probabilities
   */
  updateBarChart = (predictedProb: Float32Array) => {
    const formatter = d3.format('.4f');

    // Update the bar chart
    for (const [i, output] of this.outputWrappers.entries()) {
      const frontRect = output.select<HTMLElement>('.class-score-front');
      frontRect.style('height', `${this.predProbScale(predictedProb[i])}px`);
      frontRect.select('.class-score-label').text(formatter(predictedProb[i]));
    }
  };

  /**
   * Update visualizations for the explanations
   * @param shapValues Shap values on the input image
   */
  updateExplanation = (shapValues: number[][]) => {
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
   * Load user specified image
   * @param url Image url
   */
  handleCustomImage = async (url: string) => {
    // User gives a valid image URL
    await this.loadInputImage(url);
    // Inference and explain the input image
    this.startPredictInputImage();
    this.startExplainInputImage();
  };

  sampleClicked = async () => {
    await this.loadRandomInputImage();
    // Inference and explain the input image
    this.startPredictInputImage();
    this.startExplainInputImage();
  };

  /**
   * Load the initial input image
   */
  loadRandomInputImage = async () => {
    // Load a random image
    const randomIndex = d3.randomInt(TOTAL_IMG_NUM)() + 1;
    const basename = `${randomIndex}`.padStart(4, '0');
    const imgFile = `${
      import.meta.env.BASE_URL
    }data/classifier-images/${basename}.jpeg`;
    await this.loadInputImage(imgFile);
  };

  /**
   * Load an image from its url
   * @param url Image url
   */
  loadInputImage = async (url: string) => {
    const inputCtx = this.inputCanvas.getContext('2d')!;

    // Create a buffer context to load image
    const hiddenCanvas = createBufferCanvas(IMG_SRC_LENGTH);
    const hiddenCtx = hiddenCanvas.getContext('2d')!;

    this.inputImage = await getInputImageData(url);

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
        const resizeFactor =
          IMG_SRC_LENGTH / Math.min(inputImage.width, inputImage.height);
        resizeCanvas.width = inputImage.width * resizeFactor;
        resizeCanvas.height = inputImage.height * resizeFactor;
        resizeContext.drawImage(
          inputImage,
          0,
          0,
          resizeCanvas.width,
          resizeCanvas.height
        );

        // Step 2 - Draw resized image on original canvas.
        if (inputImage.width != inputImage.height) {
          const sx = (resizeCanvas.width - IMG_SRC_LENGTH) / 2;
          const sy = (resizeCanvas.height - IMG_SRC_LENGTH) / 2;

          context.drawImage(
            resizeCanvas,
            sx,
            sy,
            IMG_SRC_LENGTH,
            IMG_SRC_LENGTH,
            0,
            0,
            IMG_SRC_LENGTH,
            IMG_SRC_LENGTH
          );
        } else {
          context.drawImage(resizeCanvas, 0, 0);
        }
        imageData = context.getImageData(0, 0, IMG_SRC_LENGTH, IMG_SRC_LENGTH);
      } else {
        context.drawImage(inputImage, 0, 0);
        imageData = context.getImageData(
          0,
          0,
          inputImage.width,
          inputImage.height
        );
      }

      // Remove this newly created canvas element
      canvas.remove();

      resolve({ imageData });
    };
    inputImage.onerror = reject;
  });
};
