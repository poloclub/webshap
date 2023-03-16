import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
import { SLIC } from './segmentation/slic';
import type { SuperPixelOptions } from './segmentation/slic';
import { tick } from 'svelte';
import type {
  TabularData,
  TabularContFeature,
  TabularCatFeature,
  Size,
  Padding,
  SHAPRow
} from '../../types/common-types';
import { KernelSHAP } from 'webshap';
import { round, timeit, downloadJSON } from '../../utils/utils';
import { getLatoTextWidth } from '../../utils/text-width';

const DEBUG = config.debug;
const LCG = d3.randomLcg(0.20230101);
const RANDOM_INT = d3.randomInt.source(LCG);
const RANDOM_UNIFORM = d3.randomUniform.source(LCG);

const IMG_SRC_LENGTH = 64;
const IMG_LENGTH = 200;

/**
 * Class for the Image Classifier WebSHAP demo
 */

export class ImageClassifier {
  component: HTMLElement;
  imageClassifierUpdated: () => void;

  inputCanvas: HTMLCanvasElement;
  segCanvas: HTMLCanvasElement;

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
      'input-image',
      IMG_LENGTH
    );
    this.segCanvas = initCanvasElement(this.component, 'seg-image', IMG_LENGTH);

    this.loadInputImage();
  }

  loadInputImage = async () => {
    const inputCtx = this.inputCanvas.getContext('2d')!;
    const segCtx = this.segCanvas.getContext('2d')!;

    // Create a buffer context to load image
    const hiddenCanvas = createBufferCanvas(IMG_SRC_LENGTH);
    const hiddenCtx = hiddenCanvas.getContext('2d')!;

    const img = await loadImageAsync(
      `${import.meta.env.BASE_URL}data/classifier-images/bug-1.jpeg`
    );

    // Get image pixel array
    hiddenCtx.drawImage(img, 0, 0);
    const imageData = hiddenCtx.getImageData(
      0,
      0,
      IMG_SRC_LENGTH,
      IMG_SRC_LENGTH
    );
    const rgbs = rgbaToRgb(imageData.data);

    inputCtx.drawImage(
      img,
      0,
      0,
      IMG_SRC_LENGTH,
      IMG_SRC_LENGTH,
      0,
      0,
      IMG_LENGTH,
      IMG_LENGTH
    );

    // Get the segmentation
    const options: SuperPixelOptions = {
      regionSize: 16,
      minRegionSize: 100,
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

    const segImageData = new ImageData(
      new Uint8ClampedArray(segRgba),
      IMG_SRC_LENGTH,
      slic.result.height
    );

    hiddenCtx.clearRect(0, 0, IMG_SRC_LENGTH, IMG_SRC_LENGTH);
    hiddenCtx.putImageData(segImageData, 0, 0);

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

    // Destroy temp canvases
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
  const canvas = component.querySelector(`.${canvasName}`) as HTMLCanvasElement;
  canvas.width = canvasLength;
  canvas.height = canvasLength;
  return canvas;
};

/**
 * Async wrapper for the image load event
 * @param url Image source url
 * @returns Promise of the loaded image
 */
const loadImageAsync = (url: string) => {
  return new Promise<HTMLImageElement>(resolve => {
    const image = new Image();
    image.onload = () => {
      resolve(image);
    };
    image.src = url;
  });
};

/**
 * Convert RGBA data array to RGB data array
 * @param data Image data read from canvas
 * @returns RGB pixel values
 */
const rgbaToRgb = (data: Uint8ClampedArray) => {
  const results: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i % 4 !== 3) {
      results.push(data[i]);
    }
  }

  return results;
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
