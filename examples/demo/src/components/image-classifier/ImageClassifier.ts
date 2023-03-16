import d3 from '../../utils/d3-import';
import { config } from '../../config/config';
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

const IMAGE_SOURCE_LENGTH = 64;
const IMAGE_LENGTH = 200;

/**
 * Class for the Image Classifier WebSHAP demo
 */

export class ImageClassifier {
  component: HTMLElement;
  imageClassifierUpdated: () => void;

  hiddenCanvas: HTMLCanvasElement;
  inputCanvas: HTMLCanvasElement;

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
    this.hiddenCanvas = initCanvasElement(
      this.component,
      'hidden-canvas',
      IMAGE_LENGTH
    );
    this.inputCanvas = initCanvasElement(
      this.component,
      'input-image',
      IMAGE_LENGTH
    );

    this.loadInputImage();
  }

  loadInputImage = async () => {
    const hiddenCtx = this.hiddenCanvas.getContext('2d')!;
    const inputCtx = this.inputCanvas.getContext('2d')!;

    const img = await loadImageAsync(
      `${import.meta.env.BASE_URL}data/classifier-images/bug-1.jpeg`
    );

    // Get image pixel array
    hiddenCtx.drawImage(img, 0, 0);
    const pixelArray = hiddenCtx.getImageData(
      0,
      0,
      IMAGE_SOURCE_LENGTH,
      IMAGE_SOURCE_LENGTH
    );
    const rgbs = rgbaToRgb(pixelArray.data);

    inputCtx.drawImage(
      img,
      0,
      0,
      IMAGE_SOURCE_LENGTH,
      IMAGE_SOURCE_LENGTH,
      0,
      0,
      IMAGE_LENGTH,
      IMAGE_LENGTH
    );
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
