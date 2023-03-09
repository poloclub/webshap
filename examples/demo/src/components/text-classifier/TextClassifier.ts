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

// SVG constants
const GAP = 20;

/**
 * Class for the Text Classifier WebSHAP demo
 */

export class TextClassifier {
  component: HTMLElement;
  textClassifierUpdated: () => void;

  /**
   * @param args Named parameters
   * @param args.component The component
   * @param args.textClassifierUpdated A function to trigger updates
   */
  constructor({
    component,
    textClassifierUpdated
  }: {
    component: HTMLElement;
    textClassifierUpdated: () => void;
  }) {
    this.component = component;
    this.textClassifierUpdated = textClassifierUpdated;
  }
}
