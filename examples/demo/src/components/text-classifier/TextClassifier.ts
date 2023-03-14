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
const DIVERGE_COLORS = [config.colors['pink-600'], config.colors['blue-700']];
const LCG = d3.randomLcg(0.20230101);
const RANDOM_INT = d3.randomInt.source(LCG);
const RANDOM_UNIFORM = d3.randomUniform.source(LCG);

/**
 * Class for the Text Classifier WebSHAP demo
 */

export class TextClassifier {
  component: HTMLElement;
  textClassifierUpdated: () => void;

  // Visualization
  colorScale: (t: number) => string;
  shapLengthScale: d3.ScaleLinear<number, number, never>;
  shapScale: d3.ScaleLinear<number, number, never>;
  colorLegendAxis: d3.Axis<d3.NumberValue>;

  // SVG elements
  colorScaleSVG: d3.Selection<HTMLElement, unknown, null, undefined>;

  /**
   * @param args Named parameters
   * @param args.component The component
   * @param args.textClassifierUpdated A function to trigger updates
   */
  constructor({
    component,
    textClassifierUpdated,
    defaultInput
  }: {
    component: HTMLElement;
    textClassifierUpdated: () => void;
    defaultInput: string;
  }) {
    this.component = component;
    this.textClassifierUpdated = textClassifierUpdated;

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

    // Initialize the text block
    this.updateTextBlock();
  }

  initSVGs = () => {
    // Initialize the color scale svg
    const bbox = this.colorScaleSVG.node()!.getBoundingClientRect();
    const svgSize: Size = {
      width: bbox.width,
      height: bbox.height
    };
    const rectWidth = 10;

    const svgPadding: Padding = {
      top: 8,
      left: svgSize.width / 2 + rectWidth / 2,
      right: 0,
      bottom: 8
    };

    const contentGroup = this.colorScaleSVG
      .append('g')
      .attr('class', 'content');

    const axisGroup = contentGroup
      .append('g')
      .attr('class', 'axis-group')
      .attr(
        'transform',
        `translate(${svgPadding.left + 1}, ${svgPadding.top})`
      );

    contentGroup
      .append('rect')
      .attr('class', 'scale-rect')
      .attr('x', svgPadding.left)
      .attr('y', svgPadding.top)
      .attr('width', rectWidth)
      .attr('height', svgSize.height - svgPadding.top - svgPadding.bottom)
      .attr('fill', 'url(#scale-gradient-text)');

    // Fill the rect with a diverging color gradient
    const gradients = this.colorScaleSVG
      .append('defs')
      .append('linearGradient')
      .attr('gradientTransform', 'rotate(90)')
      .attr('id', 'scale-gradient-text');

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

    // Add a legend on the left of the color scale
    this.shapLengthScale.range([
      0,
      svgSize.height - svgPadding.top - svgPadding.bottom - 2
    ]);

    this.colorLegendAxis = d3
      .axisLeft(this.shapLengthScale)
      .tickValues([
        this.shapLengthScale.domain()[0],
        0,
        this.shapLengthScale.domain()[1]
      ])
      .tickFormat(d3.format('.2f'));
    axisGroup.call(this.colorLegendAxis);
    axisGroup.attr('font-size', null);
  };
}
