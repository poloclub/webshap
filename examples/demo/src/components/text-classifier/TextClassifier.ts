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
import {
  round,
  timeit,
  downloadJSON,
  haveContrast,
  getContrastRatio
} from '../../utils/utils';
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
  inputText: string;

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
    this.inputText = defaultInput;

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
    const rectWidth = 15;
    const svgPadding: Padding = {
      top: 8,
      left: 0,
      right: 0,
      bottom: 8
    };
    const axisWidth = 40;

    this.colorScaleSVG.attr('transform', `translate(${-rectWidth / 2}, 0)`);

    const contentGroup = this.colorScaleSVG
      .append('g')
      .attr('class', 'content');

    const axisGroup = contentGroup
      .append('g')
      .attr('class', 'axis-group')
      .attr('transform', `translate(${axisWidth + 1}, ${svgPadding.top + 1})`);

    contentGroup
      .append('rect')
      .attr('class', 'scale-rect')
      .attr('x', axisWidth)
      .attr('y', svgPadding.top)
      .attr('width', rectWidth)
      .attr('height', svgSize.height - svgPadding.top - svgPadding.bottom)
      .attr('fill', 'url(#scale-gradient-text)');

    // Fill the rect with a diverging color gradient
    const gradients = this.colorScaleSVG
      .append('defs')
      .append('linearGradient')
      .attr('gradientTransform', 'rotate(-90 0.5 0.5)')
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
      svgSize.height - svgPadding.top - svgPadding.bottom - 2,
      0
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

  updateTextBlock = () => {
    let words = [];
    let curWord = '';

    // eslint-disable-next-line quotes
    const punctuations = new Set([',', '.', '/', '?', '!', '@', "'", '"', '`']);

    for (let i = 0; i < this.inputText.length; i++) {
      const curChar = this.inputText[i];
      if (curChar === ' ') {
        if (curWord !== '') {
          words.push(curWord);
        }
        curWord = '';
        // eslint-disable-next-line quotes
      } else if (curChar === "'") {
        if (curWord !== '') {
          words.push(curWord);
          curWord = '';
        }
        // eslint-disable-next-line quotes
        curWord = "'";
      } else if (punctuations.has(curChar)) {
        if (curWord !== '') {
          words.push(curWord);
          words.push(curChar);
          curWord = '';
        } else {
          words.push(curChar);
        }
      } else {
        curWord += curChar.toLowerCase();
      }
    }

    words = [
      'son',
      ', ',
      'you',
      "'",
      're ',
      'too ',
      'young ',
      'and ',
      'stupid ',
      'to ',
      'tell ',
      'me ',
      'that ',
      'you ',
      'know ',
      'enough ',
      'to ',
      'claim ',
      'what ',
      'is ',
      'van',
      'dal',
      'ism ',
      'and ',
      'what ',
      'isn',
      "'",
      't',
      '.  ',
      'I ',
      'suggest ',
      'you ',
      'go ',
      'do ',
      'your ',
      'put ',
      'the ',
      'computer ',
      'down ',
      'and ',
      'do ',
      'your ',
      'homework',
      '.  ',
      'You ',
      'can ',
      'play ',
      'on ',
      'the ',
      'weekend ',
      'when ',
      'school ',
      'lets ',
      'out',
      '.'
    ];

    const shapValues = [
      0.01092504, 0.00455374, -0.00485153, -0.00448904, -0.00224217,
      -0.04614502, -0.00383218, -0.00510751, -0.06787664, -0.00415155,
      -0.00535764, -0.00535764, -0.00371679, -0.00371679, -0.00326535,
      -0.00326535, -0.00260804, -0.00260804, -0.00618124, -0.00205904,
      -0.03055499, -0.01819205, -0.01296856, -0.0064224, -0.01313715,
      -0.01091389, -0.00870228, -0.00464682, -0.00877311, -0.00895331,
      -0.00924679, 0.00626233, 0.00452801, -0.00864476, -0.00701219,
      -0.02690166, -0.01169196, -0.04226858, -0.07019774, 0.01310815,
      -0.01112492, -0.00257022, 0.00244315, 0.00703637, 0.01107377, 0.0118604,
      0.00137123, 0.0015346, -0.00076053, -0.00076053, 0.00096916, -0.00264306,
      -0.04983622, -0.04860111, -0.00971938
    ];

    // Need to get the min and max of shap values across all classes
    const shapRange: [number, number] = [Infinity, -Infinity];
    for (const value of shapValues) {
      if (value < shapRange[0]) {
        shapRange[0] = value;
      }
      if (value > shapRange[1]) {
        shapRange[1] = value;
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

    const axisGroup = this.colorScaleSVG.select<SVGGElement>('g.axis-group');
    this.colorLegendAxis = d3
      .axisLeft(this.shapLengthScale)
      .tickValues([
        this.shapLengthScale.domain()[0],
        0,
        this.shapLengthScale.domain()[1]
      ])
      .tickFormat(d3.format('.2f'));
    axisGroup.call(this.colorLegendAxis);

    // Add the words to the text block
    const textBlock = d3
      .select(this.component)
      .select<HTMLElement>('.text-block');
    textBlock.selectAll('*').remove();

    for (const [i, word] of words.entries()) {
      const backColorStr = this.colorScale(this.shapScale(-shapValues[i]));
      const backColor = d3.color(backColorStr)!.rgb();
      const frontColor = d3.color(config.colors['gray-900'])!.rgb();

      textBlock
        .append('span')
        .attr('class', 'text-word')
        .text(word)
        .style('background-color', backColorStr)
        .classed(
          'dark-background',
          getContrastRatio(
            [backColor.r, backColor.g, backColor.b],
            [frontColor.r, frontColor.g, frontColor.b]
          ) >
            getContrastRatio(
              [backColor.r, backColor.g, backColor.b],
              [255, 255, 255]
            )
        );
    }
    console.log(words);
  };
}
