<script lang="ts">
  import d3 from '../../utils/d3-import';
  import { ImageClassifier } from './ImageClassifier';
  import { onMount } from 'svelte';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';
  import iconBox from '../../imgs/icon-box.svg?raw';
  import iconRefresh from '../../imgs/icon-refresh2.svg?raw';
  import iconCheck from '../../imgs/icon-check.svg?raw';
  import iconCross from '../../imgs/icon-cross.svg?raw';
  import iconOpen from '../../imgs/icon-open.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myImageClassifier: ImageClassifier | null = null;

  // const benefits = ['Privacy', 'Ubiquity', 'Interactivity'];
  // let shownBenefits: string[] = [];
  const classes = [];

  onMount(() => {
    mounted = true;
  });

  const imageClassifierUpdated = () => {
    myImageClassifier = myImageClassifier;
  };

  const predFormatter = d3.format('.2%');

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component) {
      myImageClassifier = new ImageClassifier({
        component,
        imageClassifierUpdated
      });
    }
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './ImageClassifier.scss';
</style>

<div class="image-classifier-wrapper" bind:this="{component}">
  <div class="image-classifier">
    <div class="top-section feature">
      <span class="section-name">Input Data</span>
      <div class="svg-icon rect-button" on:click="{() => {}}">
        {@html iconRefresh}
      </div>
    </div>

    <div class="feature-box">
      <div class="input-image-wrapper image-wrapper">
        <canvas class="input-image image-canvas"></canvas>
      </div>
    </div>

    <div class="data-model-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <div class="line">
              <span class="svg-icon no-pointer">
                {@html iconBox}
              </span>
              <span class="name"> Classifier </span>
            </div>

            <div class="line">
              <span class="model"> TinyVGG </span>
            </div>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="model-explain-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <div class="line">
              <span class="svg-icon no-pointer">
                {@html iconWebshap}
              </span>
              <span class="name"> WebSHAP </span>
            </div>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="top-section output">
      <span class="section-name">Model Output</span>
      <span class="section-description"
        >Likelihood of four image categories
      </span>
    </div>

    <div class="output-box">output</div>

    <div class="data-segment-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <div class="line">
              <span class="svg-icon no-pointer">
                {@html iconBox}
              </span>
              <span class="name"> Segmenter </span>
            </div>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="segment-explain-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <div class="line">
              <span class="svg-icon no-pointer">
                {@html iconWebshap}
              </span>
              <span class="name"> WebSHAP </span>
            </div>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="segment-content">
      <div class="segment-component">
        <div class="top-section segment">
          <span class="section-name">Input Segments</span>
        </div>

        <div class="segment-box">
          <div class="seg-image-wrapper image-wrapper">
            <canvas class="seg-image image-canvas"></canvas>
          </div>

          <div class="segment-info">16 segments</div>
        </div>
      </div>
    </div>

    <div class="explain-content">
      <div class="explain-component">
        <div class="top-section explain">
          <span class="svg-icon no-pointer">
            {@html iconOpen}
          </span>
          <span class="section-name"> Model Explanation</span>
          <span class="section-description"
            >Each segment's contribution to model's prediction
          </span>
        </div>

        <div class="explain-box">
          <div class="bottom-row">
            <div class="explain-wrapper-0 explain-wrapper image-wrapper">
              <canvas class="input-image-back image-canvas"></canvas>
              <canvas class="explain-image image-canvas"></canvas>
            </div>

            <div class="explain-wrapper-1 explain-wrapper image-wrapper">
              <canvas class="input-image-back image-canvas"></canvas>
              <canvas class="explain-image image-canvas"></canvas>
            </div>

            <div class="explain-wrapper-2  explain-wrapper image-wrapper">
              <canvas class="input-image-back image-canvas"></canvas>
              <canvas class="explain-image image-canvas"></canvas>
            </div>

            <div class="explain-wrapper-3  explain-wrapper image-wrapper">
              <canvas class="input-image-back image-canvas"></canvas>
              <canvas class="explain-image image-canvas"></canvas>
            </div>
          </div>

          <div class="color-scale-wrapper">
            <svg class="color-scale-svg"></svg>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>