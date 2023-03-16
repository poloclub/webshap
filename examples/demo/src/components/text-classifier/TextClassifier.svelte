<script lang="ts">
  import d3 from '../../utils/d3-import';
  import { TextClassifier } from './TextClassifier';
  import { onMount } from 'svelte';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';
  import iconBox from '../../imgs/icon-box.svg?raw';
  import iconRefresh from '../../imgs/icon-refresh2.svg?raw';
  import iconToxic from '../../imgs/icon-toxic.svg?raw';
  import iconBenign from '../../imgs/icon-benign.svg?raw';
  import iconCheck from '../../imgs/icon-check.svg?raw';
  import iconUpload from '../../imgs/icon-upload.svg?raw';
  import iconScissor from '../../imgs/icon-scissor.svg?raw';
  import iconOpen from '../../imgs/icon-open.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myTextClassifier: TextClassifier | null = null;

  const defaultInput =
    "Son, you're too young and stupid to tell me that you know enough to claim " +
    "what is vandalism and what isn't.  I suggest you go do your put the computer " +
    'down and do your homework.  You can play on the weekend when school lets out.';

  const benefits = ['Privacy', 'Ubiquity', 'Interactivity'];
  let shownBenefits: string[] = [];

  onMount(() => {
    mounted = true;

    const timeGap = 420;
    for (let i = 0; i < benefits.length; i++) {
      setTimeout(() => {
        shownBenefits.push(benefits[i]);
        shownBenefits = shownBenefits;
      }, 500 + timeGap * i);
    }
  });

  const textClassifierUpdated = () => {
    myTextClassifier = myTextClassifier;
  };

  const predFormatter = d3.format('.2%');

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component) {
      myTextClassifier = new TextClassifier({
        component,
        textClassifierUpdated,
        defaultInput
      });
    }
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './TextClassifier.scss';
</style>

<div class="text-classifier-wrapper" bind:this="{component}">
  <div class="text-classifier">
    <div class="top-section feature">
      <span class="section-name">Input Text</span>
      <div
        class="svg-icon rect-button"
        on:click="{() => {
          if (myTextClassifier) myTextClassifier.sampleClicked();
        }}"
      >
        {@html iconRefresh}
      </div>
    </div>

    <div class="feature-box">
      <div class="intput-wrapper">
        <textarea
          class="input-area"
          autocorrect="off"
          spellcheck="false"
          rows="7"></textarea>
      </div>
    </div>

    <div class="data-model-arrow">
      <div class="background">
        <span class="line-loader hidden"></span>
        <div class="start-rectangle"></div>
        <div class="content-box">
          <div class="line">
            <span class="svg-icon no-pointer">
              {@html iconBox}
            </span>
            <span class="name"> ML Model </span>
          </div>

          <div class="line">
            <span class="model"> XtremeDistil </span>
          </div>

          <div class="loader-container hidden">
            <div class="circle-loader"></div>
            <span class="loader-label">Loading model</span>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="model-explain-arrow">
      <div class="background">
        <span class="line-loader"></span>
        <div class="start-rectangle"></div>
        <div class="content-box">
          <div class="line">
            <span class="svg-icon no-pointer">
              {@html iconWebshap}
            </span>
            <span class="name"> WebSHAP </span>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>

      <div class="benefit-panel">
        {#each benefits as benefit}
          <div class="line" class:hidden="{!shownBenefits.includes(benefit)}">
            <span class="svg-icon no-pointer">{@html iconCheck}</span>
            <span>{benefit}</span>
          </div>
        {/each}
      </div>
    </div>

    <div class="top-section output">
      <span class="section-name">Model Output</span>
      <span class="section-description">Predicted likelihood of toxicity </span>
    </div>

    <div class="output-box">
      <div class="pred-number">
        {myTextClassifier ? predFormatter(myTextClassifier.curPred) : ''}
      </div>

      <div class="pred-bar">
        <svg class="pred-bar-svg"></svg>
      </div>

      <div class="label-container">
        <div class="label placeholder hidden">
          <span class="label-icon svg-icon no-pointer">
            {@html iconBenign}
          </span>
          <span class="label-name"> Benign </span>
        </div>

        <div
          class="label approval"
          class:hidden="{myTextClassifier
            ? myTextClassifier.curPred < 0.5
            : true}"
        >
          <span class="label-icon svg-icon no-pointer">
            {@html iconToxic}
          </span>
          <span class="label-name"> Toxic </span>
        </div>

        <div
          class="label rejection"
          class:hidden="{myTextClassifier
            ? myTextClassifier.curPred >= 0.5
            : true}"
        >
          <span class="label-icon svg-icon no-pointer">
            {@html iconBenign}
          </span>
          <span class="label-name"> Benign </span>
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
            >Each word's contribution to this model's prediction
          </span>
        </div>

        <div class="explain-box">
          <div class="scale-block">
            <span class="shap-label">SHAP Values</span>
            <div class="color-scale-wrapper">
              <svg class="color-scale-svg"></svg>
            </div>
          </div>

          <div class="text-block-container">
            <div class="loader-container hidden">
              <div class="circle-loader"></div>
              <span class="loader-label">Computing SHAP values</span>
            </div>
            <div class="text-block"></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
