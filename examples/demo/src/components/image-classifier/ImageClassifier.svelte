<script lang="ts">
  import d3 from '../../utils/d3-import';
  import { ImageClassifier } from './ImageClassifier';
  import { onMount } from 'svelte';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myImageClassifier: ImageClassifier | null = null;

  // const benefits = ['Privacy', 'Ubiquity', 'Interactivity'];
  // let shownBenefits: string[] = [];

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
    <div class="top-row">
      <div class="input-image-wrapper image-wrapper">
        <canvas class="input-image image-canvas"></canvas>
      </div>
    </div>

    <div class="bottom-row">
      <div class="seg-image-wrapper image-wrapper">
        <canvas class="seg-image image-canvas"></canvas>
      </div>

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
