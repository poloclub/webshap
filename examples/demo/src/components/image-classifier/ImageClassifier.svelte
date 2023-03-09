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
  <div class="image-classifier">Image</div>
</div>
