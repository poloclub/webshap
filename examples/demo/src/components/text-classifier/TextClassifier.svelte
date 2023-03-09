<script lang="ts">
  import d3 from '../../utils/d3-import';
  import { TextClassifier } from './TextClassifier';
  import { onMount } from 'svelte';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myTextClassifier: TextClassifier | null = null;

  // const benefits = ['Privacy', 'Ubiquity', 'Interactivity'];
  // let shownBenefits: string[] = [];

  onMount(() => {
    mounted = true;
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
        textClassifierUpdated
      });
    }
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './TextClassifier.scss';
</style>

<div class="text-classifier-wrapper" bind:this="{component}">
  <div class="text-classifier">Text</div>
</div>
