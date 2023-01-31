<script lang="ts">
  import { onMount } from 'svelte';
  import iconGear from '../../imgs/icon-gear.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;

  onMount(() => {
    mounted = true;
  });

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component) {
      // Pass
    }
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './Tabular.scss';
</style>

<div class="tabular-wrapper" bind:this={component}>
  <div class="tabular" />
</div>
