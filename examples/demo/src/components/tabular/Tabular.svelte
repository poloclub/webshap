<script lang="ts">
  import { Tabular } from './Tabular';
  import { onMount } from 'svelte';
  import iconGear from '../../imgs/icon-gear.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myTabular: Tabular | null = null;

  onMount(() => {
    mounted = true;
  });

  const tabularUpdated = () => {
    myTabular = myTabular;
  };

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component) {
      myTabular = new Tabular({ component, tabularUpdated });
    }
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './Tabular.scss';
</style>

<div class="tabular-wrapper" bind:this={component}>
  <div class="tabular">
    <span class="message-board">
      {myTabular ? myTabular.message : ''}
    </span>
  </div>
</div>
