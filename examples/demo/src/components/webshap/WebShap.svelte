<script lang="ts">
  import Tabular from '../tabular/Tabular.svelte';
  import iconGithub from '../../imgs/icon-github.svg?raw';

  let component: HTMLElement | null = null;
  let view = 'loan-prediction';

  // Initialize the stores to pass to child components
  // const tooltipStore = getTooltipStore();
</script>

<style lang="scss">
  @import './WebShap.scss';
</style>

<div class="diffusiondbvis-page">
  <!-- <Tooltip {tooltipStore} /> -->

  <div class="app-wrapper">
    <div class="app-title">
      <div class="title-left">
        <!-- <div class="app-icon">
          {@html logoDiffusionDB}
        </div> -->
        <span class="app-name"> WebSHAP Demo </span>
      </div>

      <div class="title-right">
        <a class="title-link" href="https://github.com/anonacl/diffusiondb">
          {@html iconGithub}
        </a>
      </div>
    </div>

    <div class="main-app" bind:this={component}>
      <div class="main-app-container" class:hidden={view !== 'loan-prediction'}>
        <Tabular />
      </div>

      <div
        class="main-app-container"
        class:hidden={view !== 'image-classification'}
      />

      <div
        class="main-app-container"
        class:hidden={view !== 'text-classification'}
      />
    </div>

    <div class="app-tabs">
      <button
        class="tab"
        class:selected={view === 'loan-prediction'}
        on:click={() => {
          view = 'loan-prediction';
        }}
        data-text="Loan Approval Prediction">Loan Approval Prediction</button
      >

      <span class="splitter" />

      <button
        class="tab"
        class:selected={view === 'image-classification'}
        on:click={() => {
          view = 'image-classification';
        }}
        data-text="Image Classification">Image Classification</button
      >

      <span class="splitter" />

      <button
        class="tab"
        class:selected={view === 'text-classification'}
        on:click={() => {
          view = 'text-classification';
        }}
        data-text="Text Classification">Text Classification</button
      >
    </div>
  </div>
</div>
