<script lang="ts">
  import Tabular from '../tabular/Tabular.svelte';
  import iconGithub from '../../imgs/icon-github.svg?raw';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';

  let component: HTMLElement | null = null;
  let view = 'loan-prediction';

  // Initialize the stores to pass to child components
  // const tooltipStore = getTooltipStore();
</script>

<style lang="scss">
  @import './WebShap.scss';
</style>

<div class="webshap-page">
  <!-- <Tooltip {tooltipStore} /> -->

  <div class="app-wrapper">
    <div class="app-title">
      <div class="title-left">
        <div class="app-icon">
          {@html iconWebshap}
        </div>
        <div class="app-info">
          <span class="app-name"> WebSHAP </span>
          <span class="app-tagline"
            >Explain any machine learning models in your browser!</span
          >
        </div>
      </div>

      <div class="title-right">
        <a class="title-link" href="https://github.com/poloclub/webshap">
          {@html iconGithub}
        </a>
      </div>
    </div>

    <div class="main-app" bind:this="{component}">
      <div
        class="main-app-container"
        class:hidden="{view !== 'loan-prediction'}"
      >
        <Tabular />
      </div>

      <div
        class="main-app-container"
        class:hidden="{view !== 'image-classification'}"
      ></div>

      <div
        class="main-app-container"
        class:hidden="{view !== 'text-classification'}"
      ></div>
    </div>

    <!-- <div class="app-tabs">
      <button
        class="tab"
        class:selected="{view === 'loan-prediction'}"
        on:click="{() => {
          view = 'loan-prediction';
        }}"
        data-text="Loan Approval Prediction">Loan Approval Prediction</button
      >

      <span class="splitter"></span>

      <button
        class="tab"
        class:selected="{view === 'image-classification'}"
        on:click="{() => {
          view = 'image-classification';
        }}"
        data-text="Image Classification">Image Classification</button
      >

      <span class="splitter"></span>

      <button
        class="tab"
        class:selected="{view === 'text-classification'}"
        on:click="{() => {
          view = 'text-classification';
        }}"
        data-text="Text Classification">Text Classification</button
      >
    </div> -->
  </div>
</div>
