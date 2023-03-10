<script lang="ts">
  import Tabular from '../tabular/Tabular.svelte';
  import ImageClassifier from '../image-classifier/ImageClassifier.svelte';
  import TextClassifier from '../text-classifier/TextClassifier.svelte';
  import iconGithub from '../../imgs/icon-github.svg?raw';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';

  let component: HTMLElement | null = null;
  const views = [
    'loan-prediction',
    'image-classification',
    'text-classification'
  ];
  let view = views[1];

  // Initialize the stores to pass to child components
  // const tooltipStore = getTooltipStore();
</script>

<style lang="scss">
  @import './WebShap.scss';
</style>

<div class="webshap-page">
  <!-- <Tooltip {tooltipStore} /> -->

  <div class="description-panel">
    <div class="text-blocks">
      <p>
        <a href="https://github.com/poloclub/webshap">WebSHAP</a> is an open-source
        library that can explain any machine learning models on the Web.
      </p>

      <p>
        ✨There is no backend server for this demo. Everything is running in
        your browser. ✨
      </p>

      <ul>
        <li>
          Explainability by <a href="https://github.com/poloclub/webshap"
            >WebSHAP</a
          >
        </li>
        <li>
          Inference by <a href="https://onnxruntime.ai/">ONNX Runtime</a>
        </li>
        <li>
          Acceleration by <a href="https://www.tensorflow.org/js"
            >TensorFlow.js</a
          >
        </li>
      </ul>
    </div>
  </div>

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
      <div class="main-app-container" class:hidden="{view !== views[0]}">
        <Tabular />
      </div>

      <div class="main-app-container" class:hidden="{view !== views[1]}">
        <ImageClassifier />
      </div>

      <div class="main-app-container" class:hidden="{view !== views[2]}">
        <TextClassifier />
      </div>
    </div>
  </div>
</div>
