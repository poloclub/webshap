<script lang="ts">
  import Tabular from '../tabular/Tabular.svelte';
  import ImageClassifier from '../image-classifier/ImageClassifier.svelte';
  import TextClassifier from '../text-classifier/TextClassifier.svelte';
  import iconGithub from '../../imgs/icon-github.svg?raw';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';
  import iconFile from '../../imgs/icon-file.svg?raw';
  import iconVideo from '../../imgs/icon-youtube.svg?raw';
  import d3 from '../../utils/d3-import';

  let component: HTMLElement | null = null;
  const views = [
    'loan-prediction',
    'image-classification',
    'text-classification'
  ];

  const randomIndex = d3.randomInt(views.length)();
  let view = views[randomIndex];

  // Check url query to change the view
  if (window.location.search !== '') {
    const searchParams = new URLSearchParams(window.location.search);
    if (searchParams.has('model')) {
      const modelName = searchParams.get('model')!;

      switch (modelName) {
        case 'tabular': {
          view = views[0];
          break;
        }

        case 'image': {
          view = views[1];
          break;
        }

        case 'text': {
          view = views[2];
          break;
        }

        default: {
          break;
        }
      }
    }
  }
</script>

<style lang="scss">
  @import './WebShap.scss';
</style>

<div class="webshap-page">
  <!-- <Tooltip {tooltipStore} /> -->

  <div class="description-panel">
    <div class="text-blocks">
      <p>
        <a href="https://github.com/poloclub/webshap">WebSHAP</a> is a JavaScript
        library that can explain any machine learning models on the Web.
      </p>

      <p>
        ✨There is no backend server for the demos. Everything is running in
        your browser. ✨
      </p>

      <ul>
        <li>
          Explainability by <a href="https://github.com/poloclub/webshap"
            >WebSHAP</a
          >
        </li>
        {#if view === views[1]}
          <li>
            Inference by <a href="https://www.tensorflow.org/js/"
              >TensorFlow.js</a
            >
          </li>
        {:else}
          <li>
            Inference by <a href="https://onnxruntime.ai/">ONNX Runtime</a>
          </li>
        {/if}
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

      <div class="title-right"></div>
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

    <div class="app-tabs">
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
        data-text="Text Toxicity Detection">Text Toxicity Detection</button
      >
    </div>
  </div>

  <div class="text-right">
    <div class="icon-container">
      <a target="_blank" href="https://github.com/poloclub/webshap/">
        <div class="svg-icon" title="Open-source code">
          {@html iconGithub}
        </div>
        <span>Code</span>
      </a>

      <a target="_blank" href="https://youtu.be/Dju6ZRMWSAA">
        <div class="svg-icon" title="Research paper">
          {@html iconVideo}
        </div>
        <span>Video</span>
      </a>

      <a target="_blank" href="https://arxiv.org/abs/2303.09545">
        <div class="svg-icon" title="Research paper">
          {@html iconFile}
        </div>
        <span>Paper</span>
      </a>
    </div>
  </div>
</div>
