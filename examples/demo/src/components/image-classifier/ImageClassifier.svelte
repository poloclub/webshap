<script lang="ts">
  import d3 from '../../utils/d3-import';
  import { ImageClassifier } from './ImageClassifier';
  import { onMount } from 'svelte';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';
  import iconBox from '../../imgs/icon-box.svg?raw';
  import iconRefresh from '../../imgs/icon-refresh2.svg?raw';
  import iconCheck from '../../imgs/icon-check.svg?raw';
  import iconCross from '../../imgs/icon-cross.svg?raw';
  import iconUpload from '../../imgs/icon-upload.svg?raw';
  import iconScissor from '../../imgs/icon-scissor.svg?raw';
  import iconOpen from '../../imgs/icon-open.svg?raw';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myImageClassifier: ImageClassifier | null = null;
  let dialogElement: HTMLDialogElement | null = null;
  let valiImg: HTMLImageElement;
  let files;
  let usingURL = true;
  let inputValue = '';
  let showError = false;

  // https://i.imgur.com/auioQcG.png

  const benefits = ['Privacy', 'Ubiquity', 'Interactivity'];
  let shownBenefits: string[] = [];
  const classes = ['🐞 Ladybug', '☕️ Espresso', '🍊 Orange', '🚙 Sports Car'];

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

  const imageClassifierUpdated = () => {
    myImageClassifier = myImageClassifier;
  };

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

  const imageUpload = () => {
    usingURL = false;
    const reader = new FileReader();
    reader.onload = event => {
      valiImg.src = event.target.result;
    };
    reader.readAsDataURL(files[0]);
  };

  const loadCallback = () => {
    // The URL is valid, but we are not sure if loading it to canvas would be
    // blocked by crossOrigin setting. Try it here before dispatch to parent.
    // https://stackoverflow.com/questions/13674835/canvas-tainted-by-cross-origin-data
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.width = valiImg.width;
    canvas.height = valiImg.height;
    context.drawImage(valiImg, 0, 0);

    try {
      context.getImageData(0, 0, valiImg.width, valiImg.height);
      // If the foreign image does support CORS -> use this image
      // dispatch to parent component to use the input image
      myImageClassifier?.handleCustomImage(valiImg.src);
      dialogElement?.close();
    } catch (err) {
      // If the foreign image does not support CORS -> use this image
      showError = true;
    }
  };

  const addClicked = () => {
    // Validate the input URL
    showError = false;
    valiImg.crossOrigin = 'Anonymous';
    valiImg.src = inputValue;
  };

  const errorCallback = () => {
    // The URL is invalid, show an error message on the UI
    showError = true;
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './ImageClassifier.scss';
</style>

<div class="image-classifier-wrapper" bind:this="{component}">
  <dialog id="image-dialog" bind:this="{dialogElement}">
    <div class="header">Add Input Image</div>

    <div class="row-block">
      <input
        class="image-url-input"
        type="url"
        placeholder="Paste URL of image..."
        bind:value="{inputValue}"
      />

      <span>or</span>

      <div class="file">
        <label class="file-label">
          <input
            class="file-input"
            type="file"
            name="image"
            accept=".png,.jpeg,.tiff,.jpg,.png"
            bind:files="{files}"
            on:change="{imageUpload}"
          />
          <div class="button upload-button">
            <div class="svg-icon no-pointer">
              {@html iconUpload}
            </div>
            Upload
          </div>
        </label>
      </div>
    </div>

    <div class="button-block">
      <span class="error-message" class:hidden="{!showError}"
        >Failed to load, try a different image.</span
      >
      <button
        class="add-button"
        on:click="{() => {
          addClicked();
        }}">Add</button
      >

      <button
        class="close-button"
        on:click="{() => {
          dialogElement?.close();
        }}">Close</button
      >
    </div>
  </dialog>

  <div class="image-classifier">
    <div class="top-section feature">
      <span class="section-name">Input Image</span>
      <div
        class="svg-icon rect-button"
        on:click="{() => {
          if (myImageClassifier) myImageClassifier.sampleClicked();
        }}"
      >
        {@html iconRefresh}
      </div>
      <div
        class="svg-icon rect-button"
        on:click="{() => {
          dialogElement?.showModal();
        }}"
      >
        {@html iconUpload}
      </div>
    </div>

    <div class="feature-box">
      <div class="input-image-wrapper image-wrapper">
        <canvas class="input-image image-canvas"></canvas>
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
            <span class="model"> TinyVGG </span>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>

      <div class="arrow-text">
        See <a href="https://poloclub.github.io/cnn-explainer" target="_blank"
          >CNN Explainer</a
        > to learn more about TinyVGG
      </div>
    </div>

    <div class="model-explain-arrow">
      <div class="background">
        <span class="line-loader hidden"></span>
        <div class="start-rectangle"></div>
        <div class="content-box">
          <div class="line">
            <span class="svg-icon no-pointer">
              {@html iconWebshap}
            </span>
            <span class="name"> WebSHAP </span>

            <div class="loader-container hidden">
              <div class="circle-loader"></div>
            </div>
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
      <span class="section-description"
        >Predicted likelihoods of four image categories
      </span>
    </div>

    <div class="output-box">
      {#each classes as className, i}
        <div class="{`output-wrapper-${i} output-wrapper`}">
          <div class="class-score">
            <div class="class-score-back"></div>
            <div class="class-score-front">
              <span class="class-score-label">0.0000</span>
            </div>
          </div>
          <span class="class-label">{className}</span>
        </div>
      {/each}
    </div>

    <div class="data-segment-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <div class="line">
              <span class="svg-icon no-pointer">
                {@html iconScissor}
              </span>
              <span class="name"> Segmenter </span>
            </div>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="segment-explain-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <div class="line">
              <span class="svg-icon no-pointer">
                {@html iconWebshap}
              </span>
              <span class="name"> WebSHAP </span>
            </div>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>

      <div class="arrow-text">
        For efficiency, compute SHAP values for segments, not individual pixels
      </div>
    </div>

    <div class="segment-content">
      <div class="segment-component">
        <div class="top-section segment">
          <span class="section-name">Image Segments</span>
        </div>

        <div class="segment-box">
          <div class="seg-image-wrapper image-wrapper">
            <canvas class="seg-image image-canvas"></canvas>
          </div>

          <div class="segment-info">10 segments</div>
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
            >Each segment's contribution to model's prediction
          </span>
        </div>

        <div class="explain-box">
          <div class="image-row">
            {#each classes as className, i}
              <div
                class="{`explain-wrapper-${i} explain-wrapper image-wrapper`}"
              >
                <canvas class="input-image-back image-canvas"></canvas>
                <canvas class="explain-image image-canvas"></canvas>
                <span class="class-label">{className}</span>
              </div>
            {/each}
          </div>

          <div class="bottom-row">
            <span class="shap-label">SHAP Values</span>
            <div class="color-scale-wrapper">
              <svg class="color-scale-svg"></svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- An invisible image to check if the user input URL is valid -->
  <img
    style="display: none"
    id="vali-image"
    alt="hidden image"
    bind:this="{valiImg}"
    on:error="{errorCallback}"
    on:load="{loadCallback}"
  />
</div>
