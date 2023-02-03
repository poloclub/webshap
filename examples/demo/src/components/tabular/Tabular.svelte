<script lang="ts">
  import d3 from '../../utils/d3-import';
  import { Tabular } from './Tabular';
  import { onMount } from 'svelte';
  import iconBox from '../../imgs/icon-box.svg?raw';
  import iconRefresh from '../../imgs/icon-refresh2.svg?raw';
  import iconCheck from '../../imgs/icon-check.svg?raw';
  import iconCross from '../../imgs/icon-cross.svg?raw';
  import iconWebshap from '../../imgs/icon-webshap.svg?raw';

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

  const predFormatter = d3.format('.2%');

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

<div class="tabular-wrapper" bind:this="{component}">
  <div class="tabular">
    <div class="top-section feature">
      <span class="section-name">Input Data</span>
      <span class="section-description"
        >Loan applicant #{String(myTabular ? myTabular.curIndex : 0).padStart(
          3,
          '0'
        )} information</span
      >
      <div
        class="svg-icon"
        on:click="{() => {
          if (myTabular) myTabular.sampleClicked();
        }}"
      >
        {@html iconRefresh}
      </div>
    </div>

    <div class="feature-box">
      <div class="feature-section">
        {#if myTabular && myTabular.contFeatures}
          <span class="feature-header cont">Continuous Features</span>
          <div class="content-cont">
            {#each [...myTabular.contFeatures.values()] as item}
              <div class="input-wrapper">
                <span class="name">{item.displayName}</span>
                <input
                  class="feature-input"
                  type="number"
                  step="{item.requiresInt ? 1 : 0.1}"
                  on:change="{() => {
                    if (myTabular) myTabular.inputChanged();
                  }}"
                  bind:value="{item.value}"
                />
              </div>
            {/each}
          </div>
        {/if}
      </div>

      <div class="feature-section">
        {#if myTabular && myTabular.contFeatures}
          <span class="feature-header cat">Categorical Features</span>
          <div class="content-cat">
            {#each [...myTabular.catFeatures.values()] as item}
              <div class="input-wrapper">
                <span class="name">{item.displayName}</span>
                <select
                  class="feature-select"
                  bind:value="{item.value}"
                  on:change="{() => {
                    if (myTabular) myTabular.inputChanged();
                  }}"
                >
                  {#each item.allLevels as level}
                    <option value="{level.level}">{level.displayName}</option>
                  {/each}
                </select>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </div>

    <div class="data-model-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <span class="svg-icon no-pointer">
              {@html iconBox}
            </span>
            <span class="name"> XGBoost </span>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="model-explain-arrow">
      <div class="background">
        <div class="start-rectangle">
          <div class="content-box">
            <!-- <div class="control-panel">
              <select>
                <option>Background (median)</option>
              </select>
              <select>
                <option>Samples (auto)</option>
              </select>
            </div> -->
            <span class="svg-icon no-pointer">
              {@html iconWebshap}
            </span>
            <span class="name"> WebSHAP </span>
          </div>
        </div>
        <div class="end-triangle"></div>
      </div>
    </div>

    <div class="top-section output">
      <span class="section-name">Model Output</span>
      <span class="section-description">Timely repayment likelihood </span>
    </div>

    <div class="output-box">
      <div class="pred-number">
        {myTabular ? predFormatter(myTabular.curPred) : ''}
      </div>

      <div class="pred-bar">
        <svg class="pred-bar-svg"></svg>
      </div>

      <div class="label-container">
        <div class="label placeholder hidden">
          <span class="label-icon svg-icon no-pointer">
            {@html iconCross}
          </span>
          <span class="label-name"> Rejection </span>
        </div>

        <div
          class="label approval"
          class:hidden="{myTabular ? myTabular.curPred < 0.5 : true}"
        >
          <span class="label-icon svg-icon no-pointer">
            {@html iconCheck}
          </span>
          <span class="label-name"> Approval </span>
        </div>

        <div
          class="label rejection"
          class:hidden="{myTabular ? myTabular.curPred >= 0.5 : true}"
        >
          <span class="label-icon svg-icon no-pointer">
            {@html iconCross}
          </span>
          <span class="label-name"> Rejection </span>
        </div>
      </div>
    </div>

    <div class="top-section explain">
      <span class="section-name">Prediction Explanation</span>
      <span class="section-description"
        >Features' contribution to the prediction
      </span>
    </div>

    <div class="explain-content">
      <div class="arrow"></div>
      <div class="explain-box">
        <div class="header">
          <span>Top 10 Important Features and</span>
          <span class="shap-label">Their SHAP Values</span>
        </div>
        <svg class="shap-svg"></svg>
      </div>
    </div>
  </div>
</div>
