<script lang="ts">
  import { Tabular } from './Tabular';
  import { onMount } from 'svelte';
  import iconGear from '../../imgs/icon-gear.svg?raw';
  import iconRefresh from '../../imgs/icon-refresh2.svg?raw';

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

    <div class="data-model-arrow">XGBoost</div>
    <div class="model-explain-arrow">WebSHAP</div>

    <div class="top-section output">
      <span class="section-name">Model Output</span>
      <span class="section-description">Predicted likelihood of approval </span>
    </div>

    <div class="output-box">
      <span class="message-board">
        {myTabular ? myTabular.message : ''}
      </span>
    </div>

    <div class="top-section explain">
      <span class="section-name">Model Explanation</span>
      <span class="section-description"
        >Features' contribution to this prediction
      </span>
    </div>

    <div class="explain-content">
      <div class="explain-box">SHAP</div>
    </div>
  </div>
</div>
