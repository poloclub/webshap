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

<div class="tabular-wrapper" bind:this="{component}">
  <div class="tabular">
    <div class="feature-content">
      {#if myTabular && myTabular.contFeatures}
        <span class="feature-header cont">Continuous Features</span>
        <div class="content-cont">
          {#each myTabular.contFeatures as item}
            <div class="input-wrapper">
              <span class="name">{item.displayName}</span>
              <input
                class="feature-input"
                type="number"
                step="{item.requiresInt ? 1 : 0.1}"
                bind:value="{item.value}"
              />
            </div>
          {/each}
        </div>
      {/if}

      {#if myTabular && myTabular.contFeatures}
        <span class="feature-header cat">Categorical Features</span>
        <div class="content-cat">
          {#each myTabular.catFeatures as item}
            <div class="input-wrapper">
              <span class="name">{item.displayName}</span>
              <select class="feature-select" bind:value="{item.value}">
                <!-- {#each item.allValues as value}
                  <option value="{value.level}">{value.levelDisplayName}</option
                  >
                {/each} -->
              </select>
            </div>
          {/each}
        </div>
      {/if}
    </div>

    <span class="message-board">
      {myTabular ? myTabular.message : ''}
    </span>
  </div>
</div>
