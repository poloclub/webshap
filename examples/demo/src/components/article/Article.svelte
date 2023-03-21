<script lang="ts">
  import { onMount } from 'svelte';
  import WebShap from '../webshap/WebShap.svelte';
  import Youtube from './Youtube.svelte';
  import { fade, fly } from 'svelte/transition';

  import iconLogo from '../../imgs/icon-webshap.svg?raw';
  import iconGT from '../../imgs/logo-gt.svg?raw';
  import iconCopy from '../../imgs/icon-copy-box.svg?raw';
  import iconCheckBox from '../../imgs/icon-check-box.svg?raw';
  import text from './ArticleText.yml';

  let component: HTMLElement | null = null;
  let currentPlayer = null;

  let bibtexCopied = false;
  let bibtexHovering = false;

  onMount(() => {
    //pass
  });
</script>

<style lang="scss">
  @import './Article.scss';
</style>

<svelte:head>
  <!-- <script
    id="MathJax-script"
    async
    src={`${import.meta.env.BASE_URL}data/mathjax/tex-chtml.js`}></script> -->
  <script
    id="MathJax-script"
    async
    src="https://cdn.jsdelivr.net/npm/mathjax@3.2/es5/tex-mml-chtml.js"
  ></script>
</svelte:head>

<div class="article-page" bind:this="{component}">
  <div class="main-app" tabindex="-1">
    <div class="webshap-app"><WebShap /></div>
  </div>

  <div class="article">
    <h2 id="tool">
      <span>What is </span>
      <span class="svg-icon logo-icon">{@html iconLogo}</span>
      <span><span class="tool-name">WebSHAP</span>?</span>

      <!-- <span>What is WebSHAP?</span>
      <span class="svg-icon logo-icon">{@html iconLogo}</span> -->
    </h2>

    {#each text.tool as p}
      <p>{@html p}</p>
    {/each}

    <h2 id="usage">How to Use <span class="tool-name">WebSHAP</span>?</h2>
    <p>{@html text.usage.p1}</p>

    <h2 id="usage">
      Any Examples Using <span class="tool-name">WebSHAP</span>?
    </h2>
    <p>{@html text.tutorial.p1}</p>

    {#each text.tutorial.items as item, i}
      <h4 id="{item.id}">{item.name}</h4>
      <p>{@html item.descriptions[0]}</p>
      <div class="video" class:wide-video="{false}">
        <video autoplay loop muted playsinline>
          <source src="{`${import.meta.env.BASE_URL}video/${item.id}.mp4`}" />
          <track kind="captions" />
        </video>
        <div class="figure-caption">
          Video {i + 1}. {@html item.caption}
        </div>
      </div>
      {#each item.descriptions.slice(1) as p}
        <p>{@html p}</p>
      {/each}
    {/each}

    <h2 id="usage">
      How is <span class="tool-name" style="margin-right: 8px;">WebSHAP</span> Developed?
    </h2>
    <p>{@html text.development}</p>

    <h2 id="team">Who Developed <span class="tool-name">WebSHAP</span>?</h2>
    <p>{@html text.team}</p>

    <h2 id="contribute">How Can I Contribute?</h2>
    <p>{@html text.contribute[0]}</p>
    <p>{@html text.contribute[1]}</p>

    <h2 id="cite">How to learn more?</h2>

    <p>{@html text.cite.intro}</p>

    <div class="paper-info">
      <div class="left">
        <a target="_blank" href="{text.cite.paperLink}"
          ><img src="paper-preview.webp" /></a
        >
      </div>
      <div class="right">
        <a target="_blank" href="{text.cite.paperLink}"
          ><span class="paper-title">{text.cite.title}</span></a
        >
        <a target="_blank" href="{text.cite.venueLink}"
          ><span class="paper-venue">{text.cite.venue}</span></a
        >
        <div class="paper-authors">
          {#each text.cite.authors as author, i}
            <a href="{author.url}" target="_blank"
              >{author.name}{i === text.cite.authors.length - 1 ? '' : ','}</a
            >
          {/each}
        </div>
      </div>
    </div>
    <div
      class="bibtex-block"
      on:mouseenter="{() => {
        bibtexHovering = true;
      }}"
      on:mouseleave="{() => {
        bibtexHovering = false;
      }}"
    >
      <div class="bibtex">
        {@html text.cite['bibtex']}
      </div>

      <div
        class="copy-button"
        class:hide="{!bibtexHovering}"
        on:click="{() => {
          navigator.clipboard.writeText(text.cite['bibtex']).then(() => {
            bibtexCopied = true;
          });
        }}"
        on:mouseleave="{() => {
          setTimeout(() => {
            bibtexCopied = false;
          }, 500);
        }}"
      >
        {#if bibtexCopied}
          <span class="svg-icon check">{@html iconCheckBox}</span>
          <span class="copy-label check">Copied!</span>
        {:else}
          <span class="svg-icon copy">{@html iconCopy}</span>
          <span class="copy-label copy">Copy</span>
        {/if}
      </div>
    </div>
  </div>

  <div class="article-footer">
    <div class="footer-main">
      <div class="footer-logo">
        <a target="_blank" href="https://www.gatech.edu/">
          <div class="svg-logo" title="Georgia Tech">
            {@html iconGT}
          </div>
        </a>
      </div>
    </div>
  </div>
</div>
