# WebSHAP <a href="https://poloclub.github.io/webshap/"><img align="right" src="examples/demo/src/imgs/icon-webshap.svg" height="38"></img></a>

[![build](https://github.com/xiaohk/webshap/actions/workflows/build.yml/badge.svg)](https://github.com/xiaohk/webshap/actions/workflows/build.yml)
[![npm](https://img.shields.io/npm/v/webshap?color=red)](https://www.npmjs.com/package/webshap)
[![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/poloclub/webshap/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7604420.svg)](https://doi.org/10.5281/zenodo.7604420)
<!-- [![arxiv badge](https://img.shields.io/badge/arXiv-2209.09227-red)](https://arxiv.org/abs/2209.09227) -->
<!-- [![DOI:10.1145/3491101.3519653](https://img.shields.io/badge/DOI-10.1145/3491101.3519653-blue)](https://doi.org/10.1145/3491101.3519653) -->

Explaining any machine learning models directly in your browser!

<!-- <table>
  <tr>
    <td colspan="4"><a href="https://poloclub.github.io/timbertrek"><img src='https://i.imgur.com/t4qtPPX.png'></a></td>
  </tr>
  <tr></tr>
  <tr>
    <td><a href="https://poloclub.github.io/timbertrek">🚀 Live Demo</a></td>
    <td><a href="https://youtu.be/3eGqTmsStJM">📺 Demo Video</a></td>
    <td><a href="https://youtu.be/l1mr9z1TuAk">👨🏻‍🏫 Conference Talk</a></td>
    <td><a href="https://arxiv.org/abs/2209.09227">📖 Research Paper</a></td>
  </tr>
</table> -->


|<img src="https://i.imgur.com/IaYAGex.png">|
|:---:|
|<img src='https://user-images.githubusercontent.com/15007159/216748746-9cc9eb56-e456-454b-b448-52d400801610.gif'>|
|<a href="https://poloclub.github.io/webshap/">🔎 Example Web application applying WebSHAP to explain loan approval decisions|

## What is WebSHAP?

WebSHAP is a TypeScript library that adapts Kernel SHAP for the Web environments. You can use it to explain any machine learning models available on the Web directly in your browser. Given a model's prediction on a data point, WebSHAP can compute the importance score for each input feature. WebSHAP leverages modern Web technologies such as WebGL to accelerate computations. With a moderate model size and number of input features, WebSHAP can generate explanations in real time.✨

## Getting Started

### Installation

WebSHAP supports both browser and Node.js environments. To install WebSHAP, you can use `npm`:

```bash
npm install webshap
```

### Explain Machine Learning Models

WebSHAP uses the Kernel SHAP algorithm to interpret machine learning (ML) models. This algorithm uses a game theoretic approach to approximate the importance of each input feature. You can learn more about Kernel SHAP from [the original paper](https://arxiv.org/abs/1705.07874) or [this nice tutorial](https://christophm.github.io/interpretable-ml-book/shap.html).

To run WebSHAP on your model, you need to prepare the following three arguments.

|Name|Description|Type|Details|
|:---|:---|:---|:---|
|ML Model|A function that transforms input data into predicted probabilities|`(x: number[][]) => Promise<number[]>`|This function wraps your ML model inference code. WebSHAP is model-agnostic, so any model can be used (e.g. random forest, CNNs, transformers).|
|Data Point|The input data for a prediction.|`number[][]`|WebSHAP generates local explanations by computing the feature importance for individual predictions.|
|Background Data|A 2D array that represents feature "missingness" |`number[][]`|WebSHAP approximates the contribution of a feature by comparing it to its missing value (also known as the base value). Using all zeros is the simplest option, but using the median or a subset of your data can improve accuracy.|

Then, you can generate explanations with WebSHAP through two functions:

```typescript
// Import the class KernelSHAP from the webshap module
import { KernelSHAP } from 'webshap';

// Create an explainer object by feeding it with background data
const explainer = new KernelSHAP(
  (x: number[][]) => myModel(x),  // ML Model function wrapper
  backgroundData,                 // Background data
  0.2022                          // Random seed
);

// Explain one prediction
let shapValues = await explainer.explainOneInstance(x);

// By default, WebSHAP automatically chooses the number of feature
// permutations. You can also pass it as an argument here.
const nSamples = 512;
shapValues = await explainer.explainOneInstance(x, nSamples);

// Finally, `shapValues` contains the importance score for each feature in `x`
console.log(shapValue);
```

See the [WebSHAP Documentation](http://poloclub.github.io/webshap/doc/) for more details.

## Application Example
|<img src='https://i.imgur.com/42IGD2Y.png'>|
|:---:|
|[🔎 WebSHAP explaining an XGBoost-based loan approval model](https://poloclub.github.io/webshap)|

We present `Loan Explainer` as an example of applying WebSHAP to explain a financial ML model in browsers. For a live demo of Loan Explainer, visit: <https://poloclub.github.io/webshap>.

This example showcases a bank using an [XGBoost classifier](https://github.com/dmlc/xgboost) on the [LendingClub dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) to predict if a loan applicant will be able to repay the loan on time. With this model, the bank can make automatic loan approval decisions. It's important to understand how these high-stakes decisions are being made, and that's where WebSHAP comes in. It provides *private*, *ubiquitous*, and *interactive* ML explanations.

This demo runs entirely on the client side, making it accessible from desktops, tablets, and phones. The model inference is powered by [ONNX Runtime](https://github.com/microsoft/onnxruntime). The UI is implemented using [Svelte](https://github.com/sveltejs/svelte). With Loan Explainer, users can experiment with different feature inputs and instantly see the model's predictions, along with clear explanations for those predictions.


## Developing WebSHAP

Clone or download this repository:

```bash
git clone git@github.com:poloclub/webshap.git
```

Install the dependencies:

```bash
npm install
```

Use Vitest for unit testing:

```
npm run test
```

## Developing the Loan Explainer Example

Clone or download this repository:

```bash
git clone git@github.com:poloclub/webshap.git
```

Navigate to the example folder:

```bash
cd ./examples/demo
```

Install the dependencies:

```bash
npm install
```

Then run Loan Explainer:

```
npm run dev
```

Navigate to localhost:3000. You should see Loan Explainer running in your browser :)

## Credits

WebSHAP is created by <a href='https://zijie.wang/' target='_blank'>Jay Wang</a> and <a href='' target='_blank'>Polo Chau</a>.


## License

The software is available under the [MIT License](https://github.com/poloclub/webshap/blob/main/LICENSE).

## Contact

If you have any questions, feel free to [open an issue](https://github.com/poloclub/webshap/issues/new) or contact [Jay Wang](https://zijie.wang).
