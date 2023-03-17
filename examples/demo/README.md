# WebSHAP <a href="https://poloclub.github.io/webshap/"><img align="right" src="https://raw.githubusercontent.com/poloclub/webshap/main/examples/demo/src/imgs/icon-webshap.svg" height="38"></img></a>

[![build](https://github.com/xiaohk/webshap/actions/workflows/build.yml/badge.svg)](https://github.com/xiaohk/webshap/actions/workflows/build.yml)
[![npm](https://img.shields.io/npm/v/webshap?color=orange)](https://www.npmjs.com/package/webshap)
[![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/poloclub/webshap/blob/main/LICENSE)
[![arxiv badge](https://img.shields.io/badge/arXiv-2209.09227-red)](https://arxiv.org/abs/2303.09545)
[![DOI:10.1145/3543873.3587362](https://img.shields.io/badge/DOI-10.1145/3543873.3587362-blue)](https://doi.org/10.1145/3543873.3587362)

JavaScript library that explains any machine learning models in your browser!

<table>
  <tr>
    <td colspan="2"><a href="https://poloclub.github.io/webshap"><img src='https://user-images.githubusercontent.com/15007159/225991959-c2b10d8b-be24-4f5c-a6f4-2b9f5876095c.gif' width="100%"></a></td>
  </tr>
  <tr></tr>
  <tr>
    <td>
      <table>
        <tr>
          <td colspan="3" align="center">Live Explainer Demos</td>
        </tr>
        <tr></tr>
        <tr>
          <td><a href="https://poloclub.github.io/webshap/?model=tabulark">💰 Loan Prediction</a></td>
          <td><a href="https://poloclub.github.io/webshap/?model=image">🌠 Image Classifier</a></td>
          <td><a href="https://poloclub.github.io/webshap/?model=text">🔤 Toxicity Detector</a></td>
        </tr>
      </table>
    </td>
    <td>
      <table>
        <tr>
          <td colspan="1" align="center">Research Paper</td>
        </tr>
        <tr></tr>
        <tr>
          <td><a href="https://poloclub.github.io/webshap/?model=tabulark">📖 WebSHAP(TheWebConf'23)</a></td>
        </tr>
      </table>
    </td>
  </tr>
</table>

## What is WebSHAP?

WebSHAP is a JavaScript library that adapts Kernel SHAP for the Web environments. You can use it to explain any machine learning models available on the Web directly in your browser. Given a model's prediction on a data point, WebSHAP can compute the importance score for each input feature. WebSHAP leverages modern Web technologies such as WebGL to accelerate computations. With a moderate model size and number of input features, WebSHAP can generate explanations in real time.✨

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

### Demo 1: Explaining XGBoost

|<img src='https://user-images.githubusercontent.com/15007159/226003794-94b3a0d9-b132-4ab2-80fc-33aecbd66337.jpg'>|
|:---:|
|[🔎 WebSHAP explaining an XGBoost-based loan approval model](https://poloclub.github.io/webshap/?model=tabular) 💰|

We present `Loan Explainer` as an example of applying WebSHAP to explain a financial ML model in browsers. For a live demo of Loan Explainer, visit [this webpage](https://poloclub.github.io/webshap/?model=text).

This example showcases a bank using an [XGBoost classifier](https://github.com/dmlc/xgboost) on the [LendingClub dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) to predict if a loan applicant will be able to repay the loan on time. With this model, the bank can make automatic loan approval decisions. It's important to understand how these high-stakes decisions are being made, and that's where WebSHAP comes in. It provides *private*, *ubiquitous*, and *interactive* ML explanations.

This demo runs entirely on the client side, making it accessible from desktops, tablets, and phones. The model inference is powered by [ONNX Runtime](https://github.com/microsoft/onnxruntime). The UI is implemented using [Svelte](https://github.com/sveltejs/svelte). With Loan Explainer, users can experiment with different feature inputs and instantly see the model's predictions, along with clear explanations for those predictions.

### Demo 2: Explaining Convolutional Neural Networks

|<img src='https://user-images.githubusercontent.com/15007159/226048036-c92043ee-df9b-42a6-9607-a15dd3da5470.jpg'>|
|:---:|
|[🔎 WebSHAP explaining a convolutional neural network for image classification](https://poloclub.github.io/webshap/?model=image) 🌠|

We apply WebSHAP to explain convolutional neural networks (CNNs) in browsers. The live demo of this explainer is available on [this webpage](https://poloclub.github.io/webshap/?model=image).

In this example, we first train a TinyVGG model to classify images into four categories: 🐞`Ladybug`, ☕️`Espresso`, 🍊`Orange`, and 🚙`Sports Car`. TinyVGG is a type of convolutional neural network. For more details about the model architecture, check out [CNN Explainer](https://poloclub.github.io/cnn-explainer). TinyVGG is implemented using [TensorFlow.js](https://www.tensorflow.org/js).

To explain the predictions of TinyVGG, we first apply image segmentation ([SLIC](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/SLIC_Superpixels.pdf)) to divide the input image into multiple segments. Then, we compute SHAP scores on each segment for each class. The background data here are white pixels. We compute SHAP values for segments instead of raw pixels for computation efficiency. For example, in the figure above, there are only 16 input features (16 segments) for WebSHAP, but there would have been $64 \times 64 \times 3 = 12288$ input features if we use raw pixels. Finally, we visualize the SHAP scores of each segment as an overlay with a diverging color scale on top of the original input image.

Everything in this example (TinyVGG, image segmenter, WebSHAP) runs in the user's browser. In addition, WebSHAP enables *interactive* explanation: users can click a button to use a random input image or upload their own images. Both model inference and SHAP computation are real-time.

### Demo 3: Explaining Transformer-based Text Classifiers

|<img src='https://user-images.githubusercontent.com/15007159/226048056-1e5f0d2d-f7f1-4f8f-9a48-7b2b2f4cad72.jpg'>|
|:---:|
|[🔎 WebSHAP explaining a transformer model for text classification](https://poloclub.github.io/webshap/?model=text) 🔤|

We use WebSHAP to explain the predictions of a Transformer text classifier in browsers. The live demo for this explainer is accessible on [this webpage](https://poloclub.github.io/webshap/?model=text).

We train an [XtremeDistil model](https://github.com/microsoft/xtreme-distil-transformers) to predict if an input text is toxic. The XtremeDistil model is a distilled version of pre-trained transformer-based language model BERT. We train this model on the [Toxic Comments dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data). Then, we quantize and export the trained model to use `int8` weights with [ONNX](https://github.com/onnx/onnxmltools). We use [TensorFlow.js](https://github.com/tensorflow/tfjs-models/blob/master/qna/src/bert_tokenizer.ts) for tokenization and [ONNX Runtime](https://onnxruntime.ai) for model inference.

To explain the model's predictions, we compute SHAP scores for each input token. For background data, we use BERT's attention mechanism to mask tokens. For example, we represent a "missing" token by setting its [attention map](https://huggingface.co/docs/transformers/glossary#attention-mask) to `0`, which tells the model to ignore this token. Finally, we visualize the SHAP scores as token's background color with a diverging color scale.

All components in this example (XtremeDistil, tokenizer, WebSHAP) runs on the client-side. WebSHAP provides *private*, *ubiquitous*, and *interactive* explanations. Users can edit the input text and see new predictions and explanations. The model inference is real-time, and SHAP computation takes about 5 seconds for 50 tokens.

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

## Developing the Application Examples

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

Navigate to localhost:3000. You should see three Explainers running in your browser :)

## Credits

WebSHAP is created by <a href='https://zijie.wang/' target='_blank'>Jay Wang</a> and <a href='' target='_blank'>Polo Chau</a>.

## Citation

To learn more about WebSHAP, please read our [research paper](https://arxiv.org/abs/2303.09545) (published at [TheWebConf'23](https://www2023.thewebconf.org)). If you find WebSHAP useful for your research, please consider citing our paper. And if you're building any exciting projects with WebSHAP, we'd love to hear about them!

```bibTeX
@inproceedings{wangWebSHAPExplainingAny2023,
  title = {{{WebSHAP}}: {{Towards Explaining Any Machine Learning Models Anywhere}}},
  shorttitle = {{{WebSHAP}}},
  booktitle = {Companion {{Proceedings}} of the {{Web Conference}} 2023},
  author = {Wang, Zijie J. and Chau, Duen Horng},
  year = {2023},
  langid = {english}
}
```

## License

The software is available under the [MIT License](https://github.com/poloclub/webshap/blob/main/LICENSE).

## Contact

If you have any questions, feel free to [open an issue](https://github.com/poloclub/webshap/issues/new) or contact [Jay Wang](https://zijie.wang).
