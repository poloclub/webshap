/** SLICO segmentation implementation.
 *
 *    SLIC Superpixels
 *    Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal
 *    Fua, and Sabine Süsstrunk
 *    IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34,
 *    num. 11, p. 2274 - 2282, May 2012.
 *
 *  http://ivrl.epfl.ch/research/superpixels
 *
 * Copyright 2015  Kota Yamaguchi
 *
 * Modified by Mingyu Kim
 * https://github.com/ysjk2003/superpixel
 *
 * Modified by Jay Wang for WebSHAP
 * https://github.com/poloclub/webshap
 *
 */

export type SuperPixelOptionsSLICO = {
  method: string;
  maxIterations?: number;
  K?: number;
  perturb?: boolean;
  step?: number;
  enforceConnectivity?: boolean;
};

const createImageData = (width: number, height: number) => {
  const canvas = document.createElement('canvas');
  canvas.style.visibility = 'hidden';
  const context = canvas.getContext('2d')!;
  const imageData = context.createImageData(width, height);
  canvas.remove();
  return imageData;
};

class BaseSegmentation {
  imageData: ImageData;

  constructor(imageData: ImageData) {
    if (!(imageData instanceof ImageData)) throw 'Invaild ImageData';

    this.imageData = createImageData(imageData.width, imageData.height);
    this.imageData.data.set(imageData.data);
  }

  finer(scale: number) {
    throw new Error('makeNoise() must be implement.');
  }
  coarser(scale: number) {
    throw new Error('makeNoise() must be implement.');
  }
}

export class SLICO extends BaseSegmentation {
  width: number;
  height: number;
  method: string;
  K: number;
  perturb: undefined | boolean;
  maxIterations: number;
  step: number;
  enforceConnectivity: boolean;
  lvec!: Float64Array;
  avec!: Float64Array;
  bvec!: Float64Array;
  estep = 0;
  kSeedsL!: Float64Array | number[];
  kSeedsA!: Float64Array | number[];
  kSeedsB!: Float64Array | number[];
  kSeedsX!: Float64Array | number[];
  kSeedsY!: Float64Array | number[];
  edges!: Float64Array | Int32Array;
  _result!: ImageData;
  _numSegments!: number;

  get result() {
    return this._result;
  }

  get numSegments() {
    return this._numSegments;
  }

  constructor(imageData: ImageData, options: SuperPixelOptionsSLICO) {
    super(imageData);
    this.width = this.imageData.width;
    this.height = this.imageData.height;
    options = options || {};
    this.method = options.method || 'FixedK';
    this.perturb =
      typeof options.perturb === 'undefined' ? true : options.perturb;
    this.maxIterations = options.maxIterations || 10;
    this.K = options.K || 1024;
    this.step = options.step || 200;
    this.enforceConnectivity =
      options.enforceConnectivity === false ? false : true;
    this._compute();
  }

  finer() {
    const newK = Math.min(8962, Math.round(this.K * 2.0));
    if (newK !== this.K) {
      this.K = newK;
      this._compute();
    }
  }

  coarser() {
    const newK = Math.max(16, Math.round(this.K / 2.0));
    if (newK !== this.K) {
      this.K = newK;
      this._compute();
    }
  }

  _compute() {
    const labels =
      this.method === 'FixedK'
        ? this.performSLICOForGivenK()
        : this.performSLICOForGivenStepSize();
    const result = new ImageData(this.width, this.height);
    this._numSegments = remapLabels(labels);
    encodeLabels(labels, result.data);
    this._result = result;
  }

  rgb2xyz(sRGB: number[]) {
    const R = Math.floor(sRGB[0]) / 255.0,
      G = Math.floor(sRGB[1]) / 255.0,
      B = Math.floor(sRGB[2]) / 255.0,
      r = R <= 0.04045 ? R / 12.92 : Math.pow((R + 0.055) / 1.055, 2.4),
      g = G <= 0.04045 ? G / 12.92 : Math.pow((R + 0.055) / 1.055, 2.4),
      b = B <= 0.04045 ? B / 12.92 : Math.pow((R + 0.055) / 1.055, 2.4);
    return [
      r * 0.4124564 + g * 0.3575761 + b * 0.1804375,
      r * 0.2126729 + g * 0.7151522 + b * 0.072175,
      r * 0.0193339 + g * 0.119192 + b * 0.9503041
    ];
  }

  rgb2lab(sRGB: number[]) {
    const epsilon = 0.008856, //actual CIE standard
      kappa = 903.3, //actual CIE standard
      Xr = 0.950456, //reference white
      Yr = 1.0, //reference white
      Zr = 1.088754, //reference white
      xyz = this.rgb2xyz(sRGB),
      xr = xyz[0] / Xr,
      yr = xyz[1] / Yr,
      zr = xyz[2] / Zr,
      fx = xr > epsilon ? Math.pow(xr, 1.0 / 3.0) : (kappa * xr + 16.0) / 116.0,
      fy = yr > epsilon ? Math.pow(yr, 1.0 / 3.0) : (kappa * yr + 16.0) / 116.0,
      fz = zr > epsilon ? Math.pow(zr, 1.0 / 3.0) : (kappa * zr + 16.0) / 116.0;
    return [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)];
  }

  doRGBtoLABConversion(imageData: ImageData) {
    const size = this.width * this.height,
      data = imageData.data;
    this.lvec = new Float64Array(size);
    this.avec = new Float64Array(size);
    this.bvec = new Float64Array(size);
    for (let j = 0; j < size; ++j) {
      const r = data[4 * j + 0],
        g = data[4 * j + 1],
        b = data[4 * j + 2];
      const lab = this.rgb2lab([r, g, b]);
      this.lvec[j] = lab[0];
      this.avec[j] = lab[1];
      this.bvec[j] = lab[2];
    }
  }

  detectLabEdges() {
    const w = this.width;
    this.edges = fillArray(new Float64Array(this.width * this.height), 0);
    for (let j = 1; j < this.height - 1; ++j) {
      for (let k = 1; k < this.width - 1; ++k) {
        const i = Math.floor(j * this.width + k),
          dx =
            Math.pow(this.lvec[i - 1] - this.lvec[i + 1], 2) +
            Math.pow(this.avec[i - 1] - this.avec[i + 1], 2) +
            Math.pow(this.bvec[i - 1] - this.bvec[i + 1], 2),
          dy =
            Math.pow(this.lvec[i - w] - this.lvec[i + w], 2) +
            Math.pow(this.avec[i - w] - this.avec[i + w], 2) +
            Math.pow(this.bvec[i - w] - this.bvec[i + w], 2);
        this.edges[i] = dx + dy;
      }
    }
  }

  perturbSeeds() {
    const dx8 = [-1, -1, 0, 1, 1, 1, 0, -1],
      dy8 = [0, -1, -1, -1, 0, 1, 1, 1],
      numSeeds = this.kSeedsL.length;
    for (let n = 0; n < numSeeds; ++n) {
      const ox = Math.floor(this.kSeedsX[n]), //original x
        oy = Math.floor(this.kSeedsY[n]), //original y
        oind = Math.floor(oy * this.width + ox);
      let storeind = Math.floor(oind);
      for (let i = 0; i < 8; ++i) {
        const nx = Math.floor(ox + dx8[i]); //new x
        const ny = Math.floor(oy + dy8[i]); //new y
        if (nx >= 0 && nx < this.width && ny >= 0 && ny < this.height) {
          const nind = Math.floor(ny * this.width + nx);
          if (this.edges[nind] < this.edges[storeind]) storeind = nind;
        }
      }
      if (storeind != oind) {
        this.kSeedsX[n] = Math.floor(storeind % this.width);
        this.kSeedsY[n] = Math.floor(storeind / this.width);
        this.kSeedsL[n] = this.lvec[storeind];
        this.kSeedsA[n] = this.avec[storeind];
        this.kSeedsB[n] = this.bvec[storeind];
      }
    }
  }

  getLABXYSeedsForGivenStepSize(step: number, perturb?: boolean) {
    let n = 0;
    const xstrips = Math.round(0.5 + this.width / step),
      ystrips = Math.round(0.5 + this.height / step),
      xerr = Math.round(this.width - step * xstrips),
      yerr = Math.round(this.height - step * ystrips),
      xerrperstrip = xerr / xstrips,
      yerrperstrip = yerr / ystrips,
      xoff = Math.floor(step / 2),
      yoff = Math.floor(step / 2),
      numSeeds = xstrips * ystrips;
    this.kSeedsL = new Float64Array(numSeeds);
    this.kSeedsA = new Float64Array(numSeeds);
    this.kSeedsB = new Float64Array(numSeeds);
    this.kSeedsX = new Float64Array(numSeeds);
    this.kSeedsY = new Float64Array(numSeeds);
    for (let y = 0; y < ystrips; ++y) {
      const ye = Math.floor(y * yerrperstrip);
      for (let x = 0; x < xstrips; ++x) {
        const xe = Math.floor(x * xerrperstrip);
        const i = Math.floor(
          (y * step + yoff + ye) * this.width + (x * step + xoff + xe)
        );
        this.kSeedsL[n] = this.lvec[i];
        this.kSeedsA[n] = this.avec[i];
        this.kSeedsB[n] = this.bvec[i];
        this.kSeedsX[n] = x * step + xoff + xe;
        this.kSeedsY[n] = y * step + yoff + ye;
        ++n;
      }
    }
    if (perturb) this.perturbSeeds();
  }

  getLABXYSeedsForGivenK(K: number, perturb?: boolean) {
    const size = Math.floor(this.width * this.height);
    const step = Math.sqrt(size / K);
    const xoff = Math.round(step / 2);
    const yoff = Math.round(step / 2);
    let n = 0;
    let r = 0;
    this.kSeedsL = [];
    this.kSeedsA = [];
    this.kSeedsB = [];
    this.kSeedsX = [];
    this.kSeedsY = [];
    for (let y = 0; y < this.height; ++y) {
      const Y = Math.floor(y * step + yoff);
      if (Y > this.height - 1) break;
      for (let x = 0; x < this.width; ++x) {
        //var X = x*step + xoff;  //square grid
        const X = Math.floor(x * step + (xoff << (r & 0x1))); //hex grid
        if (X > this.width - 1) break;
        const i = Math.floor(Y * this.width + X);
        this.kSeedsL.push(this.lvec[i]);
        this.kSeedsA.push(this.avec[i]);
        this.kSeedsB.push(this.bvec[i]);
        this.kSeedsX.push(X);
        this.kSeedsY.push(Y);
        ++n;
      }
      ++r;
    }
    if (perturb) this.perturbSeeds();
  }

  performSuperpixelSegmentationVariableSandM(
    kLabels: Float64Array | Int32Array,
    step: number,
    maxIterations: number
  ) {
    const size = Math.floor(this.width * this.height),
      numK = this.kSeedsL.length,
      offset = Math.floor(step < 10 ? step * 1.5 : step),
      sigmal = fillArray(new Float64Array(numK), 0),
      sigmaa = fillArray(new Float64Array(numK), 0),
      sigmab = fillArray(new Float64Array(numK), 0),
      sigmax = fillArray(new Float64Array(numK), 0),
      sigmay = fillArray(new Float64Array(numK), 0),
      clusterSize = fillArray(new Int32Array(numK), 0),
      distxy = fillArray(new Float64Array(size), Infinity),
      distlab = fillArray(new Float64Array(size), Infinity),
      distvec = fillArray(new Float64Array(size), Infinity),
      maxlab = fillArray(new Float64Array(numK), Math.pow(10, 2)),
      maxxy = fillArray(new Float64Array(numK), Math.pow(step, 2));
    let numIter = 0,
      i,
      j,
      k,
      n,
      x,
      y;
    while (numIter < maxIterations) {
      ++numIter;
      // Assign the closest cluster.
      fillArray(distvec, Infinity);
      for (n = 0; n < numK; ++n) {
        const y1 = Math.floor(Math.max(0, this.kSeedsY[n] - offset)),
          y2 = Math.floor(Math.min(this.height, this.kSeedsY[n] + offset)),
          x1 = Math.floor(Math.max(0, this.kSeedsX[n] - offset)),
          x2 = Math.floor(Math.min(this.width, this.kSeedsX[n] + offset));
        for (y = y1; y < y2; ++y) {
          for (x = x1; x < x2; ++x) {
            i = Math.floor(y * this.width + x);
            if (!(y < this.height && x < this.width && y >= 0 && x >= 0))
              throw 'Assertion error';
            const l = this.lvec[i],
              a = this.avec[i],
              b = this.bvec[i];
            distlab[i] =
              Math.pow(l - this.kSeedsL[n], 2) +
              Math.pow(a - this.kSeedsA[n], 2) +
              Math.pow(b - this.kSeedsB[n], 2);
            distxy[i] =
              Math.pow(x - this.kSeedsX[n], 2) +
              Math.pow(y - this.kSeedsY[n], 2);
            const dist = distlab[i] / maxlab[n] + distxy[i] / maxxy[n];
            if (dist < distvec[i]) {
              distvec[i] = dist;
              kLabels[i] = n;
            }
          }
        }
      }
      //console.log("iter = " + numIter + ", sum_dist = " + sum(distvec));
      // Assign the max color distance for a cluster.
      if (numIter === 0) {
        fillArray(maxlab, 1);
        fillArray(maxxy, 1);
      }
      for (i = 0; i < size; ++i) {
        if (maxlab[kLabels[i]] < distlab[i]) maxlab[kLabels[i]] = distlab[i];
        if (maxxy[kLabels[i]] < distxy[i]) maxxy[kLabels[i]] = distxy[i];
      }
      // Recalculate the centroid and store in the seed values.
      fillArray(sigmal, 0);
      fillArray(sigmaa, 0);
      fillArray(sigmab, 0);
      fillArray(sigmax, 0);
      fillArray(sigmay, 0);
      fillArray(clusterSize, 0);
      for (j = 0; j < size; ++j) {
        const temp = kLabels[j];
        if (temp < 0) throw 'Assertion error';
        sigmal[temp] += this.lvec[j];
        sigmaa[temp] += this.avec[j];
        sigmab[temp] += this.bvec[j];
        sigmax[temp] += j % this.width;
        sigmay[temp] += j / this.width;
        clusterSize[temp]++;
      }
      for (k = 0; k < numK; ++k) {
        if (clusterSize[k] <= 0) clusterSize[k] = 1;
        //computing inverse now to multiply, than divide later.
        const inv = 1.0 / clusterSize[k];
        this.kSeedsL[k] = sigmal[k] * inv;
        this.kSeedsA[k] = sigmaa[k] * inv;
        this.kSeedsB[k] = sigmab[k] * inv;
        this.kSeedsX[k] = sigmax[k] * inv;
        this.kSeedsY[k] = sigmay[k] * inv;
      }
    }
  }

  enforceLabelConnectivity(
    labels: Float64Array | Int32Array,
    nlabels: Float64Array | Int32Array,
    K: number
  ) {
    const dx4 = [-1, 0, 1, 0],
      dy4 = [0, -1, 0, 1],
      size = this.width * this.height,
      SUPSZ = Math.floor(size / K);
    let c, n, x, y, nindex;
    let label = 0,
      oindex = 0,
      adjlabel = 0; // adjacent label
    const xvec = new Int32Array(size),
      yvec = new Int32Array(size);
    for (let j = 0; j < this.height; ++j) {
      for (let k = 0; k < this.width; ++k) {
        if (nlabels[oindex] < 0) {
          nlabels[oindex] = label;
          // Start a new segment.
          xvec[0] = k;
          yvec[0] = j;
          //  Quickly find an adjacent label for use later if needed.
          for (n = 0; n < 4; ++n) {
            x = Math.floor(xvec[0] + dx4[n]);
            y = Math.floor(yvec[0] + dy4[n]);
            if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
              nindex = Math.floor(y * this.width + x);
              if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
            }
          }
          let count = 1;
          for (c = 0; c < count; ++c) {
            for (n = 0; n < 4; ++n) {
              x = Math.floor(xvec[c] + dx4[n]);
              y = Math.floor(yvec[c] + dy4[n]);
              if (x >= 0 && x < this.width && y >= 0 && y < this.height) {
                nindex = Math.floor(y * this.width + x);
                if (nlabels[nindex] < 0 && labels[oindex] == labels[nindex]) {
                  xvec[count] = x;
                  yvec[count] = y;
                  nlabels[nindex] = label;
                  ++count;
                }
              }
            }
          }
          // If segment size is less then a limit, assign an
          // adjacent label found before, and decrement label count.
          if (count <= SUPSZ >> 2) {
            for (c = 0; c < count; c++) {
              const ind = Math.floor(yvec[c] * this.width + xvec[c]);
              nlabels[ind] = adjlabel;
            }
            --label;
          }
          ++label;
        }
        ++oindex;
      }
    }
    return label;
  }

  performSLICOForGivenStepSize() {
    const size = this.width * this.height,
      kLabels = fillArray(new Int32Array(size), -1);
    this.doRGBtoLABConversion(this.imageData);
    if (this.perturb) this.detectLabEdges();
    this.getLABXYSeedsForGivenStepSize(this.step, this.perturb);
    this.performSuperpixelSegmentationVariableSandM(
      kLabels,
      this.step,
      this.maxIterations
    );
    let numlabels = kLabels.length;
    if (this.enforceConnectivity) {
      const nlabels = fillArray(new Int32Array(size), -1);
      numlabels = this.enforceLabelConnectivity(
        kLabels,
        nlabels,
        size / (this.step * this.estep)
      );
      for (let i = 0; i < size; ++i) kLabels[i] = nlabels[i];
    }
    return kLabels;
  }

  performSLICOForGivenK() {
    const size = this.width * this.height,
      kLabels = fillArray(new Int32Array(size), -1);
    this.doRGBtoLABConversion(this.imageData);
    if (this.perturb) this.detectLabEdges();
    this.getLABXYSeedsForGivenK(this.K, this.perturb);
    const step = Math.sqrt(size / this.K) + 2.0;
    this.performSuperpixelSegmentationVariableSandM(
      kLabels,
      step,
      this.maxIterations
    );
    let numlabels = kLabels.length;
    if (this.enforceConnectivity) {
      const nlabels = fillArray(new Int32Array(size), -1);
      numlabels = this.enforceLabelConnectivity(kLabels, nlabels, this.K);
      for (let i = 0; i < size; ++i) kLabels[i] = nlabels[i];
    }
    return kLabels;
  }
}

function fillArray(array: Float64Array | Int32Array, value: number) {
  for (let i = 0; i < array.length; ++i) array[i] = value;
  return array;
}

// function findMinMax(data) {
//   var min = Infinity, max = -Infinity;
//   for (var i = 0; i < data.length; ++i) {
//     min = Math.min(min, data[i]);
//     max = Math.max(max, data[i]);
//   }
//   return [min, max];
// }

// function sum(data) {
//   var value = 0;
//   for (var i = 0; i < data.length; ++i)
//     value += data[i];
//   return value;
// }

// Remap label indices.
function remapLabels(labels: Int32Array | Float64Array) {
  const map: { [key: number]: number } = {};
  let index = 0;
  for (let i = 0; i < labels.length; ++i) {
    const label = labels[i];
    if (map[label] === undefined) map[label] = index++;
    labels[i] = map[label];
  }
  return index;
}

function encodeLabels(
  labels: Int32Array | Float64Array,
  data: Uint8ClampedArray
) {
  for (let i = 0; i < labels.length; ++i) {
    const label = labels[i];
    data[4 * i + 0] = 255 & label;
    data[4 * i + 1] = 255 & (label >> 8);
    data[4 * i + 2] = 255 & (label >> 16);
    data[4 * i + 3] = 255;
  }
}
