import { describe, test, expect, beforeEach } from 'vitest';
import { KernelSHAP, IrisLinearMultiClass } from '../../src/index';
import math from '../../src/utils/math-import';

const SEED = 0.20071022;

interface LocalTestContext {
  model: (x: number[][]) => Promise<number[][]>;
  data: number[][];
}

/**
 * Initialize the fixture for all tests
 */
beforeEach<LocalTestContext>(context => {
  const coefs = [
    [-0.39228899, 0.85674351, -2.23337115, -0.98440396],
    [0.55874815, -0.25246978, -0.11357907, -0.90513218],
    [-0.16645916, -0.60427374, 2.34695022, 1.88953615]
  ];
  const intercepts = [8.85065312, 1.21877625, -10.06942938];
  const model = new IrisLinearMultiClass(coefs, intercepts);

  // Wrap the model in a promise
  context.model = (x: number[][]) => {
    const promise = new Promise<number[][]>((resolve, reject) => {
      const prob = model.predictProba(x);
      resolve(prob);
    });
    return promise;
  };

  context.data = [
    [5.8, 2.8, 5.1, 2.4],
    [5.8, 2.7, 5.1, 1.9],
    [7.2, 3.6, 6.1, 2.5],
    [6.2, 2.8, 4.8, 1.8],
    [4.9, 3.1, 1.5, 0.1]
  ];
});

test<LocalTestContext>('constructor()', async ({ model, data }) => {
  const yPredProbaExp = [
    [1.81082686e-4, 5.85998237e-2, 9.41219094e-1],
    [5.62410593e-4, 1.95448232e-1, 8.03989357e-1],
    [3.42078623e-6, 1.44898582e-2, 9.85506721e-1],
    [2.01336494e-3, 4.81186435e-1, 5.168002e-1],
    [9.61680907e-1, 3.83188489e-2, 2.44545042e-7]
  ];

  const explainer = new KernelSHAP(model, data, SEED);
  await explainer.initializeModel();

  for (let i = 0; i < explainer.predictions.length; i++) {
    const curRow = explainer.predictions[i];
    for (let j = 0; j < curRow.length; j++) {
      expect(curRow[j]).toBeCloseTo(yPredProbaExp[i][j], 6);
    }
  }
});

test<LocalTestContext>('inferenceFeatureCoalitions()', async ({
  model,
  data
}) => {
  const explainer = new KernelSHAP(model, data, SEED);
  await explainer.initializeModel();

  const nSamples = 32;
  const x1 = [4.8, 3.8, 2.1, 5.4];

  // Inference on the sampled feature coalitions
  explainer.sampleFeatureCoalitions(x1, nSamples);
  await explainer.inferenceFeatureCoalitions();

  // The first 8 masks (40 samples) are deterministic, so we compare the
  // results of them with SHAP
  const yMat = explainer.yMat!;

  const expectedYMat8 = [
    [2.34024737e-4, 2.92584264e-2, 9.70507549e-1],
    [7.83806094e-4, 1.05234152e-1, 8.93982041e-1],
    [5.95290494e-6, 2.57273296e-3, 9.97421314e-1],
    [3.98048688e-3, 2.51237705e-1, 7.44781809e-1],
    [9.65035703e-1, 3.49640573e-2, 2.39918416e-7],
    [1.18388237e-1, 2.77359217e-2, 8.53875842e-1],
    [1.18388237e-1, 2.77359217e-2, 8.53875842e-1],
    [8.48781208e-2, 7.5296027e-2, 8.39825852e-1],
    [1.08259278e-1, 3.710319e-2, 8.54637532e-1],
    [1.43153664e-1, 1.42498192e-2, 8.42596517e-1],
    [7.61263115e-4, 8.12510454e-2, 9.17987691e-1],
    [2.5630753e-3, 2.62931056e-1, 7.34505868e-1],
    [4.57687762e-6, 1.55296783e-2, 9.84465745e-1],
    [7.17482502e-3, 5.65557776e-1, 4.27267399e-1],
    [9.81999391e-1, 1.80005192e-2, 8.98009388e-8],
    [3.83459082e-2, 1.05232069e-2, 9.51130885e-1],
    [3.33193346e-2, 1.02164008e-2, 9.56464265e-1],
    [1.13372736e-1, 1.2810248e-2, 8.73817016e-1],
    [3.83459082e-2, 1.05232069e-2, 9.51130885e-1],
    [5.81445787e-2, 1.14398609e-2, 9.3041556e-1],
    [6.38738355e-1, 3.57684703e-1, 3.57694207e-3],
    [6.23868246e-1, 3.75170883e-1, 9.60870367e-4],
    [5.31016397e-1, 4.67293908e-1, 1.68969532e-3],
    [5.61091557e-1, 4.38295286e-1, 6.13156739e-4],
    [8.75537945e-1, 1.24458578e-1, 3.47638462e-6],
    [1.87223884e-7, 9.79280549e-6, 9.9999002e-1],
    [1.87223884e-7, 9.79280549e-6, 9.9999002e-1],
    [1.91936214e-9, 8.3621279e-7, 9.99999162e-1],
    [7.39807714e-7, 2.04871021e-5, 9.99978773e-1],
    [7.17406018e-1, 1.82012193e-2, 2.64392763e-1],
    [3.46554078e-8, 1.42256554e-5, 9.9998574e-1],
    [2.99447062e-8, 1.3733899e-5, 9.99986236e-1],
    [8.33429858e-10, 4.44269414e-6, 9.99995556e-1],
    [1.25109909e-7, 3.97759764e-5, 9.99960099e-1],
    [4.74188212e-1, 2.87605301e-2, 4.97051258e-1],
    [9.32495551e-1, 6.65378475e-2, 9.66601136e-4],
    [9.35604187e-1, 6.41653381e-2, 2.30475371e-4],
    [9.31702316e-1, 6.70103496e-2, 1.2873343e-3],
    [9.3613236e-1, 6.36946358e-2, 1.73003939e-4],
    [9.43874083e-1, 5.61245998e-2, 1.31765293e-6]
  ];

  for (let i = 0; i < expectedYMat8.length; i++) {
    for (let j = 0; j < expectedYMat8[i].length; j++) {
      expect(yMat.get([i, j]) as number).toBeCloseTo(expectedYMat8[i][j], 8);
    }
  }

  // Compare the expected y value for all rows (we get the ground truth by
  // forcing Python SHAP to use the same kernel weight and mask as TS)
  const expectedEyMat = [
    [1.94007995e-1, 8.46534147e-2, 7.21338591e-1],
    [1.14613507e-1, 3.64241759e-2, 8.48962317e-1],
    [1.98500626e-1, 1.88654015e-1, 6.12845359e-1],
    [5.63056931e-2, 1.11025847e-2, 9.32591722e-1],
    [6.460505e-1, 3.52580672e-1, 1.36882818e-3],
    [1.43481427e-1, 3.64842564e-3, 8.52870148e-1],
    [9.48376805e-2, 5.76654166e-3, 8.99395778e-1],
    [9.35961699e-1, 6.35065542e-2, 5.3174648e-4],
    [8.02304166e-1, 1.97105432e-1, 5.90401403e-4],
    [8.51963512e-1, 1.46633558e-1, 1.4029291e-3],
    [2.00730838e-1, 1.05309211e-1, 6.93959951e-1],
    [9.61558252e-2, 5.30870748e-3, 8.98535467e-1],
    [1.42363153e-1, 3.9925927e-3, 8.53644254e-1],
    [4.1333106e-2, 3.11771136e-2, 9.2748978e-1]
  ];

  for (let i = 0; i < expectedEyMat.length; i++) {
    for (let j = 0; j < expectedEyMat[i].length; j++) {
      expect(explainer.yExpMat!.get([i, j]) as number).toBeCloseTo(
        expectedEyMat[i][j],
        8
      );
    }
  }
});

test<LocalTestContext>('explainOneInstance()', async ({ model, data }) => {
  const explainer = new KernelSHAP(model, data, SEED);
  const nSamples = 32;
  const x1 = [4.8, 3.8, 2.1, 5.4];

  const results = await explainer.explainOneInstance(x1, nSamples);
  const resultsExp = [
    [0.0745104, 0.02126633, 0.26081878, -0.4033925],
    [-0.06244084, -0.01309725, 0.05957105, -0.12841866],
    [-0.01206957, -0.00816908, -0.32038983, 0.53181117]
  ];

  for (let i = 0; i < resultsExp.length; i++) {
    for (let j = 0; j < resultsExp[i].length; j++) {
      expect(results[i][j]).toBeCloseTo(resultsExp[i][j], 6);
    }
  }
});
