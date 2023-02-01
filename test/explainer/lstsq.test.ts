import { describe, test, expect, beforeEach } from 'vitest';
import { lstsq } from '../../src/explainer/lstsq';
import math from '../../src/utils/math-import';

interface LocalTestContext {
  x: math.Matrix;
  y: math.Matrix;
  w: math.Matrix;
}

/**
 * Initialize the fixture for all tests
 */
beforeEach<LocalTestContext>(context => {
  context.x = math.reshape(
    math.matrix([
      6.5883575, 9.47744716, 6.38128927, 1.32898554, 1.42451075, 8.07888215,
      1.37843823, 5.01048122, 5.91789496, 6.74303087, 9.71877437, 2.68876159,
      2.15260096, 0.67584554, 8.76868181, 0.69520003, 3.27889612, 3.53383915,
      0.39004321, 8.98698661, 5.94193232, 5.61643321, 5.80469669, 9.02401475,
      9.45376559, 4.79489526, 1.03856608, 4.44841592, 5.27924568, 6.81838066,
      7.13956214, 6.20247773, 9.34432887, 9.58139555, 9.94923107, 3.43236642,
      2.67356961, 1.87084952, 7.85087359, 0.09172617, 3.50793606, 1.1197608,
      3.40025746, 8.02584101, 5.8887118, 8.23636548, 1.32038954, 6.14270219,
      0.13882923, 7.53383955
    ]),
    [10, 5]
  );

  context.y = math.reshape(
    math.matrix([
      61.767112, 14.04187055, 30.88552799, 51.9603377, 29.28888286, 86.85199145,
      73.45540838, 46.08049002, 43.68663906, 48.23927507
    ]),
    [10, 1]
  );

  context.w = math.reshape(
    math.matrix([
      0.14251289, 0.03092725, 0.07816275, 0.13553988, 0.02129188, 0.13268995,
      0.05887923, 0.15102889, 0.11071796, 0.13824932
    ]),
    [10, 1]
  );
});

test<LocalTestContext>('lstsq() without weight', ({ x, y }) => {
  const w = math.matrix(math.ones([10, 1]));
  const result = lstsq(x, y, w);
  const resultExp = math.reshape(
    math.matrix([-0.38057159, 1.35998903, 6.28966952, 0.75159811, 1.20430188]),
    [5, 1]
  );

  result.forEach((v, i) => {
    const curI = i as unknown as [number, number];
    expect(v).toBeCloseTo(resultExp.get(curI) as number, 4);
  });
});

test<LocalTestContext>('lstsq() diagonal w', ({ x, y }) => {
  const w = math.matrix(math.diag(new Array(10).fill(1)));
  const result = lstsq(x, y, w);
  const resultExp = math.reshape(
    math.matrix([-0.38057159, 1.35998903, 6.28966952, 0.75159811, 1.20430188]),
    [5, 1]
  );

  result.forEach((v, i) => {
    const curI = i as unknown as [number, number];
    expect(v).toBeCloseTo(resultExp.get(curI) as number, 4);
  });
});

test<LocalTestContext>('lstsq() weight', ({ x, y, w }) => {
  const result = lstsq(x, y, w);
  const resultExp = math.reshape(
    math.matrix([-0.28239368, 1.83192929, 5.51801203, 2.44676581, 1.94097814]),
    [5, 1]
  );

  result.forEach((v, i) => {
    const curI = i as unknown as [number, number];
    expect(v).toBeCloseTo(resultExp.get(curI) as number, 4);
  });
});
