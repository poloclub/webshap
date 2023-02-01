/**
 * Weighted least square solver
 * @author: Jay Wang
 */

import math from '../../src/utils/math-import';
import { tensor2d } from '@tensorflow/tfjs';

/**
 * Solves linear least squares problems for given input matrix `x`,
 * target matrix `y`, and weight matrix `w`.
 * The solution is (X'WX)^(-1)X'WY
 *
 * @param x - The input matrix, with shape (m, n)
 * @param y - The target matrix, with shape (m, 1)
 * @param w - The weight matrix, with shape (m, 1) or (m, m)
 * @returns - The matrix with the shape (n, 1), representing the solution of
 * the linear least squares problem.
 * @throws Error - If x and y have different number of samples, y has more
 * than one column, or the size of w is neither (m ,1) nor (m, m).
 */
export const lstsq = (
  x: math.Matrix,
  y: math.Matrix,
  w: math.Matrix
): math.Matrix => {
  // Validate inputs
  if (x.size()[0] !== y.size()[0]) {
    throw Error('x and y have different number of samples.');
  }

  if (x.size()[0] !== w.size()[0]) {
    throw Error('x and w have different number of samples.');
  }

  if (y.size()[1] !== 1) {
    throw Error('y has more than one columns.');
  }

  if (w.size()[1] !== 1 && w.size()[0] !== w.size()[1]) {
    throw Error('The size of w is neither (m ,1) nor (m, m).');
  }

  // If w is a vector, we first transform it into a diagonal matrix
  let wMat = w;
  if (w.size()[1] === 1) {
    const values = math.squeeze(w).toArray();
    wMat = math.matrix(math.diag(values));
  }

  // Matrix multiplication is too slow in math.js, we use ml-matrix instead
  const xTensor = tensor2d(
    x.toArray() as number[][],
    x.size() as [number, number]
  );
  const wTensor = tensor2d(
    wMat.toArray() as number[][],
    wMat.size() as [number, number]
  );
  const yTensor = tensor2d(
    y.toArray() as number[][],
    y.size() as [number, number]
  );

  const left = xTensor.transpose().matMul(wTensor).matMul(xTensor);
  const right = xTensor.transpose().matMul(wTensor).matMul(yTensor);

  // Convert `left` back to math.js for inversion
  const left2D = left.arraySync() as number[][];
  const leftMat = math.matrix(left2D);
  const leftDet = math.det(leftMat);

  // Invertible matrix
  let leftInverse: math.Matrix;
  if (leftDet !== 0) {
    leftInverse = math.inv(leftMat);
  } else {
    // Singular matrix => we take pseudo-inverse instead
    console.warn('Matrix x is singular, use pseudo-inverse instead.');
    leftInverse = math.pinv(leftMat);
  }

  const leftInverseTensor = tensor2d(
    leftInverse.toArray() as number[][],
    leftInverse.size() as [number, number]
  );
  const result = leftInverseTensor.matMul(right);

  // Convert the result Matrix to math.Matrix
  const result2D = result.arraySync() as number[][];
  return math.matrix(result2D);
};
