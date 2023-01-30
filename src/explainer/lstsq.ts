/**
 * Weighted least square solver
 * @author: Jay Wang
 */

import math from '../../src/utils/math-import';

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

  // Solve the least square through linear equation
  const left = math.multiply(math.multiply(math.transpose(x), wMat), x);

  const leftDet = math.det(left);
  let leftInverse: math.Matrix;
  // Invertible matrix
  if (leftDet !== 0) {
    leftInverse = math.inv(left);
  } else {
    // Singular matrix => we take pseudo-inverse instead
    console.warn('Matrix x is singular, use pseudo-inverse instead.');
    leftInverse = math.pinv(left);
  }
  const right = math.multiply(math.multiply(math.transpose(x), wMat), y);

  return math.multiply(leftInverse, right);
};
