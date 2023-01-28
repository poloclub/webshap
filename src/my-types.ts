/**
 * A model that outputs a 1D vector (binary classification, regression)
 */
export type SHAPModel = (x: number[][]) => number[];
