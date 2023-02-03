/**
 * Common types.
 */

type FeatureType = 'cont' | 'cat';

export interface SHAPRow {
  index: number;
  shap: number;
  name: string;
  fullName: string;
}

export interface TabularContFeature {
  name: string;
  displayName: string;
  desc: string;
  value: number;
  requiresInt: boolean;
  requiresLog: boolean;
}

export interface TabularCatFeature {
  name: string;
  displayName: string;
  desc: string;
  levelInfo: {
    [key: string]: [string, string];
  };
  allLevels: CatLevel[];
  value: string;
}

interface CatLevel {
  level: string;
  displayName: string;
}

export interface TabularData {
  xTrain: number[][];
  yTrain: number[];
  xTest: number[][];
  yTest: number[];
  featureNames: string[];
  featureTypes: FeatureType[];
  featureInfo: { [key: string]: [string, string] };
  featureLevelInfo: {
    [key: string]: {
      [key: string]: [string, string];
    };
  };
  featureRequiresLog: string[];
  featureRequireInt: string[];
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

export interface Padding {
  top: number;
  bottom: number;
  left: number;
  right: number;
}
