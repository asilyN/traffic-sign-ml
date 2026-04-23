export interface PredictionResult {
  prediction: string;
  confidence: number;
  label_index?: number;
}