export interface PredictionResult {
  prediction: string;
  confidence: number;
  label_index?: number;
  category?: string;
  /** Driver / road guidance (from labels.json `instruction`) */
  instruction?: string;
}