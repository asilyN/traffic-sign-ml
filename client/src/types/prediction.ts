export interface OtherPrediction {
  class_id: number;
  class_name: string;
  confidence: number;
}

export interface PredictionResult {
  prediction: string;
  confidence: number;
  label_index?: number;
  other_predictions?: OtherPrediction[];
}