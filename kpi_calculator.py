import numpy as np

class KPICalculator:
    """Calculates and prints key performance indicators for entity linking."""
    
    @staticmethod
    def calculate_kpis(true_labels, predicted_labels, predicted_scores, threshold=None):
        """
        Calculates standard entity linking KPIs (Precision, Recall, F1, Accuracy, Coverage).
        
        Args:
            true_labels (list): List of true entity IDs.
            predicted_labels (list): List of predicted entity IDs.
            predicted_scores (list): List of confidence scores for predictions.
            threshold (float, optional): Confidence threshold for filtering predictions.
        """
        
        # Convert to numpy arrays for easier comparison
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        # Apply threshold if provided (for NIL handling)
        if threshold is not None:
            predicted_scores = np.array(predicted_scores)
            # Set predicted_labels to None where score is below threshold
            predicted_labels[predicted_scores < threshold] = None
        
        # True Positives: Correct prediction (predicted ID matches true ID)
        tp = np.sum(predicted_labels == true_labels)
        
        # False Positives: Incorrect prediction (predicted ID is not None, but does not match true ID)
        fp = np.sum((predicted_labels != None) & (predicted_labels != true_labels))
        
        # False Negatives: Should have predicted but didn't (predicted None when should predict entity)
        # In the context of NIL handling, this is a bit more complex. 
        # For simplicity in this KPI calculation, we treat a non-match as a failure to link.
        # A more rigorous definition would depend on whether the true entity is in the KB.
        # Here, we assume all true_labels are in the KB (as per the data processing step).
        fn = np.sum(predicted_labels != true_labels) - fp
        
        # True Negatives: Not applicable in standard entity linking (we always have a true entity)
        tn = 0
        
        # Calculate KPIs
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = tp / len(true_labels) if len(true_labels) > 0 else 0.0
        
        # Additional metrics
        total_predictions = np.sum(predicted_labels != None)
        coverage = total_predictions / len(true_labels) if len(true_labels) > 0 else 0.0
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'coverage': coverage,
            'total_samples': len(true_labels),
            'total_predictions': int(total_predictions)
        }
    
    @staticmethod
    def print_kpis(kpis, model_name="Model"):
        """Pretty prints KPI metrics"""
        print(f"\n{'='*80}")
        print(f"{model_name:^80}")
        print('='*80)
        
        print("\n📊 Confusion Matrix Components:")
        print(f"   • True Positives (TP):   {kpis['true_positives']:,}")
        print(f"   • False Positives (FP):  {kpis['false_positives']:,}")
        print(f"   • False Negatives (FN):  {kpis['false_negatives']:,}")
        
        print("\n🎯 Performance Metrics:")
        print(f"   • Precision:  {kpis['precision']:.4f} ({kpis['precision']*100:.2f}%)")
        print(f"   • Recall:     {kpis['recall']:.4f} ({kpis['recall']*100:.2f}%)")
        print(f"   • F1-Score:   {kpis['f1_score']:.4f} ({kpis['f1_score']*100:.2f}%)")
        print(f"   • Accuracy:   {kpis['accuracy']:.4f} ({kpis['accuracy']*100:.2f}%)")
        print(f"   • Coverage:   {kpis['coverage']:.4f} ({kpis['coverage']*100:.2f}%)")
        
        print(f"\n📈 Summary:")
        print(f"   • Total Samples:      {kpis['total_samples']:,}")
        print(f"   • Total Predictions:  {kpis['total_predictions']:,}")
        
        print("="*80)
        
        # Interpretation
        print("\n💡 Interpretation:")
        if kpis['precision'] > 0.9:
            print("   ✅ HIGH PRECISION: Most predictions are correct")
        elif kpis['precision'] > 0.7:
            print("   ⚠️  MODERATE PRECISION: Some incorrect predictions")
        else:
            print("   ❌ LOW PRECISION: Many incorrect predictions")
        
        if kpis['recall'] > 0.9:
            print("   ✅ HIGH RECALL: Finding most correct entities")
        elif kpis['recall'] > 0.7:
            print("   ⚠️  MODERATE RECALL: Missing some correct entities")
        else:
            print("   ❌ LOW RECALL: Missing many correct entities")
        
        if kpis['f1_score'] > 0.9:
            print("   ✅ EXCELLENT F1: Great balance of precision and recall")
        elif kpis['f1_score'] > 0.7:
            print("   ⚠️  GOOD F1: Decent overall performance")
        else:
            print("   ❌ POOR F1: Needs improvement")
        
        print("="*80)
