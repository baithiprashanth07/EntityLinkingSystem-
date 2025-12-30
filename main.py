import pandas as pd
import time
import os

# Import all modular components
from config import Config
from kpi_calculator import KPICalculator
from data_processor import DataProcessor, EmbeddingGenerator
from linker_base import CrossEncoderTrainer, BaselineEntityLinker
from linker_deep import DeepEntityLinker

def main():
    """Main execution function for the Entity Linking Pipeline."""
    
    print("\n\n" + "="*80)
    print(" " * 28 + "STARTING ENTITY LINKING PIPELINE")
    print("="*80)
    
    # STEP 1: Data Processing
    if Config.RUN_PROCESSING:
        DataProcessor.run_full_processing()
    else:
        print("\n⏭️  Skipping data processing")

    # STEP 2: Embedding Generation
    if Config.RUN_EMBEDDING:
        EmbeddingGenerator.generate_embeddings_and_index()
    else:
        print("\n⏭️  Skipping embedding generation")

    # STEP 3: Cross-Encoder Training
    if Config.RUN_TRAINING:
        CrossEncoderTrainer.run_full_training()
    else:
        print("\n⏭️  Skipping cross-encoder training")

    # STEP 4: Baseline Evaluation
    baseline_kpis = None
    if Config.RUN_BASELINE_EVAL:
        print("\n" + "="*80)
        print(" " * 25 + "STEP 4: BASELINE EVALUATION")
        print("="*80)
        baseline_linker = BaselineEntityLinker()
        baseline_linker.load_knowledge_base()
        baseline_kpis = baseline_linker.evaluate(sample_size=Config.BASELINE_SAMPLE_SIZE)
    else:
        print("\n⏭️  Skipping baseline evaluation")

    # STEP 5: Deep Linker Evaluation
    deep_kpis = None
    if Config.RUN_DEEP_EVAL:
        print("\n" + "="*80)
        print(" " * 23 + "STEP 5: DEEP LINKER EVALUATION")
        print("="*80)
        deep_linker = DeepEntityLinker()
        deep_kpis = deep_linker.evaluate()
    else:
        print("\n⏭️  Skipping deep linker evaluation")

    # STEP 6: Example Predictions
    if Config.RUN_PREDICTIONS:
        print("\n" + "="*80)
        print(" " * 25 + "STEP 6: EXAMPLE PREDICTIONS")
        print("="*80)
        
        test_mentions = [
            "Aspirin", 
            "acetylsalicylic acid", 
            "Tylenol", 
            "Paracetamol",
            "acetaminophen",
            "ibuprofen",
            "non-existent drug name" # Test NIL handling
        ]
        
        # --- BASELINE PREDICTIONS ---
        print("\n🔍 BASELINE PREDICTIONS:")
        print("-" * 80)
        
        # Initialize linker if not already done
        if not Config.RUN_BASELINE_EVAL:
            baseline_linker = BaselineEntityLinker()
            baseline_linker.load_knowledge_base()
            
        results_baseline = []
        for mention in test_mentions:
            entity_id, score, canonical = baseline_linker.link_entity(mention)
            results_baseline.append({
                'Mention': mention,
                'Entity_ID': entity_id,
                'Canonical_Name': canonical,
                'Score': f"{score:.2f}"
            })
        df_baseline = pd.DataFrame(results_baseline)
        print(df_baseline.to_string(index=False))
        
        # --- DEEP LEARNING PREDICTIONS ---
        print("\n\n🚀 DEEP LEARNING PREDICTIONS (Batched & Cached):")
        print("-" * 80)
        
        # Initialize linker if not already done
        if not Config.RUN_DEEP_EVAL:
            deep_linker = DeepEntityLinker()
            
        # Use the batched method for the test set
        predictions, scores = deep_linker.link_entities_batched(test_mentions)
        
        results_deep = []
        for mention, predicted_id, score in zip(test_mentions, predictions, scores):
            canonical = None
            if predicted_id == Config.NIL_ENTITY_ID:
                canonical = Config.NIL_ENTITY_NAME
            elif predicted_id is not None:
                try:
                    canonical = deep_linker.kb_df[deep_linker.kb_df['Entity_ID'] == predicted_id]['Canonical_Name'].iloc[0]
                except IndexError:
                    canonical = "UNKNOWN_ENTITY"
            
            results_deep.append({
                'Mention': mention,
                'Entity_ID': predicted_id,
                'Canonical_Name': canonical,
                'Score': f"{score:.4f}"
            })
        df_deep = pd.DataFrame(results_deep)
        print(df_deep.to_string(index=False))

    # Final Summary
    print("\n\n" + "="*80)
    print(" " * 28 + "PIPELINE COMPLETE! 🎉")
    print("="*80)
    
    if baseline_kpis and deep_kpis:
        print("\n" + "="*80)
        print(" " * 28 + "COMPREHENSIVE COMPARISON")
        print("="*80)
        
        comparison_data = {
            'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'Coverage'],
            'Baseline': [
                f"{baseline_kpis['precision']:.4f}",
                f"{baseline_kpis['recall']:.4f}",
                f"{baseline_kpis['f1_score']:.4f}",
                f"{baseline_kpis['accuracy']:.4f}",
                f"{baseline_kpis['coverage']:.4f}"
            ],
            'Deep Learning': [
                f"{deep_kpis['precision']:.4f}",
                f"{deep_kpis['recall']:.4f}",
                f"{deep_kpis['f1_score']:.4f}",
                f"{deep_kpis['accuracy']:.4f}",
                f"{deep_kpis['coverage']:.4f}"
            ],
            'Improvement': [
                f"{(deep_kpis['precision'] - baseline_kpis['precision'])*100:+.2f}%",
                f"{(deep_kpis['recall'] - baseline_kpis['recall'])*100:+.2f}%",
                f"{(deep_kpis['f1_score'] - baseline_kpis['f1_score'])*100:+.2f}%",
                f"{(deep_kpis['accuracy'] - baseline_kpis['accuracy'])*100:+.2f}%",
                f"{(deep_kpis['coverage'] - baseline_kpis['coverage'])*100:+.2f}%"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("\n🎯 KEY INSIGHTS:")
        
        # Determine winner for each metric
        if deep_kpis['f1_score'] > baseline_kpis['f1_score']:
            improvement = (deep_kpis['f1_score'] - baseline_kpis['f1_score']) * 100
            print(f"   ✅ Deep Learning outperforms Baseline by {improvement:.2f}% in F1-Score")
        
        if deep_kpis['precision'] > baseline_kpis['precision']:
            print(f"   ✅ Deep Learning has better precision (fewer false positives)")
        
        if deep_kpis['recall'] > baseline_kpis['recall']:
            print(f"   ✅ Deep Learning has better recall (fewer false negatives)")
        
        print("\n" + "="*80)

    print("\n✅ Modularized files created:")
    print("   • config.py")
    print("   • kpi_calculator.py")
    print("   • data_processor.py (includes EmbeddingGenerator)")
    print("   • linker_base.py (includes CrossEncoderTrainer and BaselineEntityLinker)")
    print("   • linker_deep.py (includes DeepEntityLinker)")
    print("   • main.py (orchestrates the pipeline)")
    
    print("\n" + "="*80)
    print("\n📚 KPI DEFINITIONS:")
    print("-" * 80)
    print("• Precision = TP / (TP + FP)")
    print("  → Of all predictions made, how many were correct?")
    print("  → High precision = Few false positives")
    print()
    print("• Recall = TP / (TP + FN)")
    print("  → Of all correct entities, how many did we find?")
    print("  → High recall = Few false negatives")
    print()
    print("• F1-Score = 2 × (Precision × Recall) / (Precision + Recall)")
    print("  → Harmonic mean of precision and recall")
    print("  → Best overall measure of performance")
    print()
    print("• Accuracy = TP / Total Samples")
    print("  → Simple percentage of correct predictions")
    print()
    print("• Coverage = Predictions Made / Total Samples")
    print("  → Percentage of samples that received predictions")
    print("-" * 80)
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
