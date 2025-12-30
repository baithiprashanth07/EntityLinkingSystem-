import os

class Config:
    # File paths
    TRAIN_TSV = 'train_pairs.tsv'
    VALIDATION_TSV = 'validation_pairs.tsv'
    TRAIN_CSV = 'processed_entity_linking_data_train.csv'
    VALIDATION_CSV = 'processed_entity_linking_data_validation.csv'
    KB_PATH = 'knowledge_base.csv'
    FAISS_INDEX_PATH = 'faiss_index.bin'
    ID_MAP_PATH = 'faiss_id_map.pkl'
    BASELINE_MODEL_PATH = 'finetuned_entity_linker.pkl'
    CROSS_ENCODER_PATH = 'finetuned_cross_encoder'
    
    # Model configurations
    BI_ENCODER_MODEL = 'all-MiniLM-L6-v2'
    CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    
    # NIL Handling (Future Work: Add NIL detection)
    NIL_ENTITY_ID = -1 # Special ID for Not-in-Knowledge-Base
    NIL_CONFIDENCE_THRESHOLD = 0.5 # Confidence score below which a mention is considered NIL
    NIL_ENTITY_NAME = "NIL_ENTITY_PLACEHOLDER" # Name for the NIL entity

    # Training parameters
    NUM_NEGATIVES = 5
    BATCH_SIZE = 16
    NUM_EPOCHS = 4
    K_CANDIDATES = 50
    CANDIDATE_TOP_N = 10
    
    # FAISS Scalability (Future Work: Replace flat FAISS with approximate ANN indexes)
    FAISS_NLIST = 100 # Number of clusters for IndexIVF
    FAISS_NPROBE = 10 # Number of clusters to search at query time

    # Execution flags - Set what you want to run
    RUN_PROCESSING = True
    RUN_EMBEDDING = True
    RUN_TRAINING = True
    RUN_BASELINE_EVAL = True
    RUN_DEEP_EVAL = True
    RUN_PREDICTIONS = True
    
    # Evaluation settings
    BASELINE_SAMPLE_SIZE = 10000  # Sample size for baseline evaluation
    FULL_VALIDATION = False  # Set True to evaluate on full validation set

    def __init__(self):
        # Print configuration summary
        print("="*80)
        print(" " * 20 + "ENTITY LINKING SYSTEM CONFIGURATION")
        print("="*80)
        print(f"- Bi-Encoder Model: {self.BI_ENCODER_MODEL}")
        print(f"- Cross-Encoder Model: {self.CROSS_ENCODER_MODEL}")
        print(f"- Training Epochs: {self.NUM_EPOCHS}")
        print(f"- Batch Size: {self.BATCH_SIZE}")
        print(f"- FAISS Index Type: IndexIVFFlat (nlist={self.FAISS_NLIST}, nprobe={self.FAISS_NPROBE})")
        print(f"- NIL Threshold: {self.NIL_CONFIDENCE_THRESHOLD}")
        print("="*80)

# Instantiate Config for easy import
config = Config()
