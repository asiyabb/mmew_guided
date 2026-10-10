import os

class Config:
    DATA_ROOT = "/home/ab3379/work/data/MMEW_Final"
    MACRO_DIR = os.path.join(DATA_ROOT, "Macro_Expression")
    MICRO_DIR = os.path.join(DATA_ROOT, "Micro_Expression")
    EXCEL_ANNOTATIONS = os.path.join(DATA_ROOT, "MMEW_Micro_Exp.xlsx")

    EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    NUM_CLASSES = len(EMOTIONS)
    LABEL_MAP = {e: i for i, e in enumerate(EMOTIONS)}

    # Sequence and Window Specs
    SEQ_LEN = 16
    WINDOW_SIZE = 4
    WINDOW_STRIDE = 2
    IMAGE_SIZE = (224, 224)

    # Training Specs
    BATCH_SIZE = 8
    EPOCHS = 30
    LR = 1e-4

    # --- BALANCED REGULARIZATION WEIGHTS ---
    ALPHA_TRIPLET = 0.05
    BETA_TEMPORAL = 0.05

    DEVICE = "cuda"