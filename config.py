import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set ANTHROPIC_API_KEY in environment directly

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "dataset"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

# Dataset paths
TRIALS_CSV = DATASET_DIR / "trials_clean.csv"
CRITERIA_CHUNKS_CSV = DATASET_DIR / "eligibility_criteria_chunks.csv"
# Patient data: one folder per patient under patient-datasets/<id>/,
# each with patient_summary.md (the matching input) + chart / bundle / notes.
PATIENT_DATASETS_DIR = BASE_DIR / "patient-datasets"
PATIENT_SUMMARY_FILENAME = "patient_summary.md"

# Cache paths
LANCEDB_DIR = CACHE_DIR / "lancedb"
KEYWORDS_CACHE_PATH = CACHE_DIR / "keywords_cache.json"

# MedCPT HuggingFace model names
MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
MEDCPT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
EMBED_DIM = 768
MEDCPT_ARTICLE_MAX_LEN = 512
MEDCPT_QUERY_MAX_LEN = 256
MEDCPT_BATCH_SIZE = int(os.getenv("MEDCPT_BATCH_SIZE", "32"))

# Anthropic API
CLAUDE_MODEL_FAST = os.getenv("CLAUDE_MODEL_FAST", "claude-haiku-4-5-20251001")
CLAUDE_MODEL_SMART = os.getenv("CLAUDE_MODEL_SMART", "claude-sonnet-4-6")

# Retrieval settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
MEDCPT_TOP_N = 2000   # Candidates per condition query before RRF merge
RRF_K = 20            # RRF smoothing constant for per-condition result fusion

# Retrieval: max condition queries sent to MedCPT per patient (controls retrieval cost)
MAX_CONDITIONS_PER_PATIENT = int(os.getenv("MAX_CONDITIONS_PER_PATIENT", "12"))

# Matching: max criteria evaluated per trial (controls LLM cost)
MAX_CRITERIA_PER_TRIAL = 50

# Eligibility thresholds
# total_score = matching_score [-2,1] + agg_score [-2,2]
LIKELY_ELIGIBLE_THRESHOLD = 1.5
NOT_ELIGIBLE_THRESHOLD = -0.5

# Confidence threshold: likely_eligible demoted to needs_review below this
CONFIDENCE_REVIEW_THRESHOLD = float(os.getenv("CONFIDENCE_REVIEW_THRESHOLD", "0.6"))
