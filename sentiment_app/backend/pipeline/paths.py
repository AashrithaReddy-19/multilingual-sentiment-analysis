"""Filesystem locations. Existing Telugu-only outputs are read-only for the whole project."""
from pathlib import Path

NLP_ROOT = Path(__file__).resolve().parents[3]                    # D:\NLP
APP_ROOT = NLP_ROOT / "sentiment_app"
INPUTS = NLP_ROOT / "multilingual_inputs"                          # verified inputs (audited 2026-10-01)
TELUGU_OUTPUTS = NLP_ROOT / "outputs"                              # completed Telugu-only run (never written)
MULTI = APP_ROOT / "artifacts" / "multilingual"                    # all new multilingual outputs
DATA = MULTI / "data"
MODEL_A = MULTI / "model_a"
MODEL_B = MULTI / "model_b"                                        # filled only by real Colab results
COLAB_PACKAGE = APP_ROOT / "artifacts" / "colab_package"
