"""Print environment versions for reproducibility (doc section 39)."""
import platform
import sys

print(f"python={platform.python_version()}")
try:
    import torch
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
except Exception as e:
    print(f"torch=NOT_INSTALLED ({e})")
    sys.exit(1)

for pkg in ["datasets", "tokenizers", "sentencepiece", "datatrove", "pyarrow", "polars", "yaml"]:
    try:
        m = __import__(pkg)
        print(f"{pkg}={getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"{pkg}=MISSING ({e})")
