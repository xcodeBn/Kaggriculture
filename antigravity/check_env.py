import json
import sys
from pathlib import Path

print("Python executable:", sys.executable)
try:
    import kaggle_environments
    print("kaggle_environments successfully imported from:", kaggle_environments.__file__)
except Exception as e:
    print("kaggle_environments import error:", e)
