import sys
from pathlib import Path

# ensure `import phase3_reconstruction` and `import tools...` work from repo root
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
