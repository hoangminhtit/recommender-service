from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main import run_pipeline


def main() -> None:
    data_dir = CURRENT_DIR / "dwh_mock"
    run_pipeline(
        evaluate=True,
        skip_cache=True,
        local_data_dir=str(data_dir),
    )


if __name__ == "__main__":
    main()
