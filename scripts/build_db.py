"""Build (or rebuild) the mock master database: python scripts/build_db.py"""
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adreader import db  # noqa: E402
from adreader.config import DB_PATH  # noqa: E402

if __name__ == "__main__":
    db.build()
    with closing(db.connect()) as c:
        print(f"Built {DB_PATH}")
        for table, n in db.counts(c).items():
            print(f"  {table:<12} {n}")
