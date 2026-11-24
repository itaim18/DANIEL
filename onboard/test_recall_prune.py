"""REC ALL's one destructive path: prune() deletes ONLY closed sessions the laptop verified, oldest first,
and stops as soon as there is room; recall_start() refuses when nothing may be pruned. Temp dirs only.

    python3 onboard/test_recall_prune.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cockpit as c  # noqa: E402


def make(name, closed, verified):
    d = c.EXP / "2026-10-16_sensors" / name
    d.mkdir(parents=True)
    if closed:
        (d / "summary.json").write_text("{}")
    if verified:
        (d / c.VERIFIED).write_text("ok")
    return d


c.EXP = Path(tempfile.mkdtemp())
old = make("rec_100000", closed=True, verified=True)       # oldest verified: the first to go
unv = make("rec_110000", closed=True, verified=False)      # closed, NOT verified: never
opn = make("rec_120000", closed=False, verified=True)      # open (marker but no summary): never
new = make("rec_130000", closed=True, verified=True)       # newer verified: only if still short
gone = lambda: sum(not d.exists() for d in (old, unv, opn, new))
c.MIN_FREE_GB = 10
c.free_gb = lambda: 5 + 10 * gone()                        # each deletion frees 10 GB
assert c.prune() is True and not old.exists() and new.exists() and unv.exists() and opn.exists()
c.free_gb = lambda: 0.0                                    # nothing ever frees enough
assert c.prune() is False and not new.exists() and unv.exists() and opn.exists()
assert "nothing laptop-verified" in c.recall_start() and c.S["recall"] is None
print("recall prune test: OK (verified+closed only, oldest first, stops when there is room, refuses after)")
