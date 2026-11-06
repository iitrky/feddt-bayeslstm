"""Build cells.pkl for run_np.py from the original NASA .mat and CALCE .xlsx files.
Usage: python scripts/build_cells.py --nasa DIR_WITH_B00xx.mat --calce DIR_WITH_CS2_xx_FOLDERS [--oxford FILE.mat] --out cells.pkl"""
import argparse, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feddt.data import load_calce_cs2, load_nasa, load_oxford

ap = argparse.ArgumentParser()
ap.add_argument("--nasa"); ap.add_argument("--calce"); ap.add_argument("--oxford"); ap.add_argument("--out", default="cells.pkl")
a = ap.parse_args()
recs = []
if a.nasa:
    recs += [load_nasa(os.path.join(a.nasa, f"{b}.mat")) for b in ("B0005", "B0006", "B0007", "B0018")]
if a.calce:
    recs += [load_calce_cs2(os.path.join(a.calce, c)) for c in ("CS2_35", "CS2_36", "CS2_37", "CS2_38")]
if a.oxford:
    recs += load_oxford(a.oxford)
pickle.dump(recs, open(a.out, "wb"))
print([(r.cell_id, len(r.soh)) for r in recs])
