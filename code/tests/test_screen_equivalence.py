"""features={}인 BioGridX가 core.BioGrid와 같은 계산을 하는지 확인 (A0, A3, A4)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.language import make_task
from src.biogrid.core import BioGrid
from src.biogrid.screen import BioGridX

d = make_task("tag12", seed=0, n_train=2000, n_val=500, n_test=500)
for name, kw in [("A0", dict(output_signal="global", hidden_credit="global")),
                 ("A3", dict(output_signal="zone_target", hidden_credit="relay_w", gate="sensitivity")),
                 ("A4", dict(output_signal="zone_target", hidden_credit="relay_fa", gate="sensitivity"))]:
    a = BioGrid(39, 12, width=6, seed=3); b = BioGridX(39, 12, width=6, seed=3, features={})
    for i in range(0, 2000, 32):
        a.train_step(d["X_train"][i:i+32], d["y_train"][i:i+32], 0.5, **kw)
        b.train_step(d["X_train"][i:i+32], d["y_train"][i:i+32], 0.5, **kw)
    diff = max(np.abs(a.W_in - b.W_in).max(), np.abs(a.W_loc - b.W_loc).max(), np.abs(a.V - b.V).max())
    pa, pb = a.predict(d["X_val"]), b.predict(d["X_val"])
    print(name, "max param diff", diff, "same preds", bool((pa == pb).all()))
    assert diff < 1e-10 and (pa == pb).all()
print("OK")
