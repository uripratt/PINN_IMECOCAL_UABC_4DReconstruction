"""Resume los results_*.json de experiments/: ordena por VAL (la única métrica válida para elegir)
y muestra el TEST (que solo se mira, no se usa para elegir). Uso: python summarize_results.py [glob]"""
import glob
import json
import os
import sys

here = os.path.dirname(__file__)
pattern = sys.argv[1] if len(sys.argv) > 1 else "results_*.json"
rows = []
for f in sorted(glob.glob(os.path.join(here, pattern))):
    r = json.load(open(f))
    rows.append((r['val']['skill_clim'], r))

print(f"{'run':46s} {'ép*':>4s} {'val_skill':>9s} {'test_skill':>10s} {'test_skill_media':>16s} {'test_RMSE_log10':>15s} {'botella':>8s}")
for _, r in sorted(rows, key=lambda t: -t[0]):
    v, t = r['val'], r['test']
    print(f"{r['run_name'][:46]:46s} {r['best_epoch']:4d} {v['skill_clim']:+9.3f} {t['skill_clim']:+10.3f} "
          f"{t['skill_const']:+16.3f} {t['rmse_log10']:15.3f} {t['rmse_log10_bottle']:8.3f}")
print("\nskill_clim = 1 - MSE_modelo/MSE_climatología_de_train (>0: mejor que la climatología). "
      "ép* = mejor época elegida sobre val suavizado.")
