"""MODEL BAKE-OFF for the 1.5x-in-95-sessions target (companion to EXP-2026-09-27-1p5x-anatomy).

Question (user 2026-09-27): is LightGBM the right tool, or is something more powerful needed?
Same rows (logs/leader_sleeve/anatomy_1p5x/rows.parquet), same features, same walk-forward folds
(train on rows dated < Y-01-01 minus 150 days so every training target window is closed; predict year Y).
Contenders:
  logit      standardised logistic regression (median-imputed)            — linear baseline
  lgbm       LightGBM binary (same params as the anatomy run)
  xgb        XGBoost hist binary
  mlp        sklearn MLP 128-64, quantile-transformed inputs, early stop   — neural-net proxy
  lgbm_rank  LightGBM LambdaRank, one query per week                      — optimises within-week order
  ensemble   rank-average of lgbm, xgb, mlp
Training subsample for logit/mlp: 300k rows (speed); tree models use all rows.
Metrics by era (disc = 2019-2022 test years, conf = 2023+): AUC, top-1% precision & lift,
weekly top-10 TRADABLE (ADV>=5cr & close>50) hit rate — the number a portfolio actually trades.
Output: logs/leader_sleeve/anatomy_1p5x/bakeoff.json + stdout log.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path("/Users/abhinavs./Documents/Zoom")
D = ROOT / "logs/leader_sleeve/anatomy_1p5x"
S = pd.read_parquet(D / "rows.parquet")
META = {"symbol", "trade_date", "era", "close", "mcap_cr", "adv", "ind", "core", "y95", "y63", "s95", "fh95", "fc95", "pred"}
FEATS = [c for c in S.columns if c not in META]
S["year"] = S["trade_date"].dt.year
L = S.dropna(subset=["y95"]).copy()
print(f"rows {len(L):,} · features {len(FEATS)}", flush=True)

import lightgbm as lgb
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

preds = {k: pd.Series(np.nan, index=L.index) for k in ["logit", "lgbm", "xgb", "mlp", "lgbm_rank"]}
for Y in sorted(y for y in L["year"].unique() if y >= 2019):
    cut = pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=150)
    tr = L[L["trade_date"] < cut]; te = L[L["year"] == Y]
    if te.empty:
        continue
    sub = tr.sample(min(len(tr), 300_000), random_state=Y)
    lo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300, C=0.5))
    lo.fit(sub[FEATS], sub["y95"]); preds["logit"][te.index] = lo.predict_proba(te[FEATS])[:, 1]
    gb = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=63, min_child_samples=300, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    gb.fit(tr[FEATS], tr["y95"]); preds["lgbm"][te.index] = gb.predict_proba(te[FEATS])[:, 1]
    xg = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=50, tree_method="hist", eval_metric="auc", n_jobs=4)
    xg.fit(tr[FEATS], tr["y95"]); preds["xgb"][te.index] = xg.predict_proba(te[FEATS])[:, 1]
    mlp = make_pipeline(SimpleImputer(strategy="median"), QuantileTransformer(output_distribution="normal", subsample=100_000),
                        MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=1024, learning_rate_init=1e-3,
                                      max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=Y))
    mlp.fit(sub[FEATS], sub["y95"]); preds["mlp"][te.index] = mlp.predict_proba(te[FEATS])[:, 1]
    trs = tr.sort_values("trade_date")
    rk = lgb.LGBMRanker(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=300, subsample=0.8,
                        subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    rk.fit(trs[FEATS], trs["y95"].astype(int), group=trs.groupby("trade_date", sort=True).size().values)
    preds["lgbm_rank"][te.index] = rk.predict(te[FEATS])
    print(f"  fold {Y}: train {len(tr):,} test {len(te):,} done", flush=True)

O = L[preds["lgbm"].notna()].copy()
for k, v in preds.items():
    O[k] = v[O.index]
O["ensemble"] = O[["lgbm", "xgb", "mlp"]].rank(pct=True).mean(axis=1)
O["era2"] = np.where(O["year"] >= 2023, "conf", "disc")
res = {}
print("\nMODEL         era   AUC    top1%   lift  | weekly top-10 tradable: hit%  lift vs tradable base")
for k in ["logit", "lgbm", "xgb", "mlp", "lgbm_rank", "ensemble"]:
    for e in ("disc", "conf"):
        E = O[O["era2"] == e]
        auc = roc_auc_score(E["y95"], E[k])
        top1 = E.nlargest(len(E) // 100, k)["y95"].mean() * 100
        T = E[E["core"]]
        wk10 = T.sort_values(k, ascending=False).groupby("trade_date").head(10)["y95"].mean() * 100
        res[f"{k}|{e}"] = dict(auc=auc, top1=top1, lift1=top1 / (E["y95"].mean() * 100), wk10=wk10, wk10_lift=wk10 / (T["y95"].mean() * 100))
        print(f"{k:<12}  {e}  {auc:.3f}  {top1:5.1f}%  {top1/(E['y95'].mean()*100):4.1f}x  |  {wk10:5.1f}%  {wk10/(T['y95'].mean()*100):4.1f}x", flush=True)
json.dump(res, open(D / "bakeoff.json", "w"), indent=1)
print("BAKEOFF COMPLETE")
