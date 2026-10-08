"""
Fit and validate pitch-control parameters on SkillCorner open-data passes.

The question a community user needs answered: which pitch-control settings
should I use on SkillCorner broadcast tracking, and how good is the result?

1. Pass dataset: every completed/failed pass in all 20 matches, the frame it
   was played, and where the ball went (cached in data/derived/).
2. Match-grouped 5-fold cross-validation of seven model specs. Each fits
   t_react, sigma and ball speed (lambda fixed, see 4) on 16 matches and
   predicts the other 4. Specs differ in the top-speed rule (uniform 5 m/s,
   raw per-player v95, v95 with a position-group floor, goalkeepers at 5 m/s,
   both) and whether extrapolated players get their own timing uncertainty
   (sigma_extrap). Out-of-fold predictions are saved for later analysis.
3. Out-of-fold comparison with the literature defaults and with baselines
   fitted on the same folds (Voronoi, pass length, distance margin), plus
   match-level bootstrap intervals for the differences. The spec with the
   lowest out-of-fold log loss is selected and refitted on all matches.
4. Lambda profile for the selected spec: refit the others with lambda fixed
   from 4.3/s to 1000/s. The likelihood plateaus as lambda grows, so lambda is
   fixed at 40/s, the smallest round value within 0.001 log loss of the limit.
5. Robustness: by detection of the players nearest the destination, and with
   the state 0.3 s before the pass (smoothed velocities look 0.3 s ahead).

Run from the repo root:  py scripts/03_fit_pitch_control.py
Writes reports/w2_parameter_fit.md, figs/w2_fit_*.png and data/derived/pitch_control_*.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dsv import data as D, metrics as M, pitchcontrol as PC  # noqa: E402

DATA_DIR = Path(os.environ.get("DSV_DATA_DIR", ROOT / "data"))
FIGS, REPORTS, DERIVED = ROOT / "figs", ROOT / "reports", DATA_DIR / "derived"
CACHE = DERIVED / "pass_batch.npz"
FIT = ("t_react", "sigma", "ball_speed")
LAM = 40.0                     # 1/s, fixed from the lambda profile
LAM_PROFILE = (4.3, 10.0, 20.0, 40.0, 100.0, 1000.0)
INIT = replace(PC.PCParams(), lam=LAM)
DEFAULTS = "Spearman/Shaw defaults (uniform 5 m/s)"
LOGIT = "distance margin + length (logistic)"
VORONOI = "Voronoi (nearest player)"
LENGTH = "pass length (logistic)"
N_FOLDS = 5
N_BOOT = 2000
UNITS = dict(t_react="s", sigma="s", sigma_extrap="s", lam="1/s", ball_speed="m/s")


@dataclass
class Spec:
    name: str
    vmax: Optional[pd.Series]   # (match_id, player_id, period) -> vmax; None = 5 m/s for all
    fit: tuple
    rule: str                   # how to build the vmax table, for the report


def model_specs() -> list[Spec]:
    v95 = pd.read_csv(DERIVED / "v95_by_player_half.csv")
    key = ["match_id", "player_id", "period"]
    table = lambda **kw: PC.vmax_table(v95, **kw).set_index(key)["vmax"]  # noqa: E731
    raw = table(floor_quantile=None, gk_vmax=None)
    return [
        Spec("uniform 5 m/s", None, FIT, "5 m/s for everyone"),
        Spec("raw v95", raw, FIT, "vmax_table(v95, floor_quantile=None, gk_vmax=None)"),
        Spec("v95, group floor", table(floor_quantile=0.25, gk_vmax=None), FIT,
             "vmax_table(v95, floor_quantile=0.25, gk_vmax=None)"),
        Spec("v95, GK 5 m/s", table(floor_quantile=None, gk_vmax=5.0), FIT,
             "vmax_table(v95, floor_quantile=None, gk_vmax=5.0)"),
        Spec("v95, group floor, GK 5 m/s", table(), FIT, "vmax_table(v95)"),
        Spec("raw v95 + sigma_extrap", raw, FIT + ("sigma_extrap",),
             "vmax_table(v95, floor_quantile=None, gk_vmax=None)"),
        Spec("v95, group floor + sigma_extrap", table(floor_quantile=0.25, gk_vmax=None),
             FIT + ("sigma_extrap",), "vmax_table(v95, floor_quantile=0.25, gk_vmax=None)"),
    ]


# --------------------------------------------------------------------------- #
# Pass dataset
# --------------------------------------------------------------------------- #
BATCH_FIELDS = ("pos", "vel", "is_gk", "is_home", "present", "ball", "target", "detected", "player_id")


def build_cache():
    """Pass rows for every match: batch at the pass frame (f0) and 0.3 s
    earlier (f3), plus labels. vmax is assigned later per spec."""
    parts = []
    for mid in D.list_open_matches(DATA_DIR):
        md = DATA_DIR / "matches" / str(mid)
        meta = D.load_match_metadata(md, mid)
        players, ball = D.normalize_direction(*D.load_tracking(md, mid, meta), meta)
        players = D.add_velocity(players)
        ps = D.passes_with_destination(D.load_events(md, mid), ball, meta)
        per = ball.set_index("frame")["period"]
        f3 = ps["frame_end"] - 3
        ps = ps[f3.isin(per.index) & (per.reindex(f3).to_numpy() == ps["period"].to_numpy())]
        dest = ps[["dest_x", "dest_y"]].to_numpy()
        b0 = PC.point_batch(players, ball, ps["frame_end"], dest, meta)
        b3 = PC.point_batch(players, ball, ps["frame_end"] - 3, dest, meta)
        part = {f"f0_{k}": getattr(b0, k) for k in BATCH_FIELDS}
        part.update({f"f3_{k}": getattr(b3, k) for k in BATCH_FIELDS})
        part.update(match_id=np.full(len(ps), mid), period=ps["period"].to_numpy(),
                    frame=ps["frame_end"].to_numpy(), passer_home=ps["passer_home"].to_numpy(),
                    success=ps["success"].to_numpy(), pass_length=ps["pass_length"].to_numpy(),
                    date=np.full(len(ps), meta.raw["date_time"][:10]))
        parts.append(part)
        print(f"  {mid}: {len(ps)} passes", flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(CACHE, **data)
    return data


def load_data():
    if CACHE.exists() and not os.environ.get("REBUILD"):
        with np.load(CACHE, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return build_cache()


def make_batch(data, prefix, vmax_by_key):
    """PointBatch with vmax filled from a (match_id, player_id, period) series,
    defaulting to 5 m/s (and for every player when the series is None)."""
    b = PC.PointBatch(vmax=None, **{k: data[f"{prefix}_{k}"] for k in BATCH_FIELDS})
    if vmax_by_key is None:
        b.vmax = np.full(b.present.shape, 5.0)
        return b
    n = b.present.shape[1]
    keys = pd.MultiIndex.from_arrays([np.repeat(data["match_id"], n), b.player_id.ravel(),
                                      np.repeat(data["period"], n)])
    b.vmax = vmax_by_key.reindex(keys).fillna(5.0).to_numpy().reshape(b.present.shape)
    return b


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def nearest_features(b: PC.PointBatch, passer_home):
    """Distance from the destination to the passer's nearest teammate and
    nearest opponent at the pass frame, and whether each is detected."""
    d = np.where(b.present, np.linalg.norm(b.pos - b.target[:, None, :], axis=-1), np.inf)
    mate = b.is_home == passer_home[:, None]
    dm, do = np.where(mate, d, np.inf), np.where(~mate, d, np.inf)
    im, io = dm.argmin(1), do.argmin(1)
    r = np.arange(len(d))
    return dict(d_mate=dm[r, im], d_opp=do[r, io], det_mate=b.detected[r, im], det_opp=b.detected[r, io])


def baseline_predictions(train, test, feats, data):
    y_tr = data["success"][train]
    vor = feats["d_mate"] < feats["d_opp"]
    rate = {v: y_tr[vor[train] == v].mean() for v in (True, False)}
    L = data["pass_length"]
    c_len = M.fit_logistic(np.column_stack([L, L ** 2])[train], y_tr)
    margin = np.clip(feats["d_opp"] - feats["d_mate"], -40, 40)
    X = np.column_stack([margin, L, np.sign(margin)])
    c_mar = M.fit_logistic(X[train], y_tr)
    return {
        VORONOI: np.where(vor[test], rate[True], rate[False]),
        LENGTH: M.predict_logistic(c_len, np.column_stack([L, L ** 2])[test]),
        LOGIT: M.predict_logistic(c_mar, X[test]),
    }


# --------------------------------------------------------------------------- #
# Cross-validation and bootstrap
# --------------------------------------------------------------------------- #
def folds(data):
    matches = pd.Series(data["date"], index=data["match_id"]).groupby(level=0).first().sort_values()
    fold_of = {m: i % N_FOLDS for i, m in enumerate(matches.index)}
    return np.array([fold_of[m] for m in data["match_id"]])


def cross_validate(data, specs, batches):
    fold = folds(data)
    y, home = data["success"], data["passer_home"]
    feats = nearest_features(batches[specs[0].name], home)
    oof = {DEFAULTS: PC.team_control_at_points(batches["uniform 5 m/s"], home, PC.PCParams(tol=1e-4))}
    oof.update({s.name: np.zeros(len(y)) for s in specs})
    oof.update({k: np.zeros(len(y)) for k in (VORONOI, LENGTH, LOGIT)})
    rows = []
    for k in range(N_FOLDS):
        tr, te = fold != k, fold == k
        for s in specs:
            b = batches[s.name]
            t0 = time.perf_counter()
            prm, res = PC.fit_params(b.subset(tr), home[tr], y[tr], init=INIT, fit=s.fit)
            oof[s.name][te] = PC.team_control_at_points(b.subset(te), home[te], prm)
            rows.append(dict(fold=k, spec=s.name, nfev=res.nfev, **{p: getattr(prm, p) for p in s.fit}))
            print(f"  fold {k} {s.name}: " + ", ".join(f"{p}={getattr(prm, p):.3f}" for p in s.fit)
                  + f" ({res.nfev} evals, {time.perf_counter() - t0:.0f} s)", flush=True)
        for name, pr in baseline_predictions(tr, te, feats, data).items():
            oof[name][te] = pr
    return oof, pd.DataFrame(rows), feats


def match_bootstrap(data, y, pa, pb, n=N_BOOT, seed=0):
    """Point estimate and 95% interval of log loss(a) - log loss(b),
    resampling whole matches."""
    def ll(p):
        p = np.clip(p, M.EPS, 1 - M.EPS)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    diff = pd.Series(ll(pa) - ll(pb)).groupby(data["match_id"]).agg(["sum", "size"])
    idx = np.random.default_rng(seed).integers(0, len(diff), size=(n, len(diff)))
    boot = diff["sum"].to_numpy()[idx].sum(1) / diff["size"].to_numpy()[idx].sum(1)
    return diff["sum"].sum() / diff["size"].sum(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def lambda_profile(b, home, y, fit):
    rows = []
    for lam in LAM_PROFILE:
        prm, res = PC.fit_params(b, home, y, init=replace(INIT, lam=lam), fit=fit)
        rows.append(dict(lam=lam, log_loss=res.fun, **{p: getattr(prm, p) for p in fit}))
        print(f"  lambda {lam:g}: log loss {res.fun:.4f}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def calibration_figure(y, oof, names):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for name, c in zip(names, ["#8a8f98", "#2b6cb0", "#dd8a2c", "#2f855a"]):
        r = M.reliability(y, oof[name], bins=10)
        r = r[r["n"] >= 30]
        ax.plot(r["p_mean"], r["rate"], "o-", ms=4, color=c, label=name)
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
    ax.set_xlabel("predicted completion probability (out of fold)")
    ax.set_ylabel("observed completion rate")
    ax.set_title("Pass completion calibration, 20 matches", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "w2_fit_calibration.png", dpi=130); plt.close(fig)


def profile_figure(b, home, y, best, fit, lam_prof):
    fig, axes = plt.subplots(1, len(fit) + 1, figsize=(3.1 * (len(fit) + 1), 3.0), sharey=True)
    for ax, name in zip(axes, fit):
        lo, hi = PC.FIT_BOUNDS[name]
        v0 = getattr(best, name)
        xs = np.unique(np.clip(np.linspace(v0 * 0.4, v0 * 1.8, 13) if v0 > 0.05
                               else np.linspace(lo, 0.6, 13), lo, hi))
        ll = [M.log_loss(y, PC.team_control_at_points(b, home, replace(best, tol=1e-4, **{name: float(x)})))
              for x in xs]
        ax.plot(xs, ll, "o-", ms=3, color="#2b6cb0")
        ax.axvline(v0, color="#c53030", lw=1)
        ax.set_xlabel(name); ax.grid(alpha=.3)
    ax = axes[-1]
    ax.semilogx(lam_prof["lam"], lam_prof["log_loss"], "o-", ms=3, color="#2f855a")
    ax.axvline(LAM, color="#c53030", lw=1)
    ax.set_xlabel("lam (others refitted)"); ax.grid(alpha=.3)
    axes[0].set_ylabel("log loss (all passes)")
    fig.suptitle("Likelihood profiles: one parameter varied, others at the fit "
                 "(right: lambda with the others refitted)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / "w2_fit_profiles.png", dpi=130); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    t_start = time.perf_counter()
    data = load_data()
    y = data["success"].astype(float)
    home = data["passer_home"].astype(bool)
    n_matches = len(np.unique(data["match_id"]))
    print(f"{len(y):,} passes from {n_matches} matches, completion {y.mean():.1%}", flush=True)

    specs = model_specs()
    batches = {s.name: make_batch(data, "f0", s.vmax) for s in specs}
    oof, fold_params, feats = cross_validate(data, specs, batches)
    fold_params.to_csv(DERIVED / "pitch_control_fold_params.csv", index=False)
    pd.DataFrame({"match_id": data["match_id"], "frame": data["frame"], "success": data["success"],
                  **oof}).to_csv(DERIVED / "pitch_control_oof.csv", index=False)
    table = pd.DataFrame([dict(model=k, **M.summary(y, p)) for k, p in oof.items()]).sort_values("log_loss")
    print(table.round(4).to_string(index=False), flush=True)

    sel = min(specs, key=lambda s: M.log_loss(y, oof[s.name]))
    print(f"selected: {sel.name}", flush=True)
    boot = {f"vs {other}": match_bootstrap(data, y, oof[sel.name], oof[other])
            for other in [DEFAULTS, LOGIT, VORONOI] + [s.name for s in specs if s is not sel]}

    final = {}
    for s in specs:
        prm, _ = PC.fit_params(batches[s.name], home, y, init=INIT, fit=s.fit)
        final[s.name] = prm
        print(f"  final {s.name}: " + ", ".join(f"{p}={getattr(prm, p):.3f}" for p in s.fit), flush=True)
    best = final[sel.name]
    b_sel = batches[sel.name]
    lam_prof = lambda_profile(b_sel, home, y, sel.fit)
    lam_prof.to_csv(DERIVED / "pitch_control_lambda_profile.csv", index=False)

    det = np.where(feats["det_mate"] & feats["det_opp"], "both detected",
                   np.where(feats["det_mate"] | feats["det_opp"], "one extrapolated", "both extrapolated"))
    strata_models = list(dict.fromkeys(
        [sel.name, "raw v95", "raw v95 + sigma_extrap", "v95, group floor",
         "v95, group floor + sigma_extrap", LOGIT]))
    strata = pd.DataFrame([dict(stratum=st, model=m, **M.summary(y[det == st], oof[m][det == st]))
                           for st in ["both detected", "one extrapolated", "both extrapolated"]
                           for m in strata_models if (det == st).sum() >= 50])
    p_all = PC.team_control_at_points(b_sel, home, best)
    p3 = PC.team_control_at_points(make_batch(data, "f3", sel.vmax), home, best)
    lead = pd.DataFrame([dict(frame="pass frame", **M.summary(y, p_all)),
                         dict(frame="0.3 s before the pass", **M.summary(y, p3))])

    profile_figure(b_sel, home, y, best, sel.fit, lam_prof)
    calibration_figure(y, oof, [DEFAULTS, sel.name, LOGIT, VORONOI])

    fitted = {k: getattr(best, k) for k in sel.fit}
    out = dict(spec=sel.name, vmax_rule=sel.rule, params={**fitted, "lam": LAM},
               fixed={k: v for k, v in asdict(best).items() if k not in fitted and k != "lam"},
               n_passes=int(len(y)), n_matches=int(n_matches))
    (DERIVED / "pitch_control_fit.json").write_text(json.dumps(out, indent=2))
    write_report(data, y, specs, sel, table, fold_params, boot, final, strata, lead, lam_prof)
    print(f"done in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)


def write_report(data, y, specs, sel, table, fold_params, boot, final, strata, lead, lam_prof):
    L = []
    w = L.append
    best = final[sel.name]
    n_m = len(np.unique(data["match_id"]))
    d = PC.PCParams()
    w("# Fitting pitch control to SkillCorner open data\n")
    w(f"Spearman-style pitch control (`dsv/pitchcontrol.py`) fitted to **{len(y):,} passes** from all {n_m} "
      f"SkillCorner open-data matches (A-League 2024/25, broadcast tracking, 10 fps). Pass completion rate "
      f"{y.mean():.1%}. The model's completion probability is the passing team's control at the ball's "
      "destination, evaluated at the frame the pass is played.\n")

    w("## Recommended settings\n")
    w(f"Selected by out-of-fold log loss: **{sel.name}**. Top speed: `{sel.rule}`.\n")
    w("| parameter | Spearman/Shaw default | fitted | fold-to-fold range |")
    w("|---|---|---|---|")
    fp = fold_params[fold_params["spec"] == sel.name]
    for p in sel.fit:
        dv = getattr(d, p)
        w(f"| {p} ({UNITS[p]}) | {'= sigma' if dv is None else f'{dv:g}'} | **{getattr(best, p):.3f}** | "
          f"{fp[p].min():.3f} – {fp[p].max():.3f} |")
    w(f"| lam (1/s) | {d.lam:g} | **{LAM:g}** (fixed, see below) | – |")
    w(f"\nGoalkeeper control rate x{best.lam_gk_mult:g}. In code: `dsv.pitchcontrol.SKILLCORNER_OPEN_DATA`; "
      "for tracking without detection flags, `SKILLCORNER_OPEN_DATA_NO_FLAGS` (the floor-only spec's own fit).\n")

    w("## Out-of-fold performance\n")
    w(f"Match-grouped {N_FOLDS}-fold cross-validation: every number comes from matches the parameters (or "
      "baseline) never saw. Lower log loss, Brier and ECE are better; higher AUC is better.\n")
    w("| model | log loss | Brier | AUC | ECE |")
    w("|---|---|---|---|---|")
    for _, r in table.iterrows():
        name = f"**{r['model']}** (selected)" if r["model"] == sel.name else r["model"]
        w(f"| {name} | {r['log_loss']:.4f} | {r['brier']:.4f} | {r['auc']:.3f} | {r['ece']:.3f} |")
    w("\nThe seven fitted specs differ only in top speed and in `sigma_extrap`. Baselines are fitted on the same "
      f"training folds: *{VORONOI}* uses the completion rate of passes whose destination is nearest a teammate "
      f"vs an opponent; *{LOGIT}* is a logistic regression on (nearest-opponent distance − nearest-teammate "
      "distance), its sign, and pass length.\n")

    w("### Selected spec minus each alternative, 95% match-bootstrap intervals\n")
    w(f"Out-of-fold log loss difference (negative favours the selected spec), resampling the {n_m} matches "
      f"{N_BOOT:,} times.\n")
    w("| comparison | Δ log loss | 95% interval |")
    w("|---|---|---|")
    for k, (pt, lo, hi) in boot.items():
        w(f"| {k} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    w("")

    w("## Top speed: what helps\n")
    w("Final fits on all passes, one per spec:\n")
    w("| spec | top speed | t_react | sigma | sigma_extrap | ball speed |")
    w("|---|---|---|---|---|---|")
    for s in specs:
        prm = final[s.name]
        se = f"{prm.sigma_extrap:.3f}" if prm.sigma_extrap is not None else "–"
        w(f"| {s.name} | `{s.rule}` | {prm.t_react:.3f} | {prm.sigma:.3f} | {se} | {prm.ball_speed:.2f} |")
    w("")

    w("## Control rate lambda is only bounded from below\n")
    w(f"Selected spec, lambda fixed and the other parameters refitted on all passes:\n")
    w("| lambda (1/s) | log loss | " + " | ".join(sel.fit) + " |")
    w("|---|---|" + "---|" * len(sel.fit))
    for _, r in lam_prof.iterrows():
        w(f"| {r['lam']:g} | {r['log_loss']:.4f} | " + " | ".join(f"{r[p]:.3f}" for p in sel.fit) + " |")
    w("\nThe fit keeps improving as lambda grows and flattens out: the data prefer the limit where the ball "
      "goes to whoever arrives first, with sigma carrying the timing uncertainty. Part of this is built into "
      "the test, since a pass's destination is where someone actually controlled the ball. Lambda is fixed at "
      f"{LAM:g}/s, the smallest round value within about 0.001 log loss of the plateau, which keeps the "
      "goalkeeper's faster control meaningful. It was chosen before cross-validation on all passes, a "
      "one-parameter choice that leaks slightly into the out-of-fold numbers.\n")

    w("## Robustness\n")
    w("**Extrapolated positions.** Passes split by whether the passer's nearest teammate and nearest opponent "
      "to the destination were detected (out-of-fold predictions):\n")
    w("| nearest players | model | passes | log loss | AUC | ECE |")
    w("|---|---|---|---|---|---|")
    for _, r in strata.iterrows():
        w(f"| {r['stratum']} | {r['model']} | {r['n']:,} | {r['log_loss']:.4f} | {r['auc']:.3f} | {r['ece']:.3f} |")
    fe = final[sel.name]
    if fe.sigma_extrap is not None:
        w(f"\nWith `sigma_extrap`, extrapolated players get {fe.sigma_extrap / fe.sigma - 1:.0%} more "
          f"timing uncertainty than detected ones ({fe.sigma_extrap:.3f} s vs {fe.sigma:.3f} s). It helps "
          "most when both nearest players are extrapolated (where it overtakes the logistic baseline) and "
          "slightly hurts when exactly one is. Overall its gain over the same spec without it is small; "
          "see the bootstrap table.")
    w("\n**Look-ahead.** Velocities are Savitzky-Golay smoothed over ±0.3 s, so the pass-frame state uses 0.3 s "
      "of future positions. Evaluating on the state 0.3 s *before* the pass removes that (final parameters, "
      "in-sample):\n")
    w("| state used | log loss | AUC | ECE |")
    w("|---|---|---|---|")
    for _, r in lead.iterrows():
        w(f"| {r['frame']} | {r['log_loss']:.4f} | {r['auc']:.3f} | {r['ece']:.3f} |")
    w("")

    w("## Caveats\n")
    w("- The destination of a failed pass is where an opponent took the ball, which makes the test easier than "
      "predicting the intended target. This is the standard set-up (Spearman 2017), and the baselines get the "
      "same advantage.")
    w("- Passes that went out of play, or whose next possession started more than 5 s later, are excluded.")
    w(f"- The selected spec is the best of {len(specs)} on the same folds, so its out-of-fold score is slightly "
      "optimistic. The bootstrap intervals show which differences are larger than that.")
    w("- One season of one league, broadcast tracking with about 57% of positions detected. Fitted parameters "
      "absorb some tracking error and may not transfer to optical tracking.")
    w("- The fitted surface is a control share, not a calibrated pass probability: out of fold it is "
      "underconfident between about 0.5 and 0.8 (see the calibration figure), where the logistic baseline "
      "is better calibrated. If you need pass probabilities, add a recalibration layer (e.g. a logistic "
      "on logit(control)) fitted on your own training folds.")
    w("- The parameters describe *pass reception*. Using them for space control elsewhere on the pitch "
      "(Threat Conceded) is an assumption.\n")
    w("## Figures\n")
    w("![calibration](../figs/w2_fit_calibration.png)\n")
    w("![profiles](../figs/w2_fit_profiles.png)\n")
    w("Reproduce: `py scripts/03_fit_pitch_control.py` (after `scripts/01_data_audit.py`, which writes v95). "
      "Outputs: `data/derived/pitch_control_fit.json`, `pitch_control_fold_params.csv`, "
      "`pitch_control_lambda_profile.csv`.")
    (REPORTS / "w2_parameter_fit.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
