"""Fit separate cut-gain and fragmentation classifiers on frozen R23 map splits."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_logistic(x, y, balanced=False):
    weights = np.ones(len(y))
    if balanced:
        weights = np.where(y, .5/y.mean(), .5/(1-y.mean()))
    def objective(coef):
        z = x @ coef[:-1] + coef[-1]
        residual = (expit(z)-y)*weights/len(y)
        loss = np.mean(weights*(np.logaddexp(0,z)-y*z)) + .005*(coef[:-1]**2).sum()
        grad = np.r_[x.T @ residual + .01*coef[:-1], residual.sum()]
        return loss, grad
    result = minimize(objective, np.zeros(x.shape[1]+1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def score_choices(raw, model):
    x = (raw-model["mean"])/model["std"]
    gain = x @ model["gain"][:-1] + model["gain"][-1]
    harm = expit(x @ model["harm"][:-1] + model["harm"][-1])
    return gain - 3*harm


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("r23", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    with np.load(args.r23/"MAP_FIT.npz") as data:
        x, y = data["x"].astype(float), data["y"]
    known = y[:,1] >= 0
    y = y[known]
    x = x[known]
    mean, std = x.mean(0), x.std(0).clip(.00001)
    z = (x-mean)/std
    safe_gain = (y[:,0]>0)&(y[:,1]==0)
    model = dict(mean=mean, std=std, gain=fit_logistic(z,safe_gain,balanced=True),
                 harm=fit_logistic(z,y[:,1]==1))
    with np.load(args.r23/"MAP_INNER.npz") as data:
        scores = score_choices(data["x"],model)
        groups, labels = data["g"], data["y"]
    _, starts, counts = np.unique(groups,return_index=True,return_counts=True)
    if not np.all(np.diff(groups)>=0):
        raise ValueError("Noncontiguous parent groups")
    top = np.array([i+np.argmax(scores[i:i+n]) for i,n in zip(starts,counts)])
    safe_inner = (labels[:,0]>0)&(labels[:,1]==0)
    opportunity_parents = sum(bool(safe_inner[i:i+n].any()) for i,n in zip(starts,counts))
    rank_one_hits = int(safe_inner[top].sum())
    trace = []
    for threshold in np.r_[np.arange(-3., 6.01, .125), 100.]:
        decisions = labels[top[scores[top]>=threshold]]
        bad = int((decisions[:,1]==1).sum())
        good = int(((decisions[:,0]>0)&(decisions[:,1]==0)).sum())
        unknown = int((decisions[:,1]<0).sum())
        trace.append(dict(threshold=float(threshold), safe_gain=good, fragmentation_or_loss=bad,
                          unknown=unknown, choices=len(decisions), utility=good-3*bad-unknown))
    best = max(trace,key=lambda r:(r["utility"],-r["fragmentation_or_loss"],-r["unknown"],-r["choices"],r["threshold"]))
    np.savez(args.output/"LINEAR_CUT_SELECTOR.npz",**model,threshold=best["threshold"])
    (args.output/"LINEAR_SELECTOR_INNER.json").write_text(json.dumps(dict(
        fit_known=len(y),fit_safe_gain=int(safe_gain.sum()),inner_parents=len(top),
        inner_parents_with_safe_candidate=opportunity_parents,rank_one_safe_gain=rank_one_hits,
        best=best,trace=trace,penalty="3 per harmful or loss, 1 per unjudged choice",
    ),indent=2))
    print(best)


if __name__ == "__main__":
    main()
