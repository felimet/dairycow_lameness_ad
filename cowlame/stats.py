"""Descriptive and fixed-score randomization analyses of pass-level scores."""
import numpy as np
from scipy.stats import rankdata, binomtest, spearmanr
from .evaluate import metrics


def permutation_test(labels, scores, cfg):
    labels = np.asarray(labels, dtype=int)
    ranks = rankdata(scores)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if min(positive, negative) <= 0:
        raise ValueError("Permutation AUROC requires both classes")
    offset = positive * (positive + 1) / 2
    observed = (ranks @ labels - offset) / (positive * negative)
    rng = np.random.default_rng(cfg.seed)
    statistics = np.fromiter(((ranks @ rng.permutation(labels) - offset) / (positive * negative)
                              for _ in range(cfg.permutation_count)), float, cfg.permutation_count)
    count = int((statistics >= observed).sum())
    return {"observed_auroc": float(observed), "B": cfg.permutation_count, "seed": cfg.seed,
            "exceedances": count, "p": count / cfg.permutation_count,
            "p_plus_one": (count + 1) / (cfg.permutation_count + 1),
            "permutation_mean_auroc": float(statistics.mean()),
            "unit": "pass label, fixed pooled OOF scores; no retraining"}


def severity_analysis(oof):
    threshold = metrics(oof.anomaly, oof.score)["threshold"]
    flagged = oof.score.to_numpy() >= threshold
    marginal = float(flagged.mean())
    correlation = spearmanr(oof.score, oof.score_1_3)
    result = {"threshold": threshold, "marginal_flagging_rate": marginal,
              "spearman_rho": float(correlation.statistic), "spearman_p": float(correlation.pvalue),
              "severity": {}}
    for grade in (1, 2, 3):
        selection = oof.score_1_3.to_numpy() == grade
        n, detected = int(selection.sum()), int(flagged[selection].sum())
        result["severity"][str(grade)] = {"n": n, "detected": detected,
            "median_score": float(np.median(oof.score.to_numpy()[selection])),
            "binomial_p_greater": float(binomtest(detected, n, marginal, alternative="greater").pvalue)}
    return result


def crowding_analysis(oof):
    records = []
    for stratum, group in oof.groupby("max_concurrent"):
        auc = metrics(group.anomaly, group.score)["auroc"] if group.anomaly.nunique() == 2 else None
        records.append({"stratum": int(stratum), "n": len(group),
                        "n_abnormal": int(group.anomaly.sum()), "auroc": auc})
    return records


def per_cow_normalize(oof, decimals=12):
    """Within-cow z-scores; two-pass cows yield exact +/-1 ties, so round away
    float noise that would otherwise break those ties after a CSV round trip."""
    result = oof.copy()
    for _, group in result.groupby("group"):
        if len(group) > 1:
            spread = float(group.score.std(ddof=0))
            result.loc[group.index, "score"] = ((group.score - group.score.mean()) / spread
                                                 if spread > 0 else 0.0)
    result["score"] = result.score.round(decimals)
    return result
