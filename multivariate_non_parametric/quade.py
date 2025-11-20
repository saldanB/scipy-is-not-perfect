import numpy as np
from scipy.stats import f, mannwhitneyu, rankdata
from statsmodels.stats.multitest import multipletests
from itertools import combinations


class QuadeANCOVA:
    """
    Quade nonparametric ANCOVA:
    - Global test for group differences adjusting for covariates
    - Pairwise post-hoc comparisons (Holm-corrected)
    """

    def __init__(self, y, group, covariates):
        """
        y: array-like (continuous outcome)
        group: array-like (categorical groups; can be strings or numbers)
        covariates: array-like (2D matrix of covariates)
        """
        self.y = np.asarray(y)
        self.group = np.asarray(group)

        self.cov = np.asarray(covariates)
        if self.cov.ndim == 1:
            self.cov = self.cov.reshape(-1, 1)

        self.N = len(self.y)
        self.groups = np.unique(self.group)

        self._fit_residuals()
        self._rank_residuals()

    def _fit_residuals(self):
        """Regress Y on covariates and compute raw residuals."""
        X = np.column_stack([np.ones(self.N), self.cov])
        beta, _, _, _ = np.linalg.lstsq(X, self.y, rcond=None)
        self.residuals = self.y - X @ beta

    def _rank_residuals(self):
        """Rank the residuals."""
        self.residual_ranks = rankdata(self.residuals, method="average")

    def quade_test(self):
        """Perform global Quade test."""
        k = len(self.groups)
        N = self.N

        # group means of ranks
        group_means = []
        group_sizes = []

        for g in self.groups:
            mask = (self.group == g)
            group_means.append(np.mean(self.residual_ranks[mask]))
            group_sizes.append(np.sum(mask))

        group_means = np.array(group_means)
        group_sizes = np.array(group_sizes)
        overall_mean = np.mean(self.residual_ranks)

        # Between-group SS
        ss_between = np.sum(group_sizes * (group_means - overall_mean)**2)

        # Within-group SS
        ss_within = 0.0
        for g, gm in zip(self.groups, group_means):
            mask = (self.group == g)
            ss_within += np.sum((self.residual_ranks[mask] - gm)**2)

        df1 = k - 1
        df2 = N - k
        F_quade = (ss_between / df1) / (ss_within / df2)
        p_value = 1 - f.cdf(F_quade, df1, df2)

        self.F_quade, self.p_value = F_quade, p_value
        self.df1, self.df2 = df1, df2

        return F_quade, p_value, df1, df2

    def pairwise(self, method="holm"):
        """
        Pairwise Mann–Whitney tests using ranked residuals.
        Holm correction by default.
        """
        pairs = list(combinations(self.groups, 2))

        raw_p = []
        comps = []

        for g1, g2 in pairs:
            x = self.residual_ranks[self.group == g1]
            y = self.residual_ranks[self.group == g2]
            _, p = mannwhitneyu(x, y, alternative="two-sided")
            raw_p.append(p)
            comps.append((g1, g2))

        reject, p_adj, _, _ = multipletests(raw_p, method=method)

        return {
            "pairs": comps,
            "p_raw": np.array(raw_p),
            "p_adj": p_adj,
            "significant": reject
        }


######################################################################
####################### DON'T TRY THIS AT HOME #######################
######################################################################

# import numpy as np

# np.random.seed(0)

# # Simulate data
# N = 120
# group = np.repeat(["A", "B", "C"], N//3)
# age = np.random.randint(20, 70, size=N)
# sex = np.random.choice([0, 1], size=N)

# # Outcome influenced by age, sex, and group C
# X = 0.02 * age + 0.7 * sex + (group == "C") * 1.8 + np.random.normal(0, 1, size=N)

# # Run Quade ANCOVA
# qa = QuadeANCOVA(
#     y=X,
#     group=group,
#     covariates=np.column_stack([age, sex])
# )

# # Global test
# F, p, df1, df2 = qa.quade_test()
# print(f"Quade global test: F={F:.4f}, p={p:.4g}")

# # Pairwise comparisons
# pairs = qa.pairwise(method="holm")
# print("\nPairwise comparisons:")
# for (g1, g2), pr, pa, sig in zip(pairs["pairs"], pairs["p_raw"], pairs["p_adj"], pairs["significant"]):
#     print(f"{g1} vs {g2}: raw p={pr:.4g}, adj p={pa:.4g}, significant={sig}")
