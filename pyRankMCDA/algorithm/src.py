############################################################################A

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# pyRank

# Citation:
# PEREIRA, V. (2024). Project: pyRankMCDA. GitHub repository: <https://github.com/Valdecy/pyRankMCDA>

############################################################################

import itertools
import math
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import colormaps
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import kendalltau, spearmanr
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

############################################################################


class rank_aggregation():
    """
    Rank aggregation class.

    Input convention
    ----------------
    `ranks` must be a 2D array of shape (n_alternatives, n_rankings).
    Each column is a ranking and each row is an alternative.
    Values must be a permutation of 1..n_alternatives in every column,
    where rank 1 is the best alternative.

    Output convention
    -----------------
    Every aggregation method returns a rank vector of length n_alternatives.
    The i-th value is the final rank assigned to alternative a(i+1).
    """

    ############################################################################

    def __init__(self, ranks, random_state=None):
        self.G = nx.DiGraph()
        self.final_rank = []
        self.rng = np.random.default_rng(random_state)
        self.r = self._validate_ranks(ranks)
        self.methods_dict = {
            'bd':  {'value': np.array([])},  # Borda Method
            'cp':  {'value': np.array([])},  # Copeland Method
            'ffr': {'value': np.array([])},  # Fast Footrule Rank
            'fky': {'value': np.array([])},  # Fast Kemeny-Young
            'fr':  {'value': np.array([])},  # Footrule Rank
            'ky':  {'value': np.array([])},  # Kemeny-Young
            'md':  {'value': np.array([])},  # Median Rank
            'pg':  {'value': np.array([])},  # Page Rank
            'pl':  {'value': np.array([])},  # Plackett-Luce
            'rrf': {'value': np.array([])},  # Reciprocal Rank Fusion
            'sc':  {'value': np.array([])}   # Schulze Method
        }

    ############################################################################
    # Internal Helpers

    def _validate_ranks(self, ranks):
        arr = np.asarray(ranks)
        if arr.ndim != 2:
            raise ValueError("'ranks' must be a 2D array with shape (n_alternatives, n_rankings).")
        if arr.shape[0] < 2:
            raise ValueError("'ranks' must contain at least two alternatives.")
        if arr.shape[1] < 1:
            raise ValueError("'ranks' must contain at least one ranking (one column).")
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("'ranks' must contain only numeric values.")
        if np.isnan(arr).any():
            raise ValueError("'ranks' cannot contain missing values.")
        rounded = np.rint(arr)
        if not np.allclose(arr, rounded):
            raise ValueError("'ranks' must contain integer rank values.")
        arr = rounded.astype(int)
        n_alternatives = arr.shape[0]
        expected = np.arange(1, n_alternatives + 1)
        for j in range(arr.shape[1]):
            col = np.sort(arr[:, j])
            if not np.array_equal(col, expected):
                raise ValueError(
                    f"Each column of 'ranks' must be a permutation of 1..{n_alternatives}. "
                    f"Column {j} is invalid."
                )
        return arr

    def _rank_vector_to_order(self, rank_vector):
        rank_vector = np.asarray(rank_vector)
        return np.argsort(rank_vector, kind='mergesort')

    def _order_to_rank_vector(self, order):
        order = np.asarray(order, dtype=int)
        rank_vector = np.empty(order.shape[0], dtype=int)
        rank_vector[order] = np.arange(1, order.shape[0] + 1)
        return rank_vector

    def _set_final_rank_from_rank_vector(self, rank_vector):
        rank_vector = np.asarray(rank_vector, dtype=int)
        self.final_rank = [(i + 1, int(rank_vector[i])) for i in range(rank_vector.shape[0])]
        return rank_vector

    def _print_final_rank(self):
        print('')
        for alternative, rank in self.final_rank:
            print(f'a{alternative} = {rank}')

    def _objective_sum_distance(self, candidate_rank_vector, observed_rankings, counts, distance_func):
        return sum(distance_func(candidate_rank_vector, ranking) * count for ranking, count in zip(observed_rankings, counts))

    def _scores_to_rank_vector(self, scores, higher_is_better=True, verbose=True, atol=1e-12):
        scores = np.asarray(scores, dtype=float)
        if higher_is_better:
            sorted_indices = np.argsort(-scores, kind='mergesort')
        else:
            sorted_indices = np.argsort(scores, kind='mergesort')

        ordered_scores = scores[sorted_indices]
        order = []
        i = 0
        while i < len(sorted_indices):
            current_score = ordered_scores[i]
            tied_indices = [sorted_indices[i]]
            i += 1
            while i < len(sorted_indices) and np.isclose(ordered_scores[i], current_score, atol=atol, rtol=0.0):
                tied_indices.append(sorted_indices[i])
                i += 1
            if len(tied_indices) > 1:
                tied_indices = self.tie_breaker(tied_indices, verbose=verbose)
            order.extend(tied_indices)

        rank_vector = self._order_to_rank_vector(np.asarray(order, dtype=int))
        return rank_vector

    ############################################################################
    # Distances / Correlations

    def cayley_distance(self, rank1, rank2):
        rank1 = np.asarray(rank1, dtype=int)
        rank2 = np.asarray(rank2, dtype=int)
        n = len(rank1)
        rank2_inverse = [0] * n
        for idx, value in enumerate(rank2):
            rank2_inverse[value - 1] = idx
        pi = [0] * n
        for i in range(0, n):
            pi[i] = rank2_inverse[rank1[i] - 1]
        cycles = 0
        visited = [False for _ in range(0, n)]
        for i in range(0, n):
            if not visited[i]:
                cycles = cycles + 1
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = pi[j]
        return n - cycles

    def footrule_distance(self, rank1, rank2):
        rank1 = np.asarray(rank1, dtype=int)
        rank2 = np.asarray(rank2, dtype=int)
        return int(np.sum(np.abs(rank1 - rank2)))

    def kendall_tau_corr(self, rank1, rank2):
        correlation, _ = kendalltau(rank1, rank2)
        return correlation

    def kendall_tau_distance(self, rank1, rank2):
        n = len(rank1)
        tau, _ = kendalltau(rank1, rank2)
        if np.isnan(tau):
            return np.nan
        return (n * (n - 1) / 2) * (1 - tau) / 2

    def spearman_rank(self, rank1, rank2):
        correlation, _ = spearmanr(rank1, rank2)
        return correlation

    ############################################################################

    def tie_breaker(self, tied_indices, verbose=True):
        tied_indices = np.asarray(tied_indices, dtype=int)
        if verbose:
            tied_alternatives = [f'a{idx + 1}' for idx in tied_indices]
            print(f'\nTies detected among: {" , ".join(tied_alternatives)}')
            print('')
            print('Attempting to resolve ties using the Borda Method.')
            print('')

        m = self.r.shape[0]
        borda_scores = np.sum(m - self.r + 1, axis=1)
        tied_borda_scores = borda_scores[tied_indices]
        order = np.argsort(-tied_borda_scores, kind='mergesort')
        sorted_tied_indices = tied_indices[order].tolist()

        if len(np.unique(tied_borda_scores)) < len(tied_borda_scores):
            if verbose:
                print('Ties still persist after applying the Borda Method. Resolving ties randomly.')
            grouped = {}
            for idx in sorted_tied_indices:
                grouped.setdefault(borda_scores[idx], []).append(idx)
            resolved = []
            for score in sorted(grouped.keys(), reverse=True):
                group = grouped[score]
                if len(group) > 1:
                    group = self.rng.permutation(group).tolist()
                resolved.extend(group)
            sorted_tied_indices = resolved
        else:
            if verbose:
                print('Ties resolved using the Borda Method.')
        return sorted_tied_indices

    ############################################################################

    def borda_method(self, verbose=True):
        m = self.r.shape[0]
        total = np.sum(m - self.r + 1, axis=1)
        rank_vector = self._scores_to_rank_vector(total, higher_is_better=True, verbose=verbose)
        self._set_final_rank_from_rank_vector(rank_vector)
        self.methods_dict['bd']['value'] = total
        if verbose:
            self._print_final_rank()
        return rank_vector.copy()

    def copeland_method(self, verbose=True):
        n_items = self.r.shape[0]
        copeland_scores = np.zeros(n_items, dtype=float)
        for i in range(0, n_items):
            for j in range(0, n_items):
                if i != j:
                    wins = int(np.sum(self.r[i, :] < self.r[j, :]))
                    losses = int(np.sum(self.r[i, :] > self.r[j, :]))
                    if wins > losses:
                        copeland_scores[i] += 1
                    elif losses > wins:
                        copeland_scores[i] -= 1
        rank_vector = self._scores_to_rank_vector(copeland_scores, higher_is_better=True, verbose=verbose)
        self._set_final_rank_from_rank_vector(rank_vector)
        self.methods_dict['cp']['value'] = copeland_scores
        if verbose:
            self._print_final_rank()
        return rank_vector.copy()

    def fast_kemeny_young(self, max_iter=100, verbose=True):
        unique_rankings, counts = np.unique(self.r.T, axis=0, return_counts=True)
        initial_index = np.argmax(counts)
        consensus_rank_vector = unique_rankings[initial_index].copy()
        consensus_order = self._rank_vector_to_order(consensus_rank_vector)
        best_distance = self._objective_sum_distance(consensus_rank_vector, unique_rankings, counts, self.kendall_tau_distance)

        for _ in range(0, max_iter):
            improved = False
            for i in range(0, len(consensus_order)):
                for j in range(i + 1, len(consensus_order)):
                    new_order = consensus_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_rank_vector = self._order_to_rank_vector(new_order)
                    total_distance = self._objective_sum_distance(new_rank_vector, unique_rankings, counts, self.kendall_tau_distance)
                    if total_distance < best_distance:
                        consensus_order = new_order
                        consensus_rank_vector = new_rank_vector
                        best_distance = total_distance
                        improved = True
            if not improved:
                break

        self._set_final_rank_from_rank_vector(consensus_rank_vector)
        self.methods_dict['fky']['value'] = consensus_rank_vector.copy()
        if verbose:
            self._print_final_rank()
        return consensus_rank_vector.copy()

    def fast_footrule_aggregation(self, max_iter=100, verbose=True):
        unique_rankings, counts = np.unique(self.r.T, axis=0, return_counts=True)
        initial_index = np.argmax(counts)
        consensus_rank_vector = unique_rankings[initial_index].copy()
        consensus_order = self._rank_vector_to_order(consensus_rank_vector)
        best_distance = self._objective_sum_distance(consensus_rank_vector, unique_rankings, counts, self.footrule_distance)

        for _ in range(0, max_iter):
            improved = False
            for i in range(0, len(consensus_order)):
                for j in range(i + 1, len(consensus_order)):
                    new_order = consensus_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_rank_vector = self._order_to_rank_vector(new_order)
                    total_distance = self._objective_sum_distance(new_rank_vector, unique_rankings, counts, self.footrule_distance)
                    if total_distance < best_distance:
                        consensus_order = new_order
                        consensus_rank_vector = new_rank_vector
                        best_distance = total_distance
                        improved = True
            if not improved:
                break

        self._set_final_rank_from_rank_vector(consensus_rank_vector)
        self.methods_dict['ffr']['value'] = consensus_rank_vector.copy()
        if verbose:
            self._print_final_rank()
        return consensus_rank_vector.copy()

    def footrule_rank_aggregation(self, verbose=True):
        n_items = self.r.shape[0]
        n_rankings = self.r.shape[1]
        cost_matrix = np.zeros((n_items, n_items), dtype=float)
        for i in range(0, n_items):
            for j in range(0, n_items):
                cost_matrix[i, j] = sum(abs(self.r[i][k] - (j + 1)) for k in range(0, n_rankings))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_ranking = np.zeros(n_items, dtype=int)
        for i, j in zip(row_ind, col_ind):
            best_ranking[i] = j + 1
        self._set_final_rank_from_rank_vector(best_ranking)
        self.methods_dict['fr']['value'] = best_ranking.copy()
        if verbose:
            self._print_final_rank()
        return best_ranking.copy()

    def kemeny_young(self, verbose=True):
        n_items = self.r.shape[0]
        if n_items > 9:
            raise ValueError(
                "Exact Kemeny-Young grows factorially and is disabled for more than 9 alternatives. "
                "Use fast_kemeny_young() instead."
            )
        best_rank_vector = None
        min_distance = math.inf
        for candidate_order in itertools.permutations(range(0, n_items)):
            candidate_rank_vector = self._order_to_rank_vector(candidate_order)
            total_distance = sum(self.kendall_tau_distance(candidate_rank_vector, ranking) for ranking in self.r.T)
            if total_distance < min_distance:
                min_distance = total_distance
                best_rank_vector = candidate_rank_vector
        self._set_final_rank_from_rank_vector(best_rank_vector)
        self.methods_dict['ky']['value'] = best_rank_vector.copy()
        if verbose:
            self._print_final_rank()
        return best_rank_vector.copy()

    def median_rank_aggregation(self, verbose=True):
        median_ranks = np.median(self.r, axis=1)
        rank_vector = self._scores_to_rank_vector(median_ranks, higher_is_better=False, verbose=verbose)
        self._set_final_rank_from_rank_vector(rank_vector)
        self.methods_dict['md']['value'] = median_ranks
        if verbose:
            self._print_final_rank()
        return rank_vector.copy()

    def page_rank(self, alpha=0.85, max_iter=100, verbose=True):
        m = self.r.shape[0]
        n = self.r.shape[1]
        self.G = nx.DiGraph()
        for i in range(0, m):
            self.G.add_node(i)
        W = np.zeros((m, m), dtype=float)
        for i in range(0, m):
            for j in range(0, m):
                if i != j:
                    w_ij = 0
                    for k in range(0, n):
                        if self.r[i][k] > self.r[j][k]:
                            w_ij = w_ij + 1
                    if w_ij > 0:
                        W[i][j] = w_ij
        for i in range(0, m):
            total_outgoing = np.sum(W[i])
            if total_outgoing > 0:
                for j in range(0, m):
                    if W[i][j] > 0:
                        weight = W[i][j] / total_outgoing
                        self.G.add_edge(i, j, weight=weight)
            else:
                for j in range(0, m):
                    self.G.add_edge(i, j, weight=1.0 / m)
        pagerank_scores_dict = nx.pagerank(self.G, alpha=alpha, personalization=None, max_iter=max_iter, tol=1e-09, weight='weight')
        pagerank_scores = np.array([pagerank_scores_dict[i] for i in range(m)])
        rank_vector = self._scores_to_rank_vector(pagerank_scores, higher_is_better=True, verbose=verbose)
        self._set_final_rank_from_rank_vector(rank_vector)
        self.methods_dict['pg']['value'] = pagerank_scores
        if verbose:
            self._print_final_rank()
        return rank_vector.copy()

    def plackett_luce_aggregation(self, verbose=True):
        n_items = self.r.shape[0]
        initial_params = np.ones(n_items, dtype=float)
        ranking_orders = [self._rank_vector_to_order(ranking) for ranking in self.r.T]

        def neg_log_likelihood(params):
            log_likelihood = 0.0
            for order in ranking_orders:
                remaining = order.tolist()
                for i in range(0, len(remaining) - 1):
                    chosen = remaining[i]
                    denom = np.sum(params[remaining[i:]])
                    log_likelihood = log_likelihood + np.log(params[chosen]) - np.log(denom)
            return -log_likelihood

        result = minimize(
            neg_log_likelihood,
            initial_params,
            method='L-BFGS-B',
            bounds=[(1e-5, None)] * n_items,
        )
        params = result.x
        rank_vector = self._scores_to_rank_vector(params, higher_is_better=True, verbose=verbose)
        self._set_final_rank_from_rank_vector(rank_vector)
        self.methods_dict['pl']['value'] = params
        if verbose:
            self._print_final_rank()
        return rank_vector.copy()

    def reciprocal_rank_fusion(self, K=60, verbose=True):
        n_items = self.r.shape[0]
        scores = np.zeros(n_items, dtype=float)
        for rank_list in self.r.T:
            for idx, rank in enumerate(rank_list):
                scores[idx] = scores[idx] + (1 / (rank + K))
        rank_vector = self._scores_to_rank_vector(scores, higher_is_better=True, verbose=verbose)
        self._set_final_rank_from_rank_vector(rank_vector)
        self.methods_dict['rrf']['value'] = scores
        if verbose:
            self._print_final_rank()
        return rank_vector.copy()

    def schulze_method(self, verbose=True):
        n_items = self.r.shape[0]
        rankings = self.r.T
        r_i = rankings[:, :, np.newaxis]
        r_j = rankings[:, np.newaxis, :]
        pref_matrix = (r_i < r_j).astype(int)
        d = np.sum(pref_matrix, axis=0)
        p = np.copy(d)
        p[d <= d.T] = 0
        for k in range(0, n_items):
            np.maximum(p, np.minimum(p[:, k][:, np.newaxis], p[k, :]), out=p)
        comparison = p > p.T
        wins = np.sum(comparison, axis=1)
        rank_vector = self._scores_to_rank_vector(wins, higher_is_better=True, verbose=verbose)
        self._set_final_rank_from_rank_vector(rank_vector)
        self.methods_dict['sc']['value'] = wins
        if verbose:
            self._print_final_rank()
        return rank_vector.copy()

    ############################################################################

    def plot_ranks_heatmap(self, df, size_x=12, size_y=8):
        plt.figure(figsize=(size_x, size_y))
        sns.heatmap(df, annot=True, cmap='coolwarm', fmt='d', linewidths=.5, cbar=False)
        plt.title('Rankings')
        plt.ylabel('Alternatives')
        plt.xlabel('Ranking Aggregation Methods')
        plt.show()

    def plot_ranks_radar(self, df, size_x=20, size_y=12, n_rows=3, n_cols=4):
        categories = df.index
        num_vars = len(categories)
        num_methods = len(df.columns)
        cmap = colormaps.get_cmap('tab20') if num_methods > 10 else colormaps.get_cmap('tab10')
        colors = [cmap(i / num_methods) for i in range(0, num_methods)]
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(size_x, size_y), subplot_kw=dict(polar=True))
        fig.suptitle('Ranking Aggregation Methods', fontsize=20, y=1.02)
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for i, method in enumerate(df.columns):
            ax = axes[i]
            values = df[method].values.flatten().tolist()
            values = values + values[:1]
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles = angles + angles[:1]
            color = colors[i]
            ax.plot(angles, values, label=method, color=color, linewidth=2)
            ax.fill(angles, values, alpha=0.25, color=color)
            ax.set_title(method, size=13, pad=10, color=color)
            ax.grid(color='grey', linestyle='--', linewidth=0.5)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=9, color='black', rotation=45)
            y_ticks = list(range(1, num_vars + 1))
            y_ticks = [tick for tick in y_ticks if tick % 2 != 0 or tick == y_ticks[-1]]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(tick) for tick in y_ticks], fontsize=8, color='grey')
            ax.set_ylim(1, num_vars)
            ax.spines['polar'].set_visible(False)
        for j in range(num_methods, len(axes)):
            fig.delaxes(axes[j])
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def run_methods(self, methods=('bd', 'cp', 'ffr', 'fky', 'fr', 'ky', 'md', 'pg', 'pl', 'rrf', 'sc'), alpha=0.85, pg_iter=100, ky_iter=100, fr_iter=100, K=60):
        # 'bd'  -> Borda Method
        # 'cp'  -> Copeland Method
        # 'ffr' -> Fast Footrule Rank
        # 'fky' -> Fast Kemeny-Young
        # 'fr'  -> Footrule Rank
        # 'ky'  -> Kemeny-Young
        # 'md'  -> Median Rank
        # 'pg'  -> Page Rank
        # 'pl'  -> Plackett-Luce
        # 'rrf' -> Reciprocal Rank Fusion
        # 'sc'  -> Schulze Method

        available_methods = {
            'bd':  ('Borda Method',           lambda: self.borda_method(verbose=False)),
            'cp':  ('Copeland Method',        lambda: self.copeland_method(verbose=False)),
            'ffr': ('Fast Footrule Rank',     lambda: self.fast_footrule_aggregation(fr_iter, verbose=False)),
            'fky': ('Fast Kemeny-Young',      lambda: self.fast_kemeny_young(ky_iter, verbose=False)),
            'fr':  ('Footrule Rank',          lambda: self.footrule_rank_aggregation(verbose=False)),
            'ky':  ('Kemeny-Young',           lambda: self.kemeny_young(verbose=False)),
            'md':  ('Median Rank',            lambda: self.median_rank_aggregation(verbose=False)),
            'pg':  ('Page Rank',              lambda: self.page_rank(alpha, pg_iter, verbose=False)),
            'pl':  ('Plackett-Luce',          lambda: self.plackett_luce_aggregation(verbose=False)),
            'rrf': ('Reciprocal Rank Fusion', lambda: self.reciprocal_rank_fusion(K, verbose=False)),
            'sc':  ('Schulze Method',         lambda: self.schulze_method(verbose=False))
        }

        if not isinstance(methods, (list, tuple)) or len(methods) == 0:
            raise ValueError("'methods' must be a non-empty list or tuple containing method codes.")

        results = {}
        for method in methods:
            if method in available_methods:
                full_name, func = available_methods[method]
                results[full_name] = func()
            else:
                raise ValueError(f"Method '{method}' is not available. Available methods are: {list(available_methods.keys())}")

        df = pd.DataFrame(results)
        df.index = [f'a{i + 1}' for i in range(0, df.shape[0])]
        sort_c = sorted(df.columns, key=lambda col: df[col].tolist())
        df = df[sort_c]
        return df.astype(int)

    def metrics(self, df):
        methods = df.columns.tolist()
        metric_names = ['Kendall Tau Corr', 'Kendall Tau Dist', 'Cayley', 'Footrule', 'Spearman Rank']
        d_matrix = {}
        for metric in metric_names:
            distance_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:
                        rank1 = df[method1].values.astype(int)
                        rank2 = df[method2].values.astype(int)
                        if metric == 'Kendall Tau Corr':
                            distance = self.kendall_tau_corr(rank1, rank2)
                        elif metric == 'Kendall Tau Dist':
                            distance = self.kendall_tau_distance(rank1, rank2)
                        elif metric == 'Cayley':
                            distance = self.cayley_distance(rank1, rank2)
                        elif metric == 'Footrule':
                            distance = self.footrule_distance(rank1, rank2)
                        elif metric == 'Spearman Rank':
                            distance = self.spearman_rank(rank1, rank2)
                        distance_matrix.loc[method1, method2] = distance
                        distance_matrix.loc[method2, method1] = distance
                    elif i == j:
                        distance_matrix.loc[method1, method2] = 1.0 if metric in ['Kendall Tau Corr', 'Spearman Rank'] else 0.0
            d_matrix[metric] = distance_matrix
        return d_matrix

    def metrics_plot(self, d_matrix, size_x=24, size_y=6):
        metrics = ['Kendall Tau Corr', 'Kendall Tau Dist', 'Cayley', 'Footrule', 'Spearman Rank']
        fig, axes = plt.subplots(1, len(metrics), figsize=(size_x, size_y))
        palette = sns.color_palette('tab10')
        for ax, metric, color in zip(axes, metrics, palette):
            distance_matrix_example = d_matrix[(metric)]
            if metric in ['Kendall Tau Corr', 'Spearman Rank']:
                distance_matrix_example = 1 - distance_matrix_example
            if metric in ['Cayley', 'Footrule', 'Kendall Tau Dist']:
                upper_triangular = np.triu(distance_matrix_example.values, k=1)
                scaler = MinMaxScaler()
                non_diagonal_values = upper_triangular[upper_triangular > 0].reshape(-1, 1)
                if non_diagonal_values.size > 0:
                    normalized_values = scaler.fit_transform(non_diagonal_values)
                    normalized_upper_triangular = np.zeros_like(upper_triangular, dtype=float)
                    normalized_upper_triangular[upper_triangular > 0] = normalized_values.flatten()
                    normalized_distance_matrix = normalized_upper_triangular + normalized_upper_triangular.T
                else:
                    normalized_distance_matrix = np.zeros_like(upper_triangular, dtype=float)
                distance_matrix_normalized = pd.DataFrame(normalized_distance_matrix, index=distance_matrix_example.index, columns=distance_matrix_example.columns)
            else:
                distance_matrix_normalized = distance_matrix_example
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                mds_results = mds.fit_transform(distance_matrix_normalized)
            ax.scatter(mds_results[:, 0], mds_results[:, 1], s=100, color=color, alpha=0.7, edgecolor='k')
            texts = [ax.text(mds_results[i, 0], mds_results[i, 1], method, fontsize=9, ha='center', color='black') for i, method in enumerate(distance_matrix_example.index)]
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold', color=color)
            ax.set_xlabel('MDS Dimension 1', fontsize=10)
            ax.set_ylabel('MDS Dimension 2', fontsize=10)
            ax.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.7)
            ax.set_facecolor('#f9f9f9')
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(top=0.85)
        plt.show()
        return


RankAggregation = rank_aggregation

############################################################################
