############################################################################A

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# pyRank

# Citation: 
# PEREIRA, V. (2024). Project: pyRankMCDA. GitHub repository: <https://github.com/Valdecy/pyRankMCDA>

############################################################################

# Required Libraries
import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from adjustText import adjust_text
from matplotlib import colormaps
from scipy.optimize import linear_sum_assignment, minimize
from scipy.stats import kendalltau, spearmanr
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

############################################################################

# Rank Class
class rank_aggregation():
    
    ############################################################################ 
    
    def __init__(self, ranks):
        self.G            = nx.DiGraph()
        self.final_rank   = []
        self.r            = ranks
        self.methods_dict = {
            'bd':  {'value': np.array([])}, # Borda Method
            'cp':  {'value': np.array([])}, # Copeland Method
            'ffr': {'value': np.array([])}, # Fast Footrule Rank
            'fky': {'value': np.array([])}, # Fast Kemeny-Young
            'fr':  {'value': np.array([])}, # Footrule Rank
            'ky':  {'value': np.array([])}, # Kemeny-Young
            'md':  {'value': np.array([])}, # Median Rank
            'pg':  {'value': np.array([])}, # Page Rank
            'pl':  {'value': np.array([])}, # Plackett-Luce
            'rrf': {'value': np.array([])}, # Reciprocal Rank Fusion
            'sc':  {'value': np.array([])}  # Schulze Method
        }
    
    ############################################################################ 
    
    # Function: Cayley Distance
    def cayley_distance(self, rank1, rank2):
        n             = len(rank1)
        rank2_inverse = [0] * n
        for idx, value in enumerate(rank2):
            rank2_inverse[value - 1] = idx
        pi = [0] * n
        for i in range(0, n):
            pi[i] = rank2_inverse[rank1[i] - 1]
        cycles  = 0
        visited = [False for _ in range(0, n)]
        for i in range(0, n):
            if not visited[i]:
                cycles = cycles + 1
                j      = i
                while not visited[j]:
                    visited[j] = True
                    j          = pi[j]
        return n - cycles
    
    # Function: Footrule Distance
    def footrule_distance(self, rank1, rank2):
        distance = sum(abs(rank1[i] - rank2[i]) for i in range(0, len(rank1)))
        return distance
    
    # Function: Kendall Tau Correlation
    def kendall_tau_corr(self, rank1, rank2):
        correlation, _ = kendalltau(rank1, rank2)
        return correlation

    # Function: Kendall Tau Distance
    def kendall_tau_distance(self, rank1, rank2):
        n      = len(rank1)
        tau, _ = kendalltau(rank1, rank2)
        K_d    = (n * (n - 1) / 2) * (1 - tau) / 2
        return K_d
    
    # Function: Spearman Rank Correlation
    def spearman_rank(self, rank1, rank2):
        correlation, _ = spearmanr(rank1, rank2)
        return correlation
    
    ############################################################################  
    
    # Function: Tie Breaker  
    def tie_breaker(self, tied_indices, verbose = True):
        if (verbose == True):
            tied_alternatives = [f'a{idx + 1}' for idx in tied_indices]
            print(f'\nTies detected among: {" , ".join(tied_alternatives)}')
            print('')
            print('Attempting to resolve ties using the Borda Method.')
            print('')
        m = self.r.shape[0]
        X = np.zeros((m, self.r.shape[1]))
        for j in range(0, self.r.shape[1]):
            ranks    = np.argsort(np.argsort(self.r[:, j])) + 1
            X[:, j]  = m - ranks + 1
        borda_scores        = np.sum(X, axis = 1)
        tied_borda_scores   = borda_scores[tied_indices]
        sorted_tied_indices = [x for _, x in sorted(zip(-tied_borda_scores, tied_indices))]
        unique_scores       = np.unique(tied_borda_scores)
        if (len(unique_scores) < len(tied_borda_scores)):
            if (verbose == True):
                print('Ties still persist after applying the Borda Method. Resolving ties randomly.')
            np.random.shuffle(sorted_tied_indices)
        else:
            if (verbose == True):
                print('Ties resolved using the Borda Method.')
        return sorted_tied_indices
    
    ############################################################################ 
   
    # Function: Borda Rank Aggregation
    def borda_method(self, verbose = True):
        m = self.r.shape[0]
        X = np.zeros((m, self.r.shape[1]))
        for j in range(0, self.r.shape[1]):
            ranks   = np.argsort(np.argsort(self.r[:, j])) + 1
            X[:, j] = m - ranks + 1
        total           = np.sum(X, axis = 1)
        sorted_indices  = np.argsort(-total)
        
        ###
        unique_scores, counts = np.unique(total, return_counts = True)
        ties                  = unique_scores[counts > 1]
        if (len(ties) > 0):
            if (verbose == True):
                print(f'\nTies detected for the following scores: {ties}')
            for score in ties:
                tied_indices = np.where(total == score)[0]
                if (verbose == True):
                    tied_alternatives = [f'a{idx + 1}' for idx in tied_indices]
                    print(f'\nTies detected among: {" , ".join(tied_alternatives)}')
                    print('Resolving ties randomly.\n')
                np.random.shuffle(tied_indices)
                sorted_indices = np.concatenate([sorted_indices[~np.isin(sorted_indices, tied_indices)], tied_indices])
        self.final_rank = [(i + 1, sorted_indices[i] + 1) for i in range(0, len(sorted_indices))]
        ###
        
        self.methods_dict['bd']['value'] = -total
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])

    # Function: Copeland Rank Aggregation
    def copeland_method(self, verbose = True):
        n_items         = len(self.r)
        copeland_scores = np.zeros(n_items)
        for i in range(0, n_items):
            for j in range(0, n_items):
                if (i != j):
                    wins, losses = 0, 0
                    for ranking in self.r.T:
                        if (ranking[i] < ranking[j]):
                            wins = wins + 1
                        elif (ranking[i] > ranking[j]):
                            losses = losses + 1
                    if (wins > losses):
                        copeland_scores[i] = copeland_scores[i] + 1
                    elif (losses > wins):
                        copeland_scores[i] = copeland_scores[i] - 1
        sorted_indices  = np.argsort(-copeland_scores)

        ####
        sorted_wins    = copeland_scores[sorted_indices]
        final_ranking  = []
        i              = 0
        rank           = 1
        while (i < n_items):
            current_win  = sorted_wins[i]
            tied_indices = [sorted_indices[i]]
            i            = i + 1
            while (i < n_items and sorted_wins[i] == current_win):
                tied_indices.append(sorted_indices[i])
                i = i + 1
            if (len(tied_indices) > 1):
                sorted_tied_indices = self.tie_breaker(tied_indices, verbose)
                for idx in sorted_tied_indices:
                    final_ranking.append((rank, idx + 1))
                    rank = rank + 1
            else:
                final_ranking.append((rank, tied_indices[0] + 1))
                rank = rank + 1
        self.final_rank = final_ranking
        ###
        
        self.methods_dict['cp']['value'] = -copeland_scores
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return  np.array([rank for _, rank in self.final_rank])
    
    # Function: Approximate Kemeny-Young Rank Agreggation using a Greedy Approach
    def fast_kemeny_young(self, max_iter = 100, verbose = True):
        unique_rankings, counts = np.unique(self.r.T, axis = 0, return_counts = True)
        initial_index           = np.argmax(counts)
        consensus_ranking       = unique_rankings[initial_index]
        for _ in range(0, max_iter):
            improved = False
            for i in range(0, len(consensus_ranking)):
                for j in range(i + 1, len(consensus_ranking)):
                    new_ranking                    = consensus_ranking.copy()
                    new_ranking[i], new_ranking[j] = new_ranking[j], new_ranking[i]
                    total_distance                 = sum(self.kendall_tau_distance(new_ranking, rank) * count for rank, count in zip(unique_rankings, counts) )
                    if ( total_distance < sum(self.kendall_tau_distance(consensus_ranking, rank) * count for rank, count in zip(unique_rankings, counts)) ):
                        consensus_ranking = new_ranking
                        improved          = True
            if not improved:
                break
        self.final_rank = [(i+1, consensus_ranking[i]) for i in range(0, consensus_ranking.shape[0])]
        self.methods_dict['fky']['value'] = np.array([rank for _, rank in self.final_rank])
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])
    
    # Function: Approximate Footrule Rank Aggregation using a Greedy Approach
    def fast_footrule_aggregation(self, max_iter = 100, verbose = True):
        unique_rankings, counts = np.unique(self.r.T, axis = 0, return_counts = True)
        initial_index           = np.argmax(counts)
        consensus_ranking       = unique_rankings[initial_index]
        for _ in range(0, max_iter):
            improved = False
            for i in range(0, len(consensus_ranking)):
                for j in range(i + 1, len(consensus_ranking)):
                    new_ranking                    = consensus_ranking.copy()
                    new_ranking[i], new_ranking[j] = new_ranking[j], new_ranking[i]
                    total_distance                 = sum(self.footrule_distance(new_ranking, rank) * count for rank, count in zip(unique_rankings, counts))
                    if (total_distance < sum(self.footrule_distance(consensus_ranking, rank) * count for rank, count in zip(unique_rankings, counts))):
                        consensus_ranking = new_ranking
                        improved          = True
            if not improved:
                break
        self.final_rank = [(i + 1, consensus_ranking[i]) for i in range(0, consensus_ranking.shape[0])]
        self.methods_dict['ffr']['value'] = np.array([rank for _, rank in self.final_rank])
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])
    
    # Function: Footrule Rank Aggregation using Assignment Problem
    def footrule_rank_aggregation(self, verbose = True):
        n_items     = len(self.r)
        n_rankings  = len(self.r.T)
        cost_matrix = np.zeros((n_items, n_items))
        for i in range(0, n_items):
            for j in range(0, n_items):
                cost_matrix[i, j] = sum(abs(self.r[i][k] - (j + 1)) for k in range(0, n_rankings))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        best_ranking     = np.zeros(n_items, dtype = int)
        for i, j in zip(row_ind, col_ind):
            best_ranking[i] = j + 1  
        self.final_rank = [(i + 1, best_ranking[i]) for i in range(0, n_items)]
        self.methods_dict['fr']['value'] = np.array([rank for _, rank in self.final_rank])
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])
    
    # Function: Kemeny-Young Rank Aggregation
    def kemeny_young(self, verbose = True):
        n_items               = len(self.r.T[0])
        all_possible_rankings = list(itertools.permutations(range(0, n_items)))
        best_ranking          = None
        min_distance          = float('inf')
        for candidate in all_possible_rankings:
            total_distance = 0
            for ranking in self.r.T:
                distance       = self.kendall_tau_distance(candidate, ranking)
                total_distance = total_distance + distance
            if (total_distance < min_distance):
                min_distance = total_distance
                best_ranking = candidate
        best_ranking    = np.array(best_ranking) + 1
        self.final_rank = [(i + 1, best_ranking[i]) for i in range(0, best_ranking.shape[0])]
        self.methods_dict['ky']['value'] = np.array([rank for _, rank in self.final_rank])
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])
        
    # Function: Median Rank Aggregation
    def median_rank_aggregation(self, verbose = True):
        median_ranks    = np.median(self.r, axis = 1)
        sorted_indices  = np.argsort(median_ranks)
    
        ####
        sorted_medians  = median_ranks[sorted_indices]
        final_ranking   = []
        i               = 0
        rank            = 1
        n_items         = len(median_ranks)
        while (i < n_items):
            current_median = sorted_medians[i]
            tied_indices   = [sorted_indices[i]]
            i              = i + 1
            while (i < n_items and sorted_medians[i] == current_median):
                tied_indices.append(sorted_indices[i])
                i = i + 1
            if (len(tied_indices) > 1):
                sorted_tied_indices = self.tie_breaker(tied_indices, verbose=verbose)
                for idx in sorted_tied_indices:
                    final_ranking.append((rank, idx + 1))
                    rank = rank + 1
            else:
                final_ranking.append((rank, tied_indices[0] + 1))
                rank = rank + 1
        self.final_rank = final_ranking
        ####
        
        self.methods_dict['md']['value'] = median_ranks
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])
    
    # Function: Page Rank Algorithm
    def page_rank(self, alpha = 0.85, max_iter = 100, verbose = True):
        m      = len(self.r)
        n      = len(self.r[0])
        self.G = nx.DiGraph()
        for i in range(0, m):
            self.G.add_node(i)
        W = np.zeros((m, m))
        for i in range(0, m):
            for j in range(0, m):
                if (i != j):
                    w_ij = 0
                    for k in range(0, n):
                        if (self.r[i][k] > self.r[j][k]):
                            w_ij = w_ij + 1  
                    if (w_ij > 0):
                        W[i][j] = w_ij
        for i in range(0, m):
            total_outgoing = np.sum(W[i])
            if (total_outgoing > 0):
                for j in range(0, m):
                    if (W[i][j] > 0):
                        weight = W[i][j] / total_outgoing
                        self.G.add_edge(i, j, weight = weight)
            else:
                for j in range(0, m):
                    self.G.add_edge(i, j, weight = 1.0 / m)
        pagerank_scores = nx.pagerank(self.G, alpha = alpha, personalization = None, max_iter = max_iter, tol = 1e-09, weight = 'weight')
        sorted_scores   = sorted(pagerank_scores.items(), key = lambda x: x[1], reverse = True)

        ####
        final_ranking = []
        i             = 0
        rank          = 1
        n_items       = len(sorted_scores)
        while (i < n_items):
            current_score = sorted_scores[i][1]
            tied_indices  = [sorted_scores[i][0]]
            i = i + 1
            while (i < n_items and np.isclose(sorted_scores[i][1], current_score)):
                tied_indices.append(sorted_scores[i][0])
                i = i + 1
            if (len(tied_indices) > 1):
                sorted_tied_indices = self.tie_breaker(tied_indices, verbose)
                for idx in sorted_tied_indices:
                    final_ranking.append((idx + 1, rank))
                    rank = rank + 1
            else:
                final_ranking.append((tied_indices[0] + 1, rank))
                rank = rank + 1
        self.final_rank = sorted(final_ranking, key = lambda x: x[0])
        ####
        
        self.methods_dict['pg']['value'] = np.array([item[1] for item in list(pagerank_scores.items())])
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])
   
    # Function: Plackett-Luce Model Rank Aggregation
    def plackett_luce_aggregation(self, verbose = True):
        n_items        = len(self.r)
        initial_params = np.ones(n_items)
        
        ###############################
        
        def neg_log_likelihood(params):
            log_likelihood = 0
            for ranking in self.r.T:
                items = list(ranking)
                for i in range(0, len(items) - 1):
                    denom          = sum(params[items[j] - 1] for j in range(i, len(items)))
                    log_likelihood = log_likelihood + np.log(params[items[i] - 1]) - np.log(denom)
            return -log_likelihood
        
        ###############################
        
        result          = minimize(neg_log_likelihood, initial_params, method = 'L-BFGS-B', bounds = [(1e-5, None)] * n_items)
        params          = result.x
        sorted_indices  = np.argsort(-params)

        ####
        sorted_wins    = params[sorted_indices]
        final_ranking  = []
        i              = 0
        rank           = 1
        while (i < n_items):
            current_win  = sorted_wins[i]
            tied_indices = [sorted_indices[i]]
            i            = i + 1
            while (i < n_items and sorted_wins[i] == current_win):
                tied_indices.append(sorted_indices[i])
                i = i + 1
            if (len(tied_indices) > 1):
                sorted_tied_indices = self.tie_breaker(tied_indices, verbose)
                for idx in sorted_tied_indices:
                    final_ranking.append((rank, idx + 1))
                    rank = rank + 1
            else:
                final_ranking.append((rank, tied_indices[0] + 1))
                rank = rank + 1
        self.final_rank = final_ranking       
        ###
        
        self.methods_dict['pl']['value'] = params
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])

    # Function: Reciprocal Rank Fusion (RRF) Aggregation
    def reciprocal_rank_fusion(self, K = 60, verbose = True):
        n_items = len(self.r)
        scores  = np.zeros(n_items)
        for rank_list in self.r.T:
            for idx, rank in enumerate(rank_list):
                scores[idx] = scores[idx] + (1 / (rank + K))
        sorted_indices  = np.argsort(-scores)

        ####
        sorted_wins    = scores[sorted_indices]
        final_ranking  = []
        i              = 0
        rank           = 1
        while (i < n_items):
            current_win  = sorted_wins[i]
            tied_indices = [sorted_indices[i]]
            i            = i + 1
            while (i < n_items and sorted_wins[i] == current_win):
                tied_indices.append(sorted_indices[i])
                i = i + 1
            if (len(tied_indices) > 1):
                sorted_tied_indices = self.tie_breaker(tied_indices, verbose)
                for idx in sorted_tied_indices:
                    final_ranking.append((rank, idx + 1))
                    rank = rank + 1
            else:
                final_ranking.append((rank, tied_indices[0] + 1))
                rank = rank + 1
        self.final_rank = final_ranking       
        ###
        
        self.methods_dict['rrf']['value'] = scores
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([rank for _, rank in self.final_rank])
    
    # Function: Schulze Method
    def schulze_method(self, verbose = True):
        n_items     = len(self.r)
        d           = np.zeros((n_items, n_items), dtype = int)
        rankings    = self.r.T
        r_i         = rankings[:, :, np.newaxis]  
        r_j         = rankings[:, np.newaxis, :]  
        pref_matrix = (r_i < r_j).astype(int) 
        d           = d + np.sum(pref_matrix, axis = 0)
        p           = np.copy(d)
        p[d <= d.T] = 0  
        
        for k in range(0, n_items):
            np.maximum(p, np.minimum(p[:, k][:, np.newaxis], p[k, :]), out = p)
        comparison     = p > p.T
        wins           = np.sum(comparison, axis = 1)
        sorted_indices = np.argsort(-wins)

        ####
        sorted_wins    = wins[sorted_indices]
        final_ranking  = []
        i              = 0
        rank           = 1
        while (i < n_items):
            current_win  = sorted_wins[i]
            tied_indices = [sorted_indices[i]]
            i            = i + 1
            while (i < n_items and sorted_wins[i] == current_win):
                tied_indices.append(sorted_indices[i])
                i = i + 1
            if (len(tied_indices) > 1):
                sorted_tied_indices = self.tie_breaker(tied_indices, verbose)
                for idx in sorted_tied_indices:
                    final_ranking.append((rank, idx + 1))
                    rank = rank + 1
            else:
                final_ranking.append((rank, tied_indices[0] + 1))
                rank = rank + 1
        self.final_rank = final_ranking       
        ###

        self.methods_dict['sc']['value'] = -wins
        if (verbose == True):
            print('')
            for node, rank in self.final_rank:
                print(f'a{node} = {rank}')
        return np.array([alternative for _, alternative in self.final_rank])
    
    ############################################################################ 
   
    # Function: Plot Ranks Heatmap
    def plot_ranks_heatmap(self, df, size_x = 12, size_y = 8):
      plt.figure(figsize = (size_x, size_y))
      sns.heatmap(df, annot = True, cmap = 'coolwarm', fmt = 'd', linewidths = .5, cbar = False)
      plt.title('Rankings')
      plt.ylabel('Alternatives')
      plt.xlabel('Ranking Aggregation Methods')
      plt.show()

    # Function: Plot Ranks Radar
    def plot_ranks_radar(self, df, size_x = 20, size_y = 12, n_rows = 3, n_cols = 4):
        categories  = df.index
        num_vars    = len(categories)
        num_methods = len(df.columns)
        cmap        = colormaps.get_cmap('tab20') if num_methods > 10 else colormaps.get_cmap('tab10')
        colors      = [cmap(i / num_methods) for i in range(0, num_methods)]
        fig, axes   = plt.subplots(nrows = n_rows, ncols = n_cols, figsize = (size_x, size_y), subplot_kw = dict(polar = True))
        fig.suptitle('Ranking Aggregation Methods', fontsize = 20, y = 1.02)
        axes        = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        for i, method in enumerate(df.columns):
            ax     = axes[i]
            values = df[method].values.flatten().tolist()
            values = values + values[:1]
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint = False).tolist()
            angles = angles + angles[:1]
            color  = colors[i]
            ax.plot(angles, values, label = method, color = color, linewidth = 2)
            ax.fill(angles, values, alpha = 0.25, color = color)
            ax.set_title(method, size = 13, pad = 10, color = color)
            ax.grid(color = 'grey', linestyle = '--', linewidth = 0.5)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize = 9, color = 'black', rotation = 45)
            y_ticks = list(range(1, num_vars + 1))
            y_ticks = [tick for tick in y_ticks if tick % 2 != 0 or tick == y_ticks[-1]]
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([str(tick) for tick in y_ticks], fontsize = 8, color = 'grey')
            ax.set_ylim(1, num_vars)
            ax.spines['polar'].set_visible(False)
        for j in range(num_methods, len(axes)):
            fig.delaxes(axes[j])
        plt.subplots_adjust(hspace = 0.5, wspace = 0.3)
        plt.tight_layout(rect = [0, 0, 1, 0.96])
        plt.show()

    # Function: Run Selected Methods 
    def run_methods(self, methods = ['bd', 'cp', 'ffr', 'fky', 'fr', 'ky', 'md', 'pg', 'pl', 'rrf', 'sc'], alpha = 0.85, pg_iter = 100, ky_iter = 100, fr_iter = 100, K = 60):

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
        'bd':  ('Borda Method',           lambda: self.borda_method(verbose = False)),
        'cp':  ('Copeland Method',        lambda: self.copeland_method(verbose = False)),
        'ffr': ('Fast Footrule Rank',     lambda: self.fast_footrule_aggregation(fr_iter, verbose = False)),
        'fky': ('Fast Kemeny-Young',      lambda: self.fast_kemeny_young(ky_iter, verbose = False)),
        'fr':  ('Footrule Rank',          lambda: self.footrule_rank_aggregation(verbose = False)),
        'ky':  ('Kemeny-Young',           lambda: self.kemeny_young(verbose = False)),
        'md':  ('Median Rank',            lambda: self.median_rank_aggregation(verbose = False)),
        'pg':  ('Page Rank',              lambda: self.page_rank(alpha, pg_iter, verbose = False)),
        'pl':  ('Plackett-Luce',          lambda: self.plackett_luce_aggregation(verbose = False)),
        'rrf': ('Reciprocal Rank Fusion', lambda: self.reciprocal_rank_fusion(K, verbose = False)),
        'sc':  ('Schulze Method',         lambda: self.schulze_method(verbose = False))
        }
        
        if not isinstance(methods, list) or len(methods) == 0:
            raise ValueError("'Methods' must be a non-empty list containing the names of the methods to run.")
        results = {}
        for method in methods:
            if (method in available_methods):
                full_name, func    = available_methods[method]
                results[full_name] = func()
            else:
                raise ValueError(f"Method '{method}' is not available. Available methods are: {list(available_methods.keys())}")
                
        df       = pd.DataFrame(results)
        df.index = [f'a{i+1}' for i in range(0, df.shape[0])]
        sort_c   = sorted(df.columns, key=lambda col: df[col].tolist())
        df       = df[sort_c]
        return df
    
    # Function: Metrics
    def metrics(self, df):
        methods  = df.columns.tolist()
        metrics  = ['Kendall Tau Corr', 'Kendall Tau Dist', 'Cayley', 'Footrule', 'Spearman Rank']
        d_matrix = {}
        for metric in metrics:
            distance_matrix = pd.DataFrame(index = methods, columns = methods, dtype = float)
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if (i < j):
                        rank1 = df[method1].values
                        rank2 = df[method2].values
                        if (metric == 'Kendall Tau Corr'):
                            distance = self.kendall_tau_corr(rank1, rank2)
                        if (metric == 'Kendall Tau Dist'):
                            distance = self.kendall_tau_distance(rank1, rank2)
                        elif (metric == 'Cayley'):
                            distance = self.cayley_distance(rank1, rank2)
                        elif (metric == 'Footrule'):
                            distance = self.footrule_distance(rank1, rank2)
                        elif (metric == 'Spearman Rank'):
                            distance = self.spearman_rank(rank1, rank2)
                        distance_matrix.loc[method1, method2] = distance
                        distance_matrix.loc[method2, method1] = distance
                    elif (i == j):
                        distance_matrix.loc[method1, method2] = 0
            d_matrix[(metric)] = distance_matrix
        return d_matrix

    # Function: Metrics - Plot
    def metrics_plot(self, d_matrix, size_x = 24, size_y = 6):
        metrics   = ['Kendall Tau Corr', 'Kendall Tau Dist', 'Cayley', 'Footrule', 'Spearman Rank']
        fig, axes = plt.subplots(1, len(metrics), figsize = (size_x, size_y))
        palette   = sns.color_palette('tab10')
        for ax, metric, color in zip(axes, metrics, palette):
            distance_matrix_example = d_matrix[(metric)]
            if (metric in ['Kendall Tau Corr', 'Spearman Rank']):
                distance_matrix_example = 1 - distance_matrix_example
            if (metric in ['Cayley', 'Footrule', 'Kendall Tau Dist']):
                upper_triangular            = np.triu(distance_matrix_example.values, k = 1)
                scaler                      = MinMaxScaler()
                non_diagonal_values         = upper_triangular[upper_triangular > 0].reshape(-1, 1)
                normalized_values           = scaler.fit_transform(non_diagonal_values)
                normalized_upper_triangular = np.zeros_like(upper_triangular, dtype = float)
                normalized_upper_triangular[upper_triangular > 0] = normalized_values.flatten()
                normalized_distance_matrix  = normalized_upper_triangular + normalized_upper_triangular.T
                distance_matrix_normalized  = pd.DataFrame(normalized_distance_matrix, index = distance_matrix_example.index, columns = distance_matrix_example.columns)
            else:
                distance_matrix_normalized = distance_matrix_example
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category = RuntimeWarning)
                mds         = MDS(n_components = 2, dissimilarity = 'precomputed', random_state = 42)
                mds_results = mds.fit_transform(distance_matrix_normalized)
            ax.scatter(mds_results[:, 0], mds_results[:, 1], s = 100, color = color, alpha = 0.7, edgecolor = 'k')
            texts = [ax.text(mds_results[i, 0], mds_results[i, 1], method, fontsize = 9, ha = 'center', color = 'black') for i, method in enumerate(distance_matrix_example.index)]
            adjust_text(texts, ax = ax, arrowprops = dict(arrowstyle = '-', color = 'gray', lw = 0.5))
            ax.set_title(f'{metric}', fontsize = 12, fontweight = 'bold', color = color)
            ax.set_xlabel('MDS Dimension 1', fontsize = 10)
            ax.set_ylabel('MDS Dimension 2', fontsize = 10)
            ax.grid(visible = True, linestyle = '--', linewidth = 0.5, alpha = 0.7)
            ax.set_facecolor('#f9f9f9')
        plt.tight_layout(pad = 2.0)
        plt.subplots_adjust(top = 0.85)
        plt.show()
        return
           
############################################################################
