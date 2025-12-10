import numpy as np
import pandas as pd
from itertools import combinations
from itertools import permutations
from itertools import product
from scipy.optimize import linprog
from scipy.stats import norm
import cvxpy as cp
import math
import random

seed = 10
np.random.seed(seed)
random.seed(seed)

class GenData:
    DEFAULTS = {
        "temp_min": 0.4,
        "temp_max": 0.6,
        "f_rational": 0.5,
    }

    allowed_kwargs = set(DEFAULTS.keys())

    def __init__(self, n_alternatives=4, **kwargs):
        self.alternatives = list(range(n_alternatives))
        self.linear_orders = list(permutations(self.alternatives, len(self.alternatives)))
        
        for key in kwargs:
            if key not in self.allowed_kwargs:
                raise ValueError(f"Unknown parameter: {key}")

        params = {**self.DEFAULTS, **kwargs}

        for key, value in params.items():
            setattr(self, key, value)


        self.rational_los, self.irrational_los = self.split()
        self.menus = self.get_menus()

    def get_menus(self):
        menus = []
        for i in range(2, len(self.alternatives)+1):
            menus.extend(combinations(self.alternatives, i))
        return menus

    def split(self):

        number_rational = math.floor(self.f_rational*len(self.linear_orders))
        if number_rational == 0:
            rational_linearOrders = None
            irrational_linearOrders = self.linear_orders
        elif number_rational == len(self.linear_orders):
            rational_linearOrders = self.linear_orders
            irrational_linearOrders = None
        else:
            rational_linearOrders = np.random.choice(list(range(len(self.linear_orders))), size=number_rational, replace=False)
            rational_linearOrders = [self.linear_orders[i] for i in rational_linearOrders]
         
            irrational_linearOrders = [order for order in self.linear_orders if order not in rational_linearOrders]

        return rational_linearOrders, irrational_linearOrders
    
    def generate_rational(self):
        los = self.rational_los
        alt_menu = [(alt, menu) for menu in self.menus for alt in menu]
        rational_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(alt_menu), columns=los)
        for (alt, menu) in rational_df.index:
            for lo in rational_df.columns:
                rank = [a for a in lo if a in menu]
                rational_df.at[(alt, menu), lo] = 1 if rank[0] == alt else 0
        
        return rational_df
    
    def generate_all(self):
        menu_alts = [(alt, menu) for menu in self.menus for alt in menu]
        if self.f_rational != 0:
            rational_df = self.generate_rational()
        if self.f_rational != 1:
            irrational_los = self.irrational_los
            temperatures = [random.uniform(self.temp_min, self.temp_max) for _ in self.irrational_los]
            temperatures = dict(zip(irrational_los, temperatures))
            self.temperatures = temperatures
            # create dict for all utilities 
            utilities = {}
            for lo in irrational_los:
                inf = {}
                for i,alt in enumerate(lo[::-1]):
                    inf[alt] = i+1
                utilities[lo] = inf
            # make irrational df
            irrational_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(menu_alts), columns=irrational_los)
            for row_i, row in irrational_df.iterrows():
                for col_i, col in enumerate(irrational_df.columns):
                    row[col] = norm.cdf(utilities[col][row_i[0]], 0, temperatures[col])
            irrational_df = irrational_df.groupby(level=1).transform(lambda x: x/x.sum())
        if self.f_rational == 1:
            main = rational_df
        elif self.f_rational == 0:
            main = irrational_df
        else:
            main = pd.concat([rational_df, irrational_df], axis=1)
        return main

    def random_support(self):
        supp = np.random.randint(0,100, size=len(self.linear_orders))
        supp = [s/sum(supp) for s in supp]

        return dict(zip(self.linear_orders, supp))

    def get_data(self):
        data = self.generate_all()
        supports = self.random_support()
        data['overall'] = data.apply(lambda row: np.dot(np.array(row), np.array(list(supports.values()))), axis=1)
        information = {}
        for lo in [col for col in data.columns if col != 'overall']:
            inf = {}
            if self.f_rational != 0:
                inf['rational'] = lo in self.rational_los
                inf['temperature'] = None if lo in self.rational_los else self.temperatures[lo]
            else:
                inf['rational'] = False
                inf['temperature'] = self.temperatures[lo]
            inf['support'] = supports[lo]
            information[lo] = inf
        return data, information


class GenDataUnconstrained():
    def __init__(self, n_alternatives=4):
        self.n_alternatives = n_alternatives
        self.alternatives = list(range(self.n_alternatives))
        self.columns = list(permutations(self.alternatives, self.n_alternatives))
        menus = []
        for i in range(2, self.n_alternatives + 1):
            menus.extend(list(combinations(self.alternatives, i)))

        self.rows = [(alt, menu) for menu in menus for alt in menu]

    def gen_data(self):
        data_frame = pd.DataFrame(index = pd.MultiIndex.from_tuples(self.rows), columns=self.columns)
        data_frame = data_frame.map(lambda x: random.uniform(0,1))
        data_frame = data_frame.groupby(level=1).transform(lambda x: x/x.sum())

        support = [random.uniform(0,10) for _ in range(data_frame.shape[1])]
        support = [s/sum(support) for s in support]
        data_frame['overall'] = data_frame.apply(lambda row: np.dot(np.array(row), np.array(support)), axis=1)
        self.support = dict(zip(data_frame.columns, support))
        return data_frame



class SolveRum():
    def __init__(self, data):
        self.main_index = data.index
        self.alternatives = list(set([item1 for item1, _ in data.index]))
        self.menus = list(set([item2 for _, item2 in data.index]))
        self.main_columns = list(permutations(self.alternatives, len(self.alternatives)))
        self.data = data

    def generate_matrix(self, depth=0):
        rational_matrix = pd.DataFrame(index = self.main_index, columns = self.main_columns)
        for (alt, menu) in rational_matrix.index:
            for lo in rational_matrix.columns:
                rnk = [a for a in lo if a in menu]
                rational_matrix.at[(alt, menu), lo] = 1 if alt == rnk[0] else 0
        if depth == 0:
            main_return = rational_matrix
        else:
            main_return = self.generate_exp_matrix(rational_matrix, self.menus, depth=depth)

        return main_return
    
    def generate_exp_matrix(self, rational_df, menus, depth=1):
        menu_rows = {menu: [(alt, menu) for alt in menu] for menu in menus}
        new_columns = {}
        for col in rational_df.columns:
            for mistake_menus in combinations(menus, depth):    
                new_col = rational_df[col].copy()
                for menu in mistake_menus:
                    rows = menu_rows[menu] 
                    chosen_row = [r for r in rows if new_col[r] == 1][0]
                    chosen_alt = chosen_row[0]
                    wrong_alts = [a for a in menu if a != chosen_alt]
                    for wrong_alt in wrong_alts:
                        wrong_row = (wrong_alt, menu)
                        col_name = f"{col}_menu={menu}_mistake_to={wrong_alt}"
                        temp_col = new_col.copy()
                        temp_col[chosen_row] = 0
                        temp_col[wrong_row] = 1
                        new_columns[col_name] = temp_col

        mistake_df = pd.DataFrame(new_columns)
        mistake_df = mistake_df.loc[:, ~mistake_df.T.duplicated()]
        
        return mistake_df
    
    def begin_solve(self):
        d = 0
        success = False
        while not success:
            mat_main = pd.DataFrame()
            weights = []
            for i in range(d+1):
                new_mat = self.generate_matrix(i)
                weights = weights + [i+1]*new_mat.shape[1]
                mat_main = pd.concat([mat_main, new_mat], axis=1)
            max_support = mat_main.shape[1]
            weights = [10*(max(weights)+1 - w) for w in weights]
            w = -1*np.array(weights)

            A_eq = mat_main.to_numpy()
            b_eq = self.data.to_numpy()
            A_eq_sum = np.ones((1, max_support))
            b_eq_sum = np.array([1])

            A_eq_total = np.vstack([A_eq, A_eq_sum])
            b_eq_total = np.concatenate([b_eq, b_eq_sum])

            bounds = [(0,1)]*(max_support)

            result = linprog(w, A_eq = A_eq_total, b_eq = b_eq_total, bounds=bounds, method='highs')
            if not result.success:
                d += 1
            else:
                values = result.x

            success = result.success

        return values

dt, info = GenData(3, f_rational=0, temp_min=0.2,temp_max=0.9).get_data()
dt2 = GenDataUnconstrained(3).gen_data()

values = list(SolveRum(dt['overall']).begin_solve())
print(values)
information_main = pd.DataFrame.from_dict(info, orient='columns')
information_main.columns = list(info.keys())
information_main = pd.concat([information_main, pd.DataFrame(values, index=information_main.columns).T])
print(information_main)