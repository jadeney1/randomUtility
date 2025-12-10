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

class ChoiceModel:
    def __init__(self, information_input):
        self.n = information_input['n']
        self.X = information_input['alternatives']
        self.probs = information_input['probs']
        self.rows = information_input['rows']
        self.columns = information_input['columns']
        self.C = self.create_C()
        self.RationalRUM = self.trad_RUM()

    def simulate_probs(self, alpha=1.0):
        menus = []
        for r in range(2, len(self.X)+1):
            menus.extend(combinations(self.X, r))
        menu_probs = {}
        for menu in menus: 
            k = len(menu)
            probs = np.random.dirichlet([alpha]*k)
            menu_probs[menu] = dict(zip(menu, probs))

        data = []
        index = []

        for menu, probs in menu_probs.items():
            for alt, p in probs.items():
                index.append((alt, menu))
                data.append(p)
        idx = pd.MultiIndex.from_tuples(index, names=['alternative', 'menu'])

        return pd.Series(data, index=idx, name='choice_probability').sort_index(level='menu')
        
    def create_C(self):
        idx = []
        for alternative, menu in self.rows:
            for ranking in self.columns:
                rank_int = tuple(a for a in ranking if a in menu)
                if rank_int.index(alternative) == 0:
                    idx.append((alternative, menu, ranking, 1))
                else:
                    idx.append((alternative, menu, ranking, 0))
        alt_main = [item[0] for item in idx]
        menu_main = [item[1] for item in idx]
        ranking_main = [item[2] for item in idx]
        choice_main = [item[3] for item in idx]
        C = pd.DataFrame({'alternative': alt_main, 'menu': menu_main, 'ranking': ranking_main, 'choice': choice_main})
        C = C.pivot(index=['alternative', 'menu'], columns='ranking', values='choice')
        C = C.sort_index(level ='menu')

        return C
    
    def trad_RUM(self):
        C = self.C.to_numpy()
        p = self.probs.to_numpy().reshape(-1, 1)

        n = C.shape[1]
        c = np.zeros(n)
        bounds = [[0,1] for _ in range(n)]
        C_eq = np.vstack([C, np.ones(n)])
        p_eq = np.append(p, 1)

        res = linprog(c, A_eq=C_eq, b_eq=p_eq, bounds=bounds, method='highs')
        
        return res.success
    
    def generate_mistake_functions(self, df, col, d):
        menus = df.index.get_level_values("menu").unique()

        true_choice = {
            m: df[col].xs(m, level="menu").idxmax()
            for m in menus
        }

        mistake_alts = {
            m: [a for a in df[col].xs(m, level="menu").index if a != true_choice[m]]
            for m in menus
        }

        all_cf = []  

        for mistaken_menus in combinations(menus, d):

            wrong_lists = [mistake_alts[m] for m in mistaken_menus]

            for wrong_combo in product(*wrong_lists):

                cf = {}
                wrong_map = dict(zip(mistaken_menus, wrong_combo))

                for m in menus:
                    if m in wrong_map:
                        cf[m] = wrong_map[m]         
                    else:
                        cf[m] = true_choice[m]       

                all_cf.append(cf)

        return all_cf

    def mistake_df(self, df, col, d):

        mistake_functions = self.generate_mistake_functions(df, col, d)

        new_cols = []
        new_data = []

        for i, cf in enumerate(mistake_functions):
            col_name = f"m_{str(col)}_{d}_{i}"  
            new_cols.append(col_name)

            col_values = []

            for (alt, menu) in df.index:    
                chosen = cf[menu]           
                col_values.append(1 if alt == chosen else 0)

            new_data.append(col_values)

        new_df = pd.DataFrame(
            np.column_stack(new_data),
            index=df.index,
            columns=new_cols
        )

        return new_df
    
    def create_Ci(self, d=1):
        C = self.C
        main = pd.DataFrame()
        irr = pd.DataFrame()
        for i in range(1, d+1):
            for ranking in C.columns:
                irr = pd.concat([irr, self.mistake_df(C, ranking, i)], axis=1)
            main = pd.concat([main, irr], axis=1)
        nCols = [int(col.split("_")[2]) for col in main.columns]
        return main, nCols

    def solve(self, depth=0):
        if depth==0:
            s = self.trad_RUM()
            info = None
        else:    
            s = False
            C = self.C
            
            C_i, levels = self.create_Ci(d=depth)
            rho = self.probs
            m, n = C.shape
            _, k = C_i.shape

            allLevels = [0]*n + levels
            allLevels = [100**(1/(l+1)) for l in allLevels]

            main = pd.concat([C, C_i], axis=1)

            w = -1*np.array(allLevels)

            A_eq = main.to_numpy()
            b_eq = rho.to_numpy()
            A_eq_sum = np.ones((1, n+k))
            b_eq_sum = np.array([1])

            A_eq_total = np.vstack([A_eq, A_eq_sum])
            b_eq_total = np.concatenate([b_eq, b_eq_sum])

            bounds = [(0,1)]*(n+k)

            result = linprog(w, A_eq = A_eq_total, b_eq = b_eq_total, bounds=bounds, method='highs')
            s = result.success
            if result.success:
                r = pd.DataFrame({'lo': [item1 for item1, _ in zip(list(main.columns), list(result.x))], 'weight': [float(item2) for _, item2 in zip(list(main.columns), list(result.x))]})
                #r['exp_level'] = r['lo'].apply(lambda x: 0 if x in C.columns else levels[x])
                info = r
            else:
                info = None

        return s, info
    
    def iterate(self):
        depth = -1
        success = False
        while not success:
            depth += 1
            success, data = self.solve(depth=depth)
            
        return data
    


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
        menus = self.get_menus()
        rational_data = pd.DataFrame(columns=los, index=menus)
        
        rows = [(menu, alternative) for menu in rational_data.index for alternative in menu]
        rational_data = pd.DataFrame(columns=los, index = pd.MultiIndex.from_tuples(rows))
        for lo in los:
            rank = {alt: i for i, alt in enumerate(lo)}
            for menu in menus:
                top_choice = min(menu, key = lambda x: rank[x])
                for alt in menu:
                    rational_data.at[(menu, alt), lo] = 1 if alt == top_choice else 0
        
        return rational_data
    
    def generate_all(self):
        rational_data = self.generate_rational()
        los = self.irrational_los
        temperatures = [np.random.uniform(self.temp_min, self.temp_max) for _ in range(len(los))]
        self.temp_sum = dict(zip(los, temperatures))
        menu_alts = list(rational_data.index)
        utilities = {}
        for lo in los:
            values = {}
            for ind, alt in enumerate(lo[::-1]):
                values[alt] = int(1*(ind+1))
            utilities[lo] = values

        irrational_data = pd.DataFrame(index = pd.MultiIndex.from_tuples(menu_alts), columns=los)

        for row_index, row in irrational_data.iterrows():
            for col_index, col in enumerate(irrational_data.columns):
                row[col] = norm.cdf(utilities[col][row_index[1]], 0, 10*temperatures[col_index])
        irrational_data = irrational_data.reset_index()
        cols = [col for col in irrational_data.columns if col not in ['level_0', 'level_1']]
        irrational_data[cols] = irrational_data[cols]/irrational_data.groupby(['level_0'])[cols].transform('sum')
        irrational_data = irrational_data.set_index(['level_0', 'level_1'])
        main = pd.concat([rational_data, irrational_data], axis=1)

        return main

    def random_support(self):
        supp = np.random.randint(0,100, size=len(self.linear_orders))
        supp = [s/sum(supp) for s in supp]

        return dict(zip(self.linear_orders, supp))

    def get_data(self):
        data = self.generate_all()
        supports = self.random_support()
        information = {}
        for lo in data.columns:
            info = {}
            info['rational'] = lo in self.rational_los
            info['temperature'] = 0 if lo in self.rational_los else self.temp_sum[lo]
            info['support'] = supports[lo]
            information[lo] = info

        data_main = data.copy()
        data_main['overall'] = data_main.apply(lambda row: np.dot(np.array(row), np.array(list(supports.values()))), axis=1)
        
        main_choice_data = data_main[['overall']]
        main_choice_data = main_choice_data.reset_index()
        main_choice_data.index = pd.MultiIndex.from_tuples(zip(main_choice_data['level_1'], main_choice_data['level_0']), names=['alternative', 'menu'])
        main_choice_data = main_choice_data['overall']

        info_main = {'data': main_choice_data, 
                     'n': int(len(self.alternatives)), 
                     'alternatives': self.alternatives, 
                     'probs': pd.Series(main_choice_data.values),
                     'rows': main_choice_data.index,
                     'columns': self.linear_orders}
        return info_main, information


dt, inf = GenData(3).get_data()
print(ChoiceModel(dt).iterate())
