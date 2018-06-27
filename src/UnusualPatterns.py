from DataPreparation import DataPreparation
from SPAM import SPAM
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import matplotlib.ticker as mtick
from pandas.plotting import scatter_matrix
from matplotlib.ticker import FormatStrFormatter

class UnusualPatterns:
    
    def __init__(self, file, frequent_patterns, indicators, logs, rules_ids):
        self.file = file
        self.indicators = indicators
        self.patterns = frequent_patterns
        self.logs = logs
        self.rules = rules_ids
        self.indicators_global = {}
        self.dropout_per_node = dict((node,0) for node in rules_ids['id'])
        self.patterns.rename(columns={'SUP': 'Relative SUP', 'Pattern': 'Pattern'}, inplace=True)
        
        for key in self.dropout_per_node:
            self.dropout_per_node[key] = self.get_average_dropout_per_node(key)
        self.dropout_per_node = {key:val for key, val in self.dropout_per_node.items() if val != None}
            
        # How many sessions have met the goal (they went through that index)
        for index, row in indicators.iterrows():
            self.indicators_global[str(row['id'])] = 0
            for indexx, roww in self.logs.iterrows():
                if row['id'] in roww['index_in_rules_plus_one']:
                    self.indicators_global[str(row['id'])] += row['index']
    
    # Check if a pattern is contained in another pattern
    def is_sub(self, sub, lst):
        ln = len(sub)
        for i in range(len(lst) - ln + 1):
            if all(sub[j] == lst[i+j] for j in range(ln)):
                return True
        return False
    
    ############## GLOBAL ##################
    
    # Get the probability of a session going through the indicator
    def probability_indicator(self, indicator_row):
        count = 0
        for index, row in self.logs.iterrows():
            if indicator_row['id'] in self.logs.loc[index,'index_in_rules_plus_one']:
                count += 1
        return count / len(self.logs)
        
    # Get the probability of a pattern going through the indicator
    def probability_pattern_and_indicator(self, indicator_row, pattern_index):
        count = 0
        for sid in self.patterns.loc[pattern_index, 'SID']:
            if indicator_row['id'] in self.logs.loc[sid, 'index_in_rules_plus_one']:
                count += 1
        return count / len(self.logs)

    # Get the probability of a pattern
    def probability_pattern(self, pattern_index):
        return len(self.patterns.loc[pattern_index, 'SID']) / len(self.logs)    
    
    # Get the probability of a pattern going through the indicator
    def probability_indicator_given_pattern(self, indicator_row, pattern_index):
        return self.probability_pattern_and_indicator(indicator_row, pattern_index) / self.probability_pattern(pattern_index)
    
    # Formula for the global interest
    def global_interest(self, indicator_row, pattern_index):
        return self.probability_indicator_given_pattern(indicator_row, pattern_index) - self.probability_indicator(indicator_row)
        
    # Formula for the global criterion that will measure the unusualness of a pattern
    def global_criterion(self, indicator_row, pattern_index):
        return abs(self.global_interest(indicator_row, pattern_index))*self.patterns.loc[pattern_index, 'Relative SUP']
   
    # Create an fill columns
    def global_selection(self):
        self.patterns['Relative SUP'] = self.patterns['Relative SUP'].apply(lambda x: x/len(self.logs.index))
        for index, row in self.indicators.iterrows():
            self.patterns['global_interest_'+str(row['id'])] = 0
            self.patterns['global_criterion_'+str(row['id'])] = 0
        
        for index, row in self.patterns.iterrows():
            for indexx, roww in self.indicators.iterrows():
                if self.indicators.loc[indexx, 'id'] in self.patterns.loc[index, 'Pattern']:
                   self.patterns.loc[index, 'global_criterion_'+str(roww['id'])] = 0
                   self.patterns.loc[index, 'global_interest_'+str(roww['id'])] = 0
                else:
                    self.patterns.loc[index, 'global_criterion_'+str(roww['id'])] = self.global_criterion(roww, index)
                    self.patterns.loc[index, 'global_interest_'+str(roww['id'])] = self.global_interest(roww, index)  
                    
                
    # Select rows and sort them    
    def sort_by_criterion(self, criterion):
        self.patterns['sort'] = abs(self.patterns[criterion])
        self.patterns.sort_values(by= 'sort', ascending=False, inplace=True)
        self.patterns.drop('sort', axis=1, inplace=True)
        
    # Main function for global criterion 
    def global_criterion_main(self, criterion):
        self.global_selection()
        self.sort_by_criterion(criterion)
    

    ############## LOCAL ##################
    
    # Get the probability of an indicator
    def probability_input(self, input_node):
        count = 0
        for session in self.logs['index_in_rules_plus_one']:
            if input_node in session:
                count += 1
        return count / len(self.logs)
    
    # Get the support of a node
    def get_support_node(self, node):
        count = 0
        for session in self.logs['index_in_rules_plus_one']:
            if node in session:
                count += 1
        return count
    
    # Get the probability of a session going through the input of a pattern and a indicator
    def probability_input_and_indicator(self, indicator_row, input_node):
        count = 0
        for session in self.logs['index_in_rules_plus_one']:
            if input_node in session and indicator_row['id'] in session:
                count += 1
        return count / len(self.logs)
    
    # Conditional probability of indicator given input node
    def probability_of_indicator_given_input(self, indicator_row, pattern_index):
        return self.probability_input_and_indicator(indicator_row, self.patterns.loc[pattern_index,'Pattern'][0]) / self.probability_input(self.patterns.loc[pattern_index,'Pattern'][0])
        
    # Formula for the local interest
    def local_interest(self, indicator_row, pattern_index):
        return self.probability_indicator_given_pattern(indicator_row, pattern_index) - self.probability_of_indicator_given_input(indicator_row, pattern_index)
     
    # Formula for the local criterion that will measure the unusualness of a pattern
    def local_criterion(self, indicator_row, pattern_index):
        return abs(self.local_interest(indicator_row, pattern_index)) * self.patterns.loc[pattern_index, 'Relative SUP']
        
    # Create an fill columns
    def local_selection(self):
        
        for index, row in self.indicators.iterrows():
            self.patterns['local_interest_'+str(row['id'])] = 0
            self.patterns['local_criterion_'+str(row['id'])] = 0
            
        for index, row in self.patterns.iterrows():
            self.patterns.loc[index, 'Relative SUP'] = self.patterns.loc[index, 'Relative SUP'] / self.get_support_node(self.patterns.loc[index,'Pattern'][0])
            for indexx, roww in self.indicators.iterrows():
                if self.indicators.loc[indexx, 'id'] in self.patterns.loc[index, 'Pattern']:
                    self.patterns.loc[index, 'local_criterion_'+str(roww['id'])] = 0
                    self.patterns.loc[index, 'local_interest_'+str(roww['id'])] = 0
                else:
                    self.patterns.loc[index, 'local_criterion_'+str(roww['id'])] = self.local_criterion(roww, index)
                    self.patterns.loc[index, 'local_interest_'+str(roww['id'])] = self.local_interest(roww, index)

    # Main function for local criterion 
    def local_criterion_main(self, criterion):
        self.local_selection()
        self.sort_by_criterion(criterion)
    
    ############## DROPOUT ##################
    ### DROPOUT GLOBAL ###
    
     # Get the dropout of a node
    def get_average_dropout_per_node(self, node):
        ind = self.rules.index[self.rules['id'].values.tolist().index(node)]
        if self.rules.loc[ind, 'next'] == [None]:
            return None
        else:
            count_sessions_node = 0
            count_sessions_next = 0
            for index, row in self.logs.iterrows():
                if node in self.logs.loc[index, 'index_in_rules_plus_one']:
                        count_sessions_node += 1
                for next_node in self.rules.loc[node-1, 'next']:
                    nodes = [node,next_node]
                    if self.is_sub(nodes, self.logs.loc[index, 'index_in_rules_plus_one']):
                        count_sessions_next += 1
            if count_sessions_node == 0:# or count_sessions_next == 0:
                return None
            else:
                return abs((count_sessions_node - count_sessions_next)) / count_sessions_node
    
    # Get the average dropout of the chat
    def average_dropout(self):
        dropout_per_node_not_none = {d:k for d,k in self.dropout_per_node.items() if k != None}
        return float(sum(dropout_per_node_not_none.values())) / len(dropout_per_node_not_none)
    
    # Formula for the global dropout interest
    def global_dropout_interest(self, pattern, criterion):
        pattern_log = list(set(self.patterns.loc[pattern, 'Pattern']).intersection(self.dropout_per_node.keys()))
        count_dropout_per_nodes = 0
        if criterion == 'average':
            for node in pattern_log:
                count_dropout_per_nodes += self.get_average_dropout_per_node(node)
            return self.average_dropout() - (count_dropout_per_nodes / len(pattern_log))
        elif criterion == 'min':
            pat = list(map(lambda node: self.average_dropout() - self.get_average_dropout_per_node(node), pattern_log))
            pat_abs = list(map(abs, pat))
            min_elem = min(pat_abs)
            ind = pat_abs.index(min_elem)
            return pat[ind]
        else:
            sys.exit()
    
    # Formula for the global dropout criterion that will measure the unusualness of a pattern
    def global_dropout_criterion(self, pattern, criterion):
        return abs(self.global_dropout_interest(pattern, criterion)) * self.patterns.loc[pattern, 'Relative SUP']
   
    # Create an fill columns
    def global_dropout_selection(self, criterion):
        self.patterns['Relative SUP'] = self.patterns['Relative SUP'].apply(lambda x: x/len(self.logs.index))
        self.patterns['global_dropout_interest'] = 0
        self.patterns['global_dropout_criterion'] = 0
        for index, row in self.patterns.iterrows():
            self.patterns.loc[index, 'global_dropout_interest'] = self.global_dropout_interest(index, criterion)
            self.patterns.loc[index, 'global_dropout_criterion'] = self.global_dropout_criterion(index, criterion)
    
    # Main function for global dropout criterion 
    def global_dropout_criterion_main(self, criterion):
        self.global_dropout_selection(criterion)
        self.sort_by_criterion('global_dropout_criterion')
    
    
    ### DROPOUT LOCAL ###
    
    # Formula for the local dropout interest
    def local_dropout_interest(self, pattern):
        pattern_log = list(set(self.patterns.loc[pattern, 'Pattern']).intersection(self.dropout_per_node.keys()))
        count_dropout_per_nodes = 0
        for node in pattern_log:
            count_dropout_per_nodes += self.get_average_dropout_per_node(node)
        return self.get_average_dropout_per_node(pattern_log[0]) - (count_dropout_per_nodes / len(pattern_log))    
    
    # Formula for the local dropout criterion that will measure the unusualness of a pattern
    def local_dropout_criterion(self, pattern):
        return abs(self.local_dropout_interest(pattern)) * self.patterns.loc[pattern, 'Relative SUP']

    # Create an fill columns
    def local_dropout_selection(self):
        self.patterns['Relative SUP'] = self.patterns['Relative SUP'].apply(lambda x: x/len(self.logs.index))
        self.patterns['local_dropout_interest'] = 0
        self.patterns['local_dropout_criterion'] = 0
        for index, row in self.patterns.iterrows():
            self.patterns.loc[index, 'local_dropout_criterion'] = self.local_dropout_criterion(index)
            self.patterns.loc[index, 'local_dropout_interest'] = self.local_dropout_interest(index)      
        
    # Main function for local dropout criterion  
    def local_dropout_criterion_main(self):
        self.local_dropout_selection()
        self.sort_by_criterion('local_dropout_criterion')
    
    
    ###############################################

    ### REMOVE FLAT PATTERNS ### 
    
    # Remove flat patterns
    def remove_flat_patterns(self):
        for pattern in self.patterns['Pattern']:
            # scrolls through all elements of the pattern to the penultimate element
            if all(len(self.rules.loc[pattern[node]-1, 'next']) == 1 for node in range(len(pattern) - 1)):
                self.patterns.drop(self.patterns.index[self.patterns['Pattern'].values.tolist().index(pattern)], inplace=True)
    
    # Results view organization
    def cleaning_results(self):
        # Remove SID column
        self.patterns.drop(['SID'], axis=1, inplace=True)
        # swap Pattern column and Relative SUP column
        col_changed = ['Pattern','Relative SUP']        
        fixed_col = [item for item in self.patterns.columns if item not in col_changed]
        new_head = col_changed + fixed_col
        self.patterns = self.patterns[new_head]
        return self.patterns
    
    # Scatter matrix for analysis of results
    def scatter_matrix_info(self):
        self.patterns['Relative SUP'] = self.patterns['Relative SUP'].apply(lambda x: x/len(self.logs.index))
        self.patterns['global_dropout_average'] = 0
        self.patterns['global_dropout_min'] = 0
        self.patterns['local_dropout_average'] = 0
        self.patterns['local_dropout_min'] = 0
        for index, row in self.indicators.iterrows():
            self.patterns['global_macro_'+str(row['id'])] = 0
            self.patterns['global_micro_'+str(row['id'])] = 0
            self.patterns['local_macro_'+str(row['id'])] = 0
            self.patterns['local_micro_'+str(row['id'])] = 0
            self.patterns['global_'+str(row['id'])] = 0
            self.patterns['local_'+str(row['id'])] = 0 
        self.patterns['Relative SUP'] = self.patterns['Relative SUP'].apply(lambda x: x/len(self.logs.index))
        for index, row in self.patterns.iterrows():
            self.patterns.loc[index, 'global_dropout_average'] = self.global_dropout_criterion(index, 'average')
            self.patterns.loc[index, 'global_dropout_min'] = self.global_dropout_criterion(index, 'min')
            self.patterns.loc[index, 'local_dropout_average'] = self.local_dropout_criterion(index, 'average')
            self.patterns.loc[index, 'local_dropout_min'] = self.local_dropout_criterion(index, 'min')
            for indexx, roww in self.indicators.iterrows():
                self.patterns.loc[index, 'global_macro_'+str(roww['id'])] = self.global_criterion(roww, index)
                self.patterns.loc[index, 'global_micro_'+str(roww['id'])] = self.global_micro_criterion(roww, index)
                self.patterns.loc[index, 'local_macro_'+str(roww['id'])] = self.local_criterion(roww, index)
                self.patterns.loc[index, 'local_micro_'+str(roww['id'])] = self.local_micro_criterion(roww, index)
                self.patterns.loc[index, 'global_'+str(roww['id'])] = self.global_mix_criterion(roww, index)
                self.patterns.loc[index, 'local_'+str(roww['id'])] = self.local_mix_criterion(roww, index)   
    
   # Plot of the scatter matrix
    def scatter_plot(self, indicator):
        self.scatter_matrix_info()
        d_scatter_plot = {'global_p(P)': self.patterns['global_macro_'+str(indicator)], 'global_p(P|I)': self.patterns['global_micro_'+str(indicator)], 'local_p(P)': self.patterns['local_macro_'+str(indicator)], 'local_p(P|I)': self.patterns['local_micro_'+str(indicator)], 'local_|p(P)-p(P|I)|': self.patterns['local_'+str(indicator)], 'global_|p(P)-p(P|I)|': self.patterns['global_'+str(indicator)]}
        df = pd.DataFrame(d_scatter_plot)
        axes = scatter_matrix(df, alpha=0.8, figsize=(45, 45), diagonal='kde')
        
        class ScalarFormatterForceFormat(mtick.ScalarFormatter):
            pass
        
        def _set_format(self, merda, merda2):  # Override function that finds format to use.
            self.format = "%1.2f"  # Give format here

        for ax in axes.ravel():            
            class Adapter(object):
                _initialised = False
                
                def __call__(self, *arg, **kwarg):
                    value = type(self.obj).__call__(self, *arg, **kwarg)
                    if not value: return value
                    if isinstance(value, str): value = float(value.replace('−', '-'))
                    return "{0:.2f}".format(value).replace('-', '−')
            
                def __init__(self, obj):
                    self.obj = obj
                    self._initialised = True
                    
                def __getattr__(self, attr):
                    return getattr(self.obj, attr)
                
                def __setattr__(self, key, value):
                    if not self._initialised:
                        super().__setattr__(key, value)
                    else:
                        setattr(self.obj, key, value)
    
            
            ax.yaxis.set_major_formatter(Adapter(ax.yaxis.get_major_formatter()))
            ax.xaxis.set_major_formatter(Adapter(ax.xaxis.get_major_formatter()))
            
        [plt.setp(item.yaxis.get_majorticklabels(), 'size', 25) for item in axes.ravel()]
        [plt.setp(item.xaxis.get_majorticklabels(), 'size', 25) for item in axes.ravel()]
        [plt.setp(item.yaxis.get_label(), 'size', 40) for item in axes.ravel()]
        [plt.setp(item.xaxis.get_label(), 'size', 40) for item in axes.ravel()]
                
        corr = df.corr().as_matrix()
        for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
            axes[i, j].annotate("%.3f" %corr[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center', size=45)
        
        plt.show()
    
    
    
if __name__ == '__main__':
    
    # data preparation
    file = '3222'
    data_preparation = DataPreparation(file)
    logs = data_preparation.logs_preparation()
    rules = data_preparation.rules_preparation()
    indicators = data_preparation.indexes_preparation(rules)
    rules_ids = data_preparation.rules_to_ids(rules)
    
    # frequent patterns
    spam = SPAM(file)
    spam.set_max_gap(1)
    spam.set_min_pattern_length(2)
    spam.set_max_pattern_length(3)
    spam.spam_algorithm(logs, 0.5)
    
    #unusual patterns
    ind = str(indicators.loc[0][0])
    logs = logs.reset_index()
    unusual_patterns = UnusualPatterns(file, spam.result, indicators, logs, rules_ids)
    
    #GLOBAl
    #unusual_patterns.global_criterion_main('global_criterion_' + ind)
    #LOCAL
    unusual_patterns.local_criterion_main('local_criterion_' + ind)
    #DROPOUT GLOBAL - average OR min
    #unusual_patterns.global_dropout_criterion_main('average')
    #DROPOUT Local
    #unusual_patterns.local_dropout_criterion_main()
    
    # Remove flat patterns
    #unusual_patterns.remove_flat_patterns()
      
    unusual_patterns.indicators.rename(columns={'id':'Node', 'index':'Value', 'index_name':'Indicator Name'}, inplace=True)
    unusual_patterns.rules.rename(columns={'id':'Node', 'name':'Name', 'data':'Text','type':'Type', 'next': 'Next Nodes'}, inplace=True)
    unusual_patterns.logs.rename(columns={'session_id':'Session ID', 'index_in_rules_plus_one':'Session'}, inplace=True)

    patterns = unusual_patterns.cleaning_results()