import numpy as np
import pandas as pd
import math
import sys
from DataPreparation import DataPreparation
from Bitmap import Bitmap
from Prefix import Prefix
from Itemset import Itemset
from UnusualPatterns import UnusualPatterns

class SecondApproach:
    
    def __init__(self, file, indicators, logs, rules_ids, criterion):
        self.file = file
        self.indicators = indicators
        self.logs = logs
        self.rules = rules_ids
        self.criterion = criterion
        # input parameters
        self.min_sup = 0
        self.min_pattern_length = 0
        self.max_pattern_length = 1000 
        self.pattern_count = 0 # initialize the number of patterns found
        self.max_gap = sys.maxsize
        self.show_sequence_ids = False
        self.vertical_db = {}  # vertical database dictionary (integer:Bitmap)
        self.result = pd.DataFrame()
        self.sequence_size = list()
        self.last_bit_index = 0
     
    # Set the min pattern length
    def set_min_pattern_length(self, min_pattern_length):
        self.min_pattern_length = min_pattern_length
     
    # Get max pattern length
    def set_max_pattern_length(self, max_pattern_length):
        self.max_pattern_length = max_pattern_length
     
    # Get max gap
    def set_max_gap(self, gap):
        self.max_gap = gap
    
    # Get the sids where a given item appears
    def get_sids(self, item):
        result = list()
        for pos in np.where(self.vertical_db[item].bitmap)[0]: # array with values equal to True
            sids_lower_than = filter(lambda x: x <= pos, self.sequence_size)
            sids = min(sids_lower_than, key=lambda x: abs(x-pos))
            result.append(self.sequence_size.index(sids))
        return result
    
    # Save a pattern of size 1 to the output file (the key = node)
    def save_pattern(self, key):
        self.pattern_count += 1
        self.result = self.result.append({'SID': self.vertical_db[key].get_sids(self.sequence_size), 'SUP': self.vertical_db[key].get_support(), 'Pattern': [key]}, ignore_index=True) 

     # Save a pattern of size > 1 to the output file (the key = node)
    def save_pattern_size_larger_than_one(self, temporary_patterns):
        for index, row in temporary_patterns.iterrows():
            self.pattern_count += 1
            pattern = list()
            for itemset in row['prefix'].itemsets:
                pattern += itemset.items
            self.result = self.result.append({'SID': row['bitmap'].get_sids(self.sequence_size), 'SUP': row['bitmap'].get_support(), 'Pattern': pattern}, ignore_index=True)  # guardar sid, pattern and support no resultado 
        
    # Second approach algorithm
    def second_approach_algorithm(self, input_logs, min_sup_rel, pass_next_level):
        #STEP 0: SCAN THE DATABASE TO STORE THE FIRST BIT POSITION OF EACH SEQUENCE AND CALCULATE THE TOTAL NUMBER OF BIT FOR EACH BITMAP
        bit_index = 0
        for (idx, row) in input_logs.iterrows():
            self.sequence_size.append(bit_index)
            bit_index += len(row.loc['index_in_rules_plus_one']) # I want in bit_index the size of the sequence to know the start id of the new sequence
        self.last_bit_index = bit_index - 1 # last item of the last seq 
        
        self.min_sup = math.ceil(min_sup_rel * len(self.sequence_size)) #absolute minimum support
        if self.min_sup == 0: 
            self.min_sup = 1
        
        #STEP1: SCAN THE DATABASE TO CREATE THE BITMAP VERTICAL DATABASE REPRESENTATION
        sid = tid = 0 # sequence id and itemset id
        for (idx, row) in input_logs.iterrows():
            for itemset in row.loc['index_in_rules_plus_one']: 
                if itemset in self.vertical_db:
                    self.vertical_db[itemset].register_bit(sid, tid, self.sequence_size)
                else:
                    self.vertical_db[itemset] = Bitmap(self.last_bit_index)
                    self.vertical_db[itemset].register_bit(sid, tid, self.sequence_size)
                tid += 1
            sid += 1
            tid = 0
        
        # STEP2: REMOVE INFREQUENT ITEMS FROM THE DATABASE BECAUSE THEY WILL NOT APPEAR IN ANY FREQUENT SEQUENTIAL PATTERNS
        for key in list(self.vertical_db.keys()):
            # if the cardinality of this bitmap is lower than the itemset is removed
            if self.vertical_db[key].get_support() < self.min_sup:
                del self.vertical_db[key]
        
        #STEP3: BREADTH FIRST SEARCH
        prefixs = pd.DataFrame(columns = ['prefix', 'bitmap'])
        for key in self.vertical_db.keys(): 
            # We create a prefix with that item 
            prefix = Prefix()
            prefix.add_itemset(Itemset())
            prefix.itemsets[0].add_item(key)
            prefixs = prefixs.append({'prefix': prefix, 'bitmap': self.vertical_db[key]}, ignore_index=True)
        self.bfs_search(prefixs, 2, pass_next_level)
        self.ranking_results()
    
    # Pruning algorithm based in beam search 
    def bfs_search(self, prefixs, level, pass_next_level): 
        result_to_test = pd.DataFrame(columns = ['prefix', 'bitmap'])# results of the patterns of this level
        # S-STEPS
        for index, prefix in prefixs.iterrows():
            temp_result_level = pd.DataFrame(columns = ['s_temp_item', 's_temp_bitmap'])
            for key in self.vertical_db.keys():
                new_bitmap = prefix['bitmap'].create_new_bitmap_s_step(self.vertical_db[key], self.sequence_size, self.last_bit_index, self.max_gap)
                
                if new_bitmap.support_without_gap_total >= self.min_sup:
                # record that item and pattern in temporary variables
                    temp_result_level = temp_result_level.append({'s_temp_item': key, 's_temp_bitmap': new_bitmap}, ignore_index=True)
            
            for index, row in temp_result_level.iterrows():
                item = row['s_temp_item']
                items_s_step = Itemset()
                iss = items_s_step.add_item(item)
                prefix_s_step = prefix['prefix'].clone_sequence()
                prefix_s_step.add_itemset(iss)
                s_temp_bitmap = row['s_temp_bitmap']
                
                #minimum suport equal of 1%
                if s_temp_bitmap.support >= self.min_sup:
                    result_to_test = result_to_test.append({'prefix': prefix_s_step, 'bitmap': s_temp_bitmap}, ignore_index=True)
        result_level = self.save_temporary_patterns(result_to_test)
        level_patterns = self.ranking_unusual_patterns(result_level, pass_next_level)
        
        self.result = self.result.append(level_patterns, ignore_index=True)
        
        prefixFilter = result_to_test['prefix'] \
            .apply(lambda prefix: tuple(prefix.get_itemsets()) in tuple(map(tuple,level_patterns['Pattern'])))
        
        if self.max_pattern_length > level:
            self.bfs_search(result_to_test[prefixFilter], level+1, pass_next_level)
     
    # Auxiliary function of patterns pruning function - save temporary patterns
    def save_temporary_patterns(self, temporary_patterns):
        result = pd.DataFrame(columns = ['SID', 'SUP', 'Pattern'])
        for index, row in temporary_patterns.iterrows():
            pattern = list()
            for itemset in row['prefix'].itemsets:
                pattern += itemset.items
            result = result.append({'SID': row['bitmap'].get_sids(self.sequence_size), 'SUP': row['bitmap'].get_support(), 'Pattern': pattern}, ignore_index=True)  
        return result
    
    #Auxiliary function of patterns pruning function - Ranking patterns by the unusualness
    def ranking_unusual_patterns(self, patterns, pass_next_level):
        logs = self.logs.reset_index()
        unusual_patterns = UnusualPatterns(self.file, patterns, self.indicators, logs, self.rules)
        if self.criterion.startswith('local_criterion'):
            unusual_patterns.local_criterion_main(self.criterion)
        elif self.criterion.startswith('global_criterion'):
            unusual_patterns.global_criterion_main(self.criterion)
        elif self.criterion == 'drop_local':
            unusual_patterns.local_dropout_criterion_main()
        elif self.criterion == 'drop_global_average':
            unusual_patterns.global_dropout_criterion_main('average')
        elif self.criterion == 'drop_global_min':
            unusual_patterns.global_dropout_criterion_main('min')
        else:
            print('Measure of interest does not exist')
            sys.exit()
        return unusual_patterns.patterns.head(pass_next_level)
    
    #Ranking the final patterns
    def ranking_results(self):
        
        if self.criterion == 'drop_local':
            self.result['sort'] = abs(self.result['local_dropout_criterion'])
        elif self.criterion == 'drop_global_average' or self.criterion == 'drop_global_min':
            self.result['sort'] = abs(self.result['global_dropout_criterion'])
        else:
            self.result['sort'] = abs(self.result[self.criterion])
        self.result.sort_values(by= 'sort', ascending=False, inplace=True)
        self.result.drop('sort', axis=1, inplace=True)
        
   
    # Results view organization
    def cleaning_results(self):
        #Remove SID column
        self.result.drop(['SID'], axis=1, inplace=True)
        #swap Pattern column and Relative SUP column
        col_changed = ['Pattern','Relative SUP']
        fixed_col = [item for item in self.result.columns if item not in col_changed]
        new_head = col_changed + fixed_col
        self.result = self.result[new_head]
        return self.result
          
if __name__ == '__main__':
    
    file = '3222'
    data_preparation = DataPreparation(file)
    logs = data_preparation.logs_preparation()
    rules = data_preparation.rules_preparation()    
    indicators = data_preparation.indexes_preparation(rules)
    rules_ids = data_preparation.rules_to_ids(rules)
    
    ind = '16'
    logs = logs.reset_index()
    
    #GLOBAL
    second_approach = SecondApproach(file, indicators, logs, rules_ids, 'global_criterion_' + ind)
    #LOCAL
    #second_approach = SecondApproach(file, indicators, logs, rules_ids, 'local_criterion_' + ind)
    #DROPOUT GLOBAL - 'drop_global_average' or 'drop_global_min'
    #second_approach = SecondApproach(file, indicators, logs, rules_ids, 'drop_global_average')
    #DROPOUT Local
    #second_approach = SecondApproach(file, indicators, logs, rules_ids, 'drop_local')
    
    second_approach.set_max_gap(1)
    second_approach.set_min_pattern_length(2)
    second_approach.set_max_pattern_length(3)
    
    second_approach.second_approach_algorithm(logs, 0.5, 1)
    patterns = second_approach.cleaning_results()
    
        
    
    
    