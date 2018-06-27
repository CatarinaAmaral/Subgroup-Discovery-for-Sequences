import numpy as np
import pandas as pd
import math
import sys
from DataPreparation import DataPreparation
from Bitmap import Bitmap
from Prefix import Prefix
from Itemset import Itemset

class SPAM:
    
    def __init__(self, file):
        self.file = file
        # input parameters
        self.min_sup = 0
        self.min_pattern_length = 0
        self.max_pattern_length = 1000 
        self.pattern_count = 0 # initialize the number of patterns found
        self.max_gap = sys.maxsize
        self.show_sequence_ids = False
        self.vertical_db = {}  # vertical database dictionary (integer:Bitmap)
        self.result = pd.DataFrame(columns = ['SID', 'SUP', 'Pattern']) # results of frequent patterns
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
    def save_pattern_size_larger_than_one(self, prefix_input, bitmap_input):
        self.pattern_count += 1
        pattern = list()
        for itemset in prefix_input.itemsets:
            pattern += itemset.items
        self.result = self.result.append({'SID': bitmap_input.get_sids(self.sequence_size), 'SUP': bitmap_input.get_support(), 'Pattern': pattern}, ignore_index=True)  

    # Algorithm to discover the patterns
    def spam_algorithm(self, input_logs, min_sup_rel):
        #STEP 0: SCAN THE DATABASE TO STORE THE FIRST BIT POSITION OF EACH SEQUENCE AND CALCULATE THE TOTAL NUMBER OF BIT FOR EACH BITMAP
        bit_index = 0
        for (idx, row) in input_logs.iterrows():
            self.sequence_size.append(bit_index)
            bit_index += len(row.loc['index_in_rules_plus_one']) # I want in bit_index the size of the sequence to know the start id of the new sequence
        self.last_bit_index = bit_index - 1 
        
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
        frequent_items = list()
        for key in list(self.vertical_db.keys()):
            # if the cardinality of this bitmap is lower than the itemset is removed
            if self.vertical_db[key].get_support() < self.min_sup:
                del self.vertical_db[key]
            else:
                # otherwise, we save this item as a frequent sequential pattern of size 1
                if self.min_pattern_length <= 1 and self.max_pattern_length >= 1:
                    self.save_pattern(key) 
                frequent_items.append(key) 
        
        # STEP3: WE PERFORM THE RECURSIVE DEPTH FIRST SEARCH TO FIND LONGER SEQUENTIAL PATTERNS RECURSIVELY
        if self.max_pattern_length > 1:
            # for each frequent item
            for key in list(self.vertical_db.keys()): 
                # We create a prefix with that item 
                prefix = Prefix()
                prefix.add_itemset(Itemset())
                prefix.itemsets[0].add_item(key)
                # We call the depth first search method with that prefix list of frequent items to try to find sequential patterns by appending some of these items
                self.dfs_pruning(prefix, self.vertical_db[key], frequent_items, frequent_items, key, 2)
                
                
    # dfsPruning method as described in the SPAM paper
    def dfs_pruning(self, prefix, prefix_bitmap, items_s_steps, items_i_steps, min_for_i_steps, size_current_prefix):
        # S-STEPS
        # Temporary variables (as described in the paper)
        s_temp = list()
        s_temp_bitmaps = list()
        for item in items_s_steps:
            new_bitmap = prefix_bitmap.create_new_bitmap_s_step(self.vertical_db[item], self.sequence_size, self.last_bit_index, self.max_gap)
            
            # if the support is higher than minsup
            if new_bitmap.support_without_gap_total >= self.min_sup:
                # record that item and pattern in temporary variables
                s_temp.append(item)
                s_temp_bitmaps.append(new_bitmap)
        
        # for each pattern recorded for the s-step
        for pos in range(len(s_temp)):            
            item = s_temp[pos]
            items_s_step = Itemset()
            iss = items_s_step.add_item(item)
            prefix_s_step = prefix.clone_sequence()
            prefix_s_step.add_itemset(iss)
            s_temp_bitmap = s_temp_bitmaps[pos]
            
            if s_temp_bitmap.support >= self.min_sup:
                # save the pattern
                if size_current_prefix >= self.min_pattern_length:
                    self.save_pattern_size_larger_than_one(prefix_s_step, s_temp_bitmap)
                # recursively try to extend that pattern
                if self.max_pattern_length > size_current_prefix:
                    self.dfs_pruning(prefix_s_step, s_temp_bitmap, s_temp, s_temp, s_temp[pos], size_current_prefix + 1)
    
    # Save the results in a file
    def write_to_file(self, file):
        results_to_strings = self.result.apply(lambda sequence:" -1 ".join(str(x) for x in sequence['Pattern']) + " -1 #SUP: " + str(sequence['SUP']), axis = 1)
        results_to_strings.to_csv('../results-SPAM/' + self.file + '-out.txt', index=False, header=None)
       
   
          
if __name__ == '__main__':
    f = 'test1'
    l = DataPreparation(f)
    logs = l.logs_preparation()
    
    s = SPAM(f)
    #s.set_max_gap(1)
    s.set_min_pattern_length(2)
    s.set_max_pattern_length(3)
    s.spam_algorithm(logs, 0.4)
    
    s.write_to_file(f)
    
    
    