import numpy as np
from Itemset import Itemset

class Prefix:
    
    def __init__(self):
        self.itemsets = list() #list of itemsets
      
    # Add a given itemset
    def add_itemset(self, itemset):
        self.itemsets.append(itemset)
        return self
    
    # Make a copy of that sequence
    def clone_sequence(self):
        sequence = Prefix()
        for itemset in self.itemsets:
            sequence.add_itemset(itemset.clone_itemset())
        return sequence
    
    # Get the itemset at a given position
    def get_itemset_at_position(self, position):
        return self.itemsets[position]
    
    # Get the ith item in this sequence (no matter in which itemset), if it does not return 'None'
    def get_ith_item(self, ith):
        for itemset in self.itemsets:
            if len(itemset.items) > ith:
                return itemset.items[ith]
            else:
                ith -= len(itemset.items)
    
    # Return the sum of the total number of items in this sequence
    def get_count_items(self):
        count = 0
        for itemset in self.itemsets:
            count += len(itemset.items)
        return count
    
    # Get all the items in the Prefix
    def get_itemsets(self):
        result = []
        for itemset in self.itemsets:
            result.extend(itemset.items)
        return result
            

if __name__ == '__main__':
    i = Itemset()
    i.add_item(4)
    i.add_item(5)
    i.add_item(3)
    i.add_item(3)
    
    e = Itemset()
    e.add_item(4)
    e.add_item(3)
    e.add_item(2)
    
    p = Prefix()
    p.add_itemset(i)
    p.add_itemset(e)
    for itemset in p.itemsets:
        print(itemset.items)
    print (p.get_ith_item(7))
    print(p.get_count_items())
        
        