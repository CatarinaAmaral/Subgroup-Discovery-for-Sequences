class Itemset:
    
    def __init__(self):
        self.items = list()
        
    # Add a given item
    def add_item(self, item):
        self.items.append(item)
        return self
    
    # This method makes a copy of an itemset
    def clone_itemset(self):
        itemset = Itemset()
        for item in self.items:
            itemset.add_item(item)
        return itemset
     
    # Get an item at a given position in this itemset
    def get_item_at_position(self, position):
        return self.items[position]
    
    # This methods makes a copy of this itemset but without items having a support lower than minsup #TODOver melhor
    def itemset_minus_items(self, map_sid_support, rel_min_sup):
        itemset = Itemset()
        abs_min_sup = len(map_sid_support) * rel_min_sup
        for key in list(map_sid_support.keys()):
            if map_sid_support[key] >= abs_min_sup:
                itemset.add_item(key)
        return itemset
    
    # This methods checks if another itemset is contained in this one
    def contains_all(self, itemset):
        return all(x in self.items for x in itemset.items)
     

if __name__ == '__main__':
    i = Itemset()
    i.add_item(4)
    i.add_item(5)
    i.add_item(3)
    i.add_item(3)
    print(i.items)
    print(i.get_item_at_position(2))
    mapSidSup = dict({4:1, 5:1, 3:2})
    it = i.itemset_minus_items(mapSidSup, 0.5)
    print(it.items)
    e = Itemset()
    e.add_item(4)
    e.add_item(3)
    e.add_item(3)
    