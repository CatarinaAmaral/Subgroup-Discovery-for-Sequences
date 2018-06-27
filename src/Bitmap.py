import numpy as np
import sys
from bitarray import bitarray

class Bitmap:

    def __init__(self, size):
            lenght = size + 1
            bit_array = bitarray(lenght)
            bit_array.setall(False)
            self.bitmap = bit_array #create a new bitmap with the respective size and put all positions to false (0)
            self.support = 0 # the number of bits that are currently set to 1 corresponding to different sequences
            self.first_itemset_id = -1 # the id of the first itemset containing a bit set to 1 (in any sequence)
            self.first_itemset_id = -1 # the id of the first itemset containing a bit set to 1 (in any sequence)
            self.last_sid = -1
            self.support_without_gap_total = 0
            self.sid_sum = 0

    # Set a bit to 1 in this bitmap (sid corresponding to that bit, tid corresponding to that bit, list of sequence length to know how many bits are allocated to each sequence)
    def register_bit(self, sid, tid, sequence_size):
        # calculate the position of the bit that we need to set to 1
        pos = sequence_size[sid] + tid
        # set the bit to 1
        self.bitmap[pos] = True
        # Update the  count of bit set to 1     
        if sid != self.last_sid: #within the same sequence does not take into account repeated - remove in the second version of the algorithm
            self.support += 1;
            self.sid_sum += sid
        if self.first_itemset_id == -1 | tid < self.first_itemset_id:
            self.first_itemset_id = tid
        # remember the last SID with a bit set to 1
        self.last_sid = sid
    
    
    # Get the support
    def get_support(self):
        return self.support
    
    def get_sids(self, sequence_size):
        result = list()
        for pos in np.where(self.bitmap)[0]: # array with values equal to True
            sids_lower_than = filter(lambda x: x <= pos, sequence_size)
            sids = min(sids_lower_than, key=lambda x: abs(x-pos))
            result.append(sequence_size.index(sids))
        return result
    
    # Get a bit position in the bitmap and sequence_size and return the sid of the sequence of the bit with position pos
    def bit_to_sid(self, pos, sequence_size):
        sids_lower_than = [i for i in sequence_size if i <= pos]
        return sequence_size.index(max(sids_lower_than))
    
    # Get the last bit of this sequence 
    def get_last_bit_of_sid(self, sid, sequence_size, last_bit_index):
        if (sid + 1) >= len(sequence_size):
            return last_bit_index
        else:
            return sequence_size[sid + 1] - 1
    
    # Create a new bitmap for the s-step by doing a AND between this bitmap and the bitmap of an item
    def create_new_bitmap_s_step(self, bitmap_item, sequences_size, last_bit_index, max_gap):
        # create a new bitset that will be use for the new bitmap
        new_bitmap = Bitmap(last_bit_index)
        # if no maxGap constraint is used
        if max_gap == sys.maxsize:
            # We do an AND with the bitmap of the item and this bitmap
            for pos in list(filter(lambda x: self.bitmap[x], range(len(self.bitmap)))): # array with values equal to True
                sid = self.bit_to_sid(pos, sequences_size)
                # get the index of the last bit representing this sequence (sid)
                last_bit_of_sid = self.get_last_bit_of_sid(sid, sequences_size, last_bit_index)
                match = False
                for bit in list(filter(lambda x: bitmap_item.bitmap[x], range(pos+1, last_bit_of_sid+1))):
                    new_bitmap.bitmap[bit] = True
                    match = True
                    tid = bit - sequences_size[sid]
                    
                    if self.first_itemset_id == -1 or tid < self.first_itemset_id: 
                        self.first_itemset_id = tid
                
                if match:
                    if sid != new_bitmap.last_sid:
                        #update the support
                        new_bitmap.support += 1
                        new_bitmap.support_without_gap_total += 1
                        new_bitmap.sid_sum += sid
                        new_bitmap.last_sid = sid
                    
                # SPAM OPTIMIZATION - to skip the bit from the same sequence
                pos = last_bit_of_sid
                
        # If we need to check the max gap constraint
        else:
            # variable to keep track of the previous sid for support count without gap
            previous_sid = -1
            for pos in list(filter(lambda x: self.bitmap[x], range(len(self.bitmap)))): # array with values equal to True
                sid = self.bit_to_sid(pos, sequences_size)
                last_bit_of_sid = self.get_last_bit_of_sid(sid, sequences_size, last_bit_index)
                
                match = False
                match_without_gap = False
                for bit in list(filter(lambda x: bitmap_item.bitmap[x], range(pos+1, last_bit_of_sid+1))):
                    match_without_gap = True
                    
                    # if the maxgap constraint is not respected, we don't need to continue
                    if (bit - pos) > max_gap:
                        break;
                        
                    # set the bit to 1 in the new bitmap
                    new_bitmap.bitmap[bit] = True
                    # remember that we found that item 
                    match = True
                    
                    # get the tid
                    tid = bit - sequences_size[sid]
                    if self.first_itemset_id == -1 or tid < self.first_itemset_id: 
                        self.first_itemset_id = tid 
                
                if match_without_gap and previous_sid != sid:
                    new_bitmap.support_without_gap_total += 1
                    previous_sid = sid
                
                if match:
                    if sid != new_bitmap.last_sid:
                        #update the support
                        new_bitmap.support += 1
                        new_bitmap.sid_sum += sid
                    
                    new_bitmap.last_sid = sid
                    # We don't do that when we are using the gap onstraint because we need to check all positions (pos = last_bit_of_sid)
        return new_bitmap
				
                    
        
if __name__ == '__main__':
    b = Bitmap(16)
    l = list([1,2,3,3])
    b.register_bit(1, 1, l)
    print(b.bitmap)
    print(b.bitmap.count(True))
    b.register_bit(1, 2, l)
    print(b.bitmap.count(True))
    print(b.bitmap)
    
    