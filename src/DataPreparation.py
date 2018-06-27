import numpy as np
import pandas as pd
import json
import sys

class DataPreparation:
    
    def __init__(self, file):
        self.file = file
        self.sessions = pd.DataFrame(columns = ['session_id', 'index_in_rules_plus_one'])
    
    # Verify that the map version is the same as the logs
    def check_version(self, maps, logs):
        if len(set(logs[' chat_version'])) == 1:
            if maps['data']['version'] == logs[' chat_version'].values[0]:
                return True
            else:
                return False
        else:
            return False
    
    # Get map chat nodes
    def get_steps(self, rules):
        steps = pd.DataFrame(rules['steps'])
        steps1 = pd.concat([steps['id'],steps['name'],steps['data'],steps['type'],steps['next']], axis=1)
        for step in range(len(steps1['next'])):
            steps1.loc[step, 'next'] = [steps1.loc[step, 'next']]
        for index in range(len(steps['data'])):
            if 'text' in steps['data'][index]:
                steps1['data'][index] = steps['data'][index]['text']
            else:
                steps1['data'][index] = 'No text.'
            if 'targetRules' in steps['data'][index]:
                for intexx in range(len(steps['data'][index]['targetRules'])):
                    steps1['next'][index] = np.append(steps1['next'][index], steps['data'][index]['targetRules'][intexx]['step'])
                steps1['next'][index] = list(set(steps1['next'][index]))
        if len(steps1['id'].unique()) != len(steps1['id']):
            print('IDs repetidos nas regras')
        else:
            return steps1

    # Read the rules file
    def read_rules(self, maps_name):
        maps = open(maps_name).read()
        return json.loads(maps)
        
    # Read the logs file
    def read_logs(self, log_name):
        log = pd.read_csv(log_name)
        return log
    
    # Read the indexes file
    def read_indexes(self, indexes_name):
        ind =pd.read_csv(indexes_name, sep=" ", header=None)
        ind.columns = ["id", "index", "index_name"]
        return ind
    
    # Remove transactions from nodes for themselves
    def remove_circular_logs(self, logs_df):
        logs_df.drop(logs_df[logs_df[' action_id'] == logs_df[' label']][' action_id'].index, inplace=True)
    
    # Check if the node exists in the rules and assign node index of the rules in the logs
    def check_node_exists(self, chat_rules, chat_transactions_unique):
        chat_rules_id_set = set(chat_rules['id'])
        for node_id in set(chat_transactions_unique[' action_id']):
            if node_id in chat_rules_id_set:
                chat_transactions_unique.loc[chat_transactions_unique[' action_id'] == node_id, 'index_in_rules_plus_one'] = \
                    chat_rules.loc[chat_rules['id'] == node_id, 'id'].index[0] + 1
        return chat_transactions_unique[chat_transactions_unique['index_in_rules_plus_one'] != -1]
    
    # Select logs that matter
    def select_logs(self, chat_logs):
        chat_logs = chat_logs.sort_values(['user_id', ' user_time'])
        resumes_to_keep = ~chat_logs[chat_logs[' action'] == '000 - CHAT_RESUME'].duplicated(['user_id', ' action_id', ' action']) # "resumes" que não são repetidos
        chat_transactions = chat_logs[' action'] == 'CHAT_TRANSITION' 
        chat_ends = chat_logs[' action'][chat_logs[' action'].str.endswith('CHAT_END')]
        #Filter only logs that are "transaction" or ("resume" not repeated)
        chat_transactions_resume_unique = chat_logs[resumes_to_keep.reindex(chat_transactions.index, fill_value=False) | chat_transactions | chat_ends].copy()
        chat_transactions_resume_unique['index_in_rules_plus_one'] = -1
        return chat_transactions_resume_unique
    
    # Pass identifiers of rules nodes to indices
    def rules_ids_to_index(self, chat_rules):
        for index, row in chat_rules.iterrows():
            if row['next'] != [None]:
                row['next'] = list(map(lambda next_node: chat_rules.index[chat_rules['id'] == next_node.replace('"', '')].tolist()[-1] + 1,row['next']))                
        return chat_rules
    
    # Pass identifiers of rules nodes to indices
    def rules_to_ids(self, chat_rules):
        for index, row in chat_rules.iterrows():
            if row['next'] != [None]:
                row['next'] = list(map(lambda next_node: chat_rules.index[chat_rules['id'] == next_node.replace('"', '')].tolist()[-1] + 1,row['next']))                
        for index, row in chat_rules.iterrows():
            row['id'] = chat_rules.index[chat_rules['id'] == row['id'].replace('"', '')].tolist()[-1] + 1
        return chat_rules
    
    # Delete sessions that do not follow the rules
    def delete_impossible_sessions(self, chat_rules, transactions_listed):
        for index, row in transactions_listed.iteritems():
            for ind in range(len(row)-1):
                if row[ind] != row[ind+1] and row[ind+1] not in chat_rules.loc[row[ind]-1, 'next']:
                    transactions_listed.drop([index], inplace=True)
                    break         
        return transactions_listed
    
    # Get indexes prepared
    def indexes_preparation(self, chat_rules):
        chat_indexes = self.read_indexes('../data/ind/' + self.file + '.txt')
        chat_indexes['id'] = chat_indexes['id'].map(lambda ind: chat_rules.index[chat_rules['id'] == ind.replace('"', '')].tolist()[-1] + 1)
        return chat_indexes
        
    # Get steps rules
    def rules_preparation(self):
        chat_maps = self.read_rules('../data/maps/' + self.file + '.json')
        return self.get_steps(chat_maps) 
    
    # Get logs prepared
    def logs_preparation(self):
        chat_maps = self.read_rules('../data/maps/' + self.file + '.json')
        chat_logs = self.read_logs('../data/logs/' + self.file + '.csv')
    
        #check versions
        if self.check_version(chat_maps, chat_logs) == False:
            print('Versões incompatíveis')
            sys.exit()
        
        chat_rules = self.rules_preparation()
        
        chat_transactions = self.select_logs(chat_logs)   
        chat_transactions_indexed = self.check_node_exists(chat_rules, chat_transactions)
        
        #Group by user_id and transform the indexes into list
        transactions_listed = chat_transactions_indexed.groupby('user_id')['index_in_rules_plus_one'].apply(list)
        
        #delete sessions that do not follow the rules
        chat_rules = self.rules_ids_to_index(chat_rules)
        transactions_listed = self.delete_impossible_sessions(chat_rules, transactions_listed)
        
        self.sessions['session_id'] = transactions_listed.index
        self.sessions['index_in_rules_plus_one'] = transactions_listed.values
        self.sessions = self.sessions[self.sessions['index_in_rules_plus_one'].map(len) != 1]
        
        logs_to_strings = self.sessions.apply(lambda user:" -1 ".join(str(x) for x in user['index_in_rules_plus_one']) + " -1 -2", axis = 1)
        logs_to_strings.to_csv('../prepared-logs/' + self.file + '-in.txt', index=False, header=None)
        
        return self.sessions

if __name__ == '__main__':
    p = DataPreparation('3245')
    rules = p.rules_preparation()
    logs = p.logs_preparation()
    rules = p.rules_to_ids(rules)
    indexes = p.indexes_preparation(rules)

    