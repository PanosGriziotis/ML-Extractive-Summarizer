from nltk import sent_tokenize
import pandas as pd
import json, gzip
from feature_extractor import FeatureExtractor
from typing import List, Dict, Any, Union
import random

def parse_json_file (line: Dict[str, Any], data_entry: dict = None):

    """create a dictionary of text and summary extracted from a line in newsroom json file"""

    json_object = json.loads(line)
    data_entry['text'] =  json_object['text']
    data_entry['summary'] = json_object['summary']
    return data_entry

def keep_main_text (text: str):

    """check for title or metadata in given text and discard them"""

    new_lines = text.split('\n')
    new_text = []
    for line in new_lines:
        if len (line.split(' ')) > 10: 
            new_text.append(line)
    return ' '.join (new_text)

def extract_data_from_json (data_file: str, test_set:bool = False, dev_data_file: str=None, split_ratio: int=None, max_num_of_entries: int = None):

    """get explicit number of data pairs (text-summary) for train, dev or test data using newsroom json files"""

    data = list()
    dev_data = list()

    if dev_data_file != None:
        data_files = [data_file, dev_data_file]    
    else:
        data_files = [data_file]
    
    if not test_set:
        num_dev = round((max_num_of_entries * split_ratio) / 100)             
        if num_dev < 1:
            raise ValueError('Dev set must have at least one entry. Try different values for either split_ratio or max_num_of_entries parameters') 
            
    for file in data_files:
        
        if file.endswith ('.jsonl.gz'):
            r_file = gzip.open(file, 'r')
        elif file.endswith ('.jsonl'):
            r_file = open (file, 'r')      
        else: 
            raise Exception ('Non compatible file format. Only json and jsonl formats are accepted')
        
        num_entry = 0

        while True:

            line = r_file.readline()
            num_entry += 1
 
            # train or test set 
            if data_files.index (file) == 0:

                data_entry = {}
                parse_json_file(line, data_entry)
                if test_set:
                    data_entry['text'] = keep_main_text (data_entry['text'], test_set=True)
                data.append (data_entry)

                if num_entry == max_num_of_entries:
                    break

            # dev set
            if data_files.index(file) == 1:

                data_entry = {}
                parse_json_file(line, data_entry)
                dev_data.append (data_entry)

                if num_entry == num_dev:
                    break 
    
        r_file.close()
    
    if not test_set:
        if len(dev_data) == 0:
            dev_data = random.choices(data, k=num_dev)
        return data, dev_data 
    else:
        return data
    
def convert_to_dataset (text_data: Union[list[dict], str], test_set:bool = False):

    """create feature matrix of shape (n, 3) and/or score vector of shape (n, 1) for each text in given data, with n = number of sentences in a given text"""

    X_data = list()
    y_data = list()
    
    if type(text_data) == str:
        
        fe = FeatureExtractor(sent_tokenize(text_data))

        X_data.append(fe.get_feature_matrix()) # > pd.Daraframe (n,4)
    
    elif type(text_data) == list:

        for entry in text_data:

            fe = FeatureExtractor(sent_tokenize(entry['text'])) 
            
            X_data.append(fe.get_feature_matrix()) # > pd.Dataframe (n,4)

            y_data.append(fe.get_similarity_scores (entry['summary'])) # > pd.Dataframe (n,1)
    
    if len(y_data) > 0:
        if not test_set:
            return pd.concat(X_data).reset_index(drop=True), pd.concat(y_data).reset_index(drop=True)
        else:
            return X_data, y_data
    else:
        return X_data[0]
        
def create_sentence_dict (text: List[str]):
    
    """create an index, sentence dictionary for each given text"""
    
    i = 0
    sentence_dict = dict()
    for sentence in text:
        sentence_dict [i] = sentence
        i += 1
    return sentence_dict

def open_file(txt_file, clean_title=True):

    """open a given txt file and return plain text"""

    with open(txt_file, 'r', encoding='utf-8') as file:
        text = file.read()
        if clean_title:
            return keep_main_text(text)
        else:
            return text

