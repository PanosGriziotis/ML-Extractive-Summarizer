from typing import List
import pandas as pd
from rouge import Rouge 
import pickle
import argparse
import os
import sys
from data_handling import extract_data_from_json, convert_to_dataset, create_sentence_dict
from summarizer import summarize
from nltk import sent_tokenize

class Evaluator():
    """
    class for evaluating a trained svr model and generated system summaries using given test data 
    """
    def __init__(self, test_file:str = None, model_path:str = None,  max_num_of_entries:int=None):

        self.model_path = model_path
        self.test_data = extract_data_from_json (test_file, test_set=True, max_num_of_entries=max_num_of_entries)
        self.x_test, self.y_test = convert_to_dataset(self.test_data, test_set=True) 

    def get_system_summaries (self, max_len:int = None):
        """
        generate summaries from given texts
        """
        
        system_summaries  = []
        
        for entry, x in zip (self.test_data, self.x_test):
                
                sentence_dict = create_sentence_dict(sent_tokenize (entry['text']))
                
                summary = summarize (x, sentence_dict, self.model_path, max_len)
                
                system_summaries.append(summary)

        return system_summaries

    def evaluate_summaries (self, system_summaries: List[str] ):
        """
        evaluate system summaries by comparing them with reference summaries using rouge metrics.
        """

        rouge = Rouge()

        ref_summaries = list()
        for entry in self.test_data:

            ref_summaries.append (entry['summary'])

        rouge_scores = rouge.get_scores(system_summaries, ref_summaries, avg = True)

        return rouge_scores


    def evaluate_model (self):
        """
        compute r squared measure to evaluate model's performance 
        """
        
        model = pickle.load(open(self.model_path, 'rb'))

        X_test = pd.concat(self.x_test).reset_index(drop=True)
        y_test = pd.concat(self.y_test).reset_index(drop=True)

        return model.score (X_test, y_test)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='evaluate summarization system on newsroom test data')

    parser.add_argument(
        '--test_file',
        type = str,
        required=True,
        metavar = '',
        help='test data file path'
        )
    parser.add_argument(
        '-m',
        type=str,
        default = './models/model.svr',
        required= not os.path.isfile('./models/model.svr'),
        metavar='',
        help = 'saved model file path'
        )
    parser.add_argument(
        '--num',
        type = int,
        default = 10,
        metavar='',
        help = 'number of text-summary pairs to include in test set'
        )
    parser.add_argument(
        '-l',
        type = int,
        default = 3,
        metavar = '',
        help='number of sentences to include in each summary'
        )
    parser.add_argument(
        '--out',
        action= 'store_true',
        help = 'if flag is given, generated summaries are saved in files'
        )

    args = parser.parse_args()
    
    # evaluation.py [-h] --test_file  [-m] [--num] [-l] [--out]

    print ('\nEvaluating on test data ...')

    evaluator = Evaluator(args.test_file, args.m, args.num)

    system_summaries = evaluator.get_system_summaries(args.l)

    rouge_scores = evaluator.evaluate_summaries(system_summaries)

    r_score = evaluator.evaluate_model()

    save_dir = './results'
    if not os.path.isdir (save_dir):
        os.mkdir (save_dir)
    with open(save_dir + '/eval.txt', 'w+') as f:
        f.write (f'{rouge_scores}\n\n R-squared:\t{r_score}')
    
    if '--out' in sys.argv:
        save_dir = './results/summaries'
        if not os.path.isdir (save_dir):
            os.mkdir (save_dir)
        for i, summary in enumerate (system_summaries):
            with open (save_dir+f'/{i}.txt', 'w+', encoding="utf-8") as f:
                f.write(summary)
    
    print ('\nDone!')