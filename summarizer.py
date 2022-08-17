from typing import Dict
import pandas as pd
import pickle
import pandas as pd
import argparse
import sys
import os
from data_handling import convert_to_dataset, create_sentence_dict, open_file
from nltk import sent_tokenize
from pathlib import Path

def summarize (X_data: pd.DataFrame, sentence_dict: Dict[int, str], model_path: str = None , max_len: int = None):
    
    """
    construct a summary from given feature matrix using model predictions and selecting the top k most important sentences from original text
    """

    model = pickle.load(open(model_path, 'rb'))

    y_pred = model.predict(X_data)

    X_data = X_data.assign(y_pred = y_pred)

    n_best_indeces = list(X_data.nlargest(max_len, 'y_pred').index)

    summary_sentences = [sentence_dict[index] for index in n_best_indeces]
    
    return ' '.join(summary_sentences)

def get_summary_from_txt (file: str, model_path: str = None , max_len: int = None):
    
    """
    get summary from a given txt file
    """

    if not file.endswith ('.txt'):
        raise Exception ('Non compatible file format. Only .txt files are accepted')
    
    text = open_file(file, clean_title=True)
    
    sentence_dict = create_sentence_dict(sent_tokenize(text))
    
    summary = summarize(convert_to_dataset(text), sentence_dict , model_path, max_len)

    return summary

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='generate summary for a given text')

    parser.add_argument(
        '-f',
        type = str,
        metavar = '',
        required=not '-dir' in sys.argv,
        help='text file path'
        )
    parser.add_argument(
        '-dir',
        type = str,
        metavar = '',
        required=not '-f' in sys.argv,
        help='directory path of text files'
        )
    parser.add_argument(
        '-l',
        type = int,
        default = 3,
        metavar = '',
        help='number of sentences to include in summary'
        )
    parser.add_argument(
        '-m',
        type = str,
        default = './models/model.svr',
        required= not os.path.isfile('./models/model.svr'),
        metavar = '',
        help='trained model path'
        )
    parser.add_argument(
        '-o',
        type = str,
        metavar = '',
        help='file to write generated summary'
        )
    args = parser.parse_args()
    
    # summarizer.py [-h] -f  -dir  [-l] [-m] [-o]

    print ('\nGenerating summary/ies ...\n')

    if '-dir' in sys.argv:
        summaries = []
        files = Path(args.dir)
        for file in files.iterdir():
            summaries.append (get_summary_from_txt(str(file), args.m, args.l))
        save_dir = './summaries'
        if not os.path.isdir (save_dir):
            os.mkdir (save_dir)
        for i, summary in enumerate (summaries):
            with open (save_dir+f'/{i}.txt', 'w+', encoding="utf-8") as f:
                f.write(summary)
    
    elif '-f' in sys.argv:
        summary = get_summary_from_txt(args.f, args.m, args.l)
        if '-o' in sys.argv:
            with open (args.o, 'w') as writer:
                writer.write(summary)
        else:
            print (summary)
    
    print ('\nDone!')
    
