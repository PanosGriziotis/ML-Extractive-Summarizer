# ML Extractive Summarizer

A feature-based summarization system using Support Vector Regression (SVR) technique for sentence scoring. 

## Installation

1) Run:

```bash
git clone https://github.com/PanosGriziotis/ML-Extractive-Summarizer.git
```

2) install required packages:

```bash
pip install -r requirements.txt
```
Make sure you have installed Python >=3.8

If you aim to train a new model or evaluate the system on test data, you must do the following:

3) download the complete newsroom dataset as described in [here](https://lil.nlp.cornell.edu/newsroom/download/index.html)

2) After downloading, extract downloaded folder to get train, dev and test json.gz files:

```bash
tar xvf newsroom-release.tar
```

## Usage

###  Get summary

- from  a singe txt file:

```bash
python summarizer.py -f <file_path> -o <output_file>
```
- from a folder of multiple txt files:

```bash
python summarizer.py -dir <file_path>
```
**Note**: It's not necessary to train a new model. The summarizer uses by default a pre-trained model from 100 text-summary pairs saved in */models* directory 

Options for **summarizer.py**:

| Options     | Description |
| :---        |    :----:   |
|  -f         | text file path |
|  -dir         | directory path of text files | 
|  -l        | number of sentences to include in summary | 
|  -m        | trained model path | 
|  -o         | file to write generated summary | 


### Train a new model 

- using newsroom train data:

```bash
python model.py --train_file <newsroom_train_file> 
```
- using newsroom train data and dev data:

```bash
python model.py --train_file <newsroom_train_file> --dev <newsroom_dev_file>
```

Options for **model.py**:

| Options     | Description |
| :---        |    :----:   |
|  --train_file         | train data file path |
|  --dev         |  developement data file path | 
|  --num         | number of text-summary pairs to include in train set | 
|  --split       | proportion of dev data | 
|  -o         | output folder directory to save trained model  | 

### Evaluate summarizer 

- using newsroom test data:

```bash
python model.py --test_file <newsroom_test_file> 
```
Model evaluation is done with r-squared measurement. System summaries are evaluated using 3 different types of rouge metrics (rouge-1, rouge-2, rouge-l)

Options for **model.py**:

| Options     | Description |
| :---        |    :----:   |
|  --test_file         | test data file path |
|  -m         |   saved model file path| 
|  --num         | number of text-summary pairs to include in test set | 
|  -l        | number of sentences to include in each summary | 
|  --out        | if flag is given, generated summaries are saved in files | 
