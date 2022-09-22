from typing import List
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy 
from nltk import  sent_tokenize
import pandas as pd


class FeatureExtractor():
    """
    class for generating feature matrix and score vector for given text and summary
    """
    def __init__(self, input_sentences: List[str] = None):

        self.nlp = spacy.load("en_core_web_sm")
        self.input_sentences = input_sentences  
        self.tokenized_sentences = self.normalize_sentences(self.input_sentences) #[['a','b','c'],['d','e',f']]
        self.cwf_scores = list()
        self.entity_scores = list()
        self.sp_scores = list()

    def get_feature_matrix (self):
        
        """
        create a dataframe of shape (n, 3). The rows represent sentences and  the columns represent sentence related features
        """
        
        self.get_content_word_frequency ()
        self.get_entity_ratio()
        self.get_sentence_position_score()
        
        f_matrix = pd.DataFrame({'cwf': self.cwf_scores, 'ents': self.entity_scores, 'sp': self.sp_scores})
        
        return f_matrix
            
    def normalize_sentences (self, list_of_sentences:list):
        
        """
        replace tokens with lemmas, discard stop words (keep only content words), drop punctuation and numbers for given sentences
        """

        sentences = list(self.nlp.pipe(list_of_sentences, disable=[ "parser", "ner"]))
        return [[str(token.lemma_) for token in sentence if (not token.is_stop and not token.is_punct and not token.like_num)] for sentence in sentences]

    def get_content_word_frequency (self):

        """
        Find the average word probability (word occurencies / total number of words in text) of all content words in a sentence
        """
        
        cw_list = list(itertools.chain.from_iterable(self.tokenized_sentences))
        
        cw_frequencies = {} 
        for w in cw_list:
            if w in cw_frequencies:
                cw_frequencies[w] += 1
            else:
                cw_frequencies[w] = 1

        for sentence in self.tokenized_sentences:
            
            probs = [cw_frequencies[w] / len(cw_list) for w in sentence ] # pc{ti) (n/N)

            self.cwf_scores.append(sum(probs) / (len(sentence) + 1)) # sum(pc(ti)) / |s|+1

        
        return self.cwf_scores

    def get_entity_ratio (self):

        """
        find the ratio of name entities (eg. ORG, GPE) to the total number of words in a sentence
        """

        # join tokenized sentences
        norm_sentences = [' '.join(sentence) for sentence in self.tokenized_sentences]
            
        for sentence in self.nlp.pipe(norm_sentences, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
            num_ents = 0
            for ent in sentence.ents:
                num_ents += 1
            self.entity_scores.append(num_ents / (len((str(sentence)).split()) + 1))

        return self.entity_scores

    def get_sentence_position_score (self):
        
        """
        find the ratio of the position of a sentence in its document to the total number of sentences of the document
        """

        for index, sentence in enumerate(self.input_sentences, 1):
            self.sp_scores.append (index / len(self.input_sentences)) 

        return self.sp_scores

    def get_similarity_scores (self, summary_doc: str):

        """
        Find the cosine similarity score between a sentence's and summary's word-count vectors.
        """

        y_scores = list()

        ref_doc = ' '.join (list(itertools.chain.from_iterable(self.normalize_sentences(sent_tokenize(summary_doc)))))  
        vectorizer = CountVectorizer(max_features=50)

        for sentence in self.tokenized_sentences:   
            vectors = vectorizer.fit_transform([' '.join(sentence), ref_doc])
            y_scores.append(cosine_similarity(vectors[0], vectors).item((0,1))) 
                
        return pd.DataFrame({'scores':y_scores})
