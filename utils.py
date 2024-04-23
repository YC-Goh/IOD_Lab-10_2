import numpy as np
import pandas as pd
import regex as re
import joblib
import en_core_web_sm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC

nlp = en_core_web_sm.load()
classifier = LinearSVC()

def clean_text(text):
    ##  Remove all punctuation except those that may affect the actual meaning of the sentence
    text = re.sub(r'[^a-zA-Z0-9 :;,\.!\?\'%\-]', ' ', text)
    ##  Remove spaces between consecutive pairs of certain punctuation marks
    text = re.sub(r'(\.) +(\.)', r'\1\2', text)
    text = re.sub(r'([!\?]) +([!\?])', r'\1\2', text)
    ##  Always add a space in front of commas, semicolons and colons (and remove excess when repeated)
    text = re.sub(r'([:;,])\1*', r'\1 ', text)
    ##  If a period is not for decimal notation, add a space in front
    text = re.sub(r'\.(?!(?<=[0-9]\.)[0-9]|\.)', '. ', text)
    ##  Add spaces in front of sequences of exclamation and question marks
    text = re.sub(r'([!\?]+)', r'\1 ', text)
    ##  For multiple periods in a row, standardise with a triple period (for the ... punctuation mark)
    text = re.sub(r'(\.)\1{2,}', r'\1\1\1', text)
    ##  For multiple of exclamation and question marks in a row, standardise with a double mark
    text = re.sub(r'([!?])\1{1,}', r'\1\1', text)
    ##  For multiple pairs of exclamation and question marks in a row, standardise with a single of each
    text = re.sub(r'([!?])\1([!?])\2', r'\1\2', text)
    ##  Remove all single quotes that are not immediately between 2 letters
    text = re.sub(r'\'(?!(?<=[a-zA-Z]\')[a-zA-Z])', ' ', text)
    ##  Remove percent signs that do not signal actual percentages
    text = re.sub(r'(?<![0-9])%', ' ', text)
    ##  Dashes connecting letters can arguably usually be replaced with spaces and still keep selamtic meaning, 
    ##  but when they connect numbers they usually indicate a range and should be preserved
    text = re.sub(r'(\d)\s*(-)\s*(\d)', r'\1\2\3', text)
    text = re.sub(r'-(?!(?<=[0-9]-)[0-9])', ' ', text)
    ##  Reduce spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    ##  Vowels repeated more than twice are likely being done for emphasis
    text = re.sub(r'([aeiouy])\1\1+', r'\1\1', text)
    ##  Multiple vowels repeated twice in a row are also likely being done for emphasis
    ##  Note that by the definition of the previous line, it is impossible for capture group 1 to be the same vowel as capture group 2
    text = re.sub(r'([aeiouy])\1([aeiouy])\2', r'\1\2', text)

    return text
	
def convert_text(text):
    sent = nlp(text)
    ents = {x.text: x for x in sent.ents}
    tokens = []
    for w in sent:
        if w.is_stop or w.is_punct:
            continue
        if w.text in ents:
            tokens.append(w.text)
        else:
            tokens.append(w.lemma_.lower())
    text = ' '.join(tokens)

    return text


class preprocessor(TransformerMixin, BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(clean_text).apply(convert_text)