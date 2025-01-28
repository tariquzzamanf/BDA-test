from random import shuffle
import pandas as pd
from bnlp import BengaliWord2Vec, BengaliCorpus

class BengaliSynonymReplacer:
   def __init__(self):
       self.word2vec_model = BengaliWord2Vec()
       self.bengali_stopwords = set(BengaliCorpus.stopwords)

   def get_synonyms(self, target_word, num_similar=10):
       try:
           similar_words = self.word2vec_model.get_most_similar_words(target_word, topn=num_similar)
           synonym_set = {word_pair[0] for word_pair in similar_words}
           synonym_set.discard(target_word)
           return list(synonym_set)
       except KeyError:
           return []

   def augment(self, original_text, num_replacements, debug=False):
       tokens = original_text.split()
       augmented_tokens = tokens.copy()
       
       candidate_words = list(set(word for word in tokens 
                                if word not in self.bengali_stopwords))
       shuffle(candidate_words)
       
       replacements_made = 0
       for current_word in candidate_words:
           available_synonyms = self.get_synonyms(current_word)
           if available_synonyms:
               chosen_synonym = random.choice(available_synonyms)
               augmented_tokens = [chosen_synonym if token == current_word 
                                 else token for token in augmented_tokens]
               replacements_made += 1
               
           if replacements_made >= num_replacements:
               break

       augmented_text = ' '.join(augmented_tokens)
       return augmented_text + "(sr)" if debug else augmented_text

   def augment_dataset(self, dataframe, num_replacements):
       dataframe['augmented_sentence'] = dataframe.apply(
           lambda row: self.augment(row['sentence1'], num_replacements), 
           axis=1
       )
       dataframe["method"] = f"sr{num_replacements}"
       return dataframe