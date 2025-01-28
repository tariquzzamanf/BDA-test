from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from normalizer import normalize
import pandas as pd

class BanglaParaphraseGenerator:
   def __init__(self):
       self.model = AutoModelForSeq2SeqLM.from_pretrained("csebuetnlp/banglat5_banglaparaphrase")
       self.tokenizer = AutoTokenizer.from_pretrained(
           "csebuetnlp/banglat5_banglaparaphrase", 
           use_fast=False
       )

   def generate_paraphrase(self, text, max_length=100):
       """Generate paraphrase for given Bengali text."""
       normalized_text = normalize(text)
       input_ids = self.tokenizer(normalized_text, return_tensors="pt").input_ids
       output_ids = self.model.generate(input_ids, max_length=max_length)
       paraphrased_text = self.tokenizer.batch_decode(
           output_ids, 
           skip_special_tokens=True
       )[0]
       return paraphrased_text

   def augment_dataset(self, dataframe):
       """Augment dataset with paraphrased sentences."""
       processed_df = dataframe[['sentence1', 'label']].copy()
       processed_df = processed_df.rename(columns={'sentence1': 'original_sentence'})
       
       processed_df['augmented_sentence'] = processed_df.apply(
           lambda row: self.generate_paraphrase(row['original_sentence']), 
           axis=1
       )
       processed_df['method'] = 'pp'
       
       return processed_df