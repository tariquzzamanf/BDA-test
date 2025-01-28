from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize

class BackTranslationAugmentor:
   def __init__(self, 
                bn_to_en_model="csebuetnlp/banglat5_nmt_bn_en",
                en_to_bn_model="csebuetnlp/banglat5_nmt_en_bn"):
       # Initialize models and tokenizers
       self.bn_to_en_model = AutoModelForSeq2SeqLM.from_pretrained(bn_to_en_model)
       self.bn_to_en_tokenizer = AutoTokenizer.from_pretrained(bn_to_en_model, use_fast=False)

       self.en_to_bn_model = AutoModelForSeq2SeqLM.from_pretrained(en_to_bn_model) 
       self.en_to_bn_tokenizer = AutoTokenizer.from_pretrained(en_to_bn_model, use_fast=False)

   def augment(self, text, debug=False):
       """Augment Bengali text via back-translation through English."""
       # Bengali -> English
       bn_encoded = self.bn_to_en_tokenizer.encode(normalize(text), return_tensors="pt")
       en_outputs = self.bn_to_en_model.generate(bn_encoded)
       en_text = self.bn_to_en_tokenizer.decode(en_outputs[0], skip_special_tokens=True)

       # English -> Bengali  
       en_encoded = self.en_to_bn_tokenizer.encode(normalize(en_text), return_tensors="pt")
       bn_outputs = self.en_to_bn_model.generate(en_encoded)
       bn_text = self.en_to_bn_tokenizer.decode(bn_outputs[0], skip_special_tokens=True)

       return bn_text + '(bt)' if debug else bn_text