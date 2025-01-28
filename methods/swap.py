import random

class RandomSwapper:
   def __init__(self):
       pass

   def swap_random_words(self, word_list):
       """Randomly swap two words in a list if list has 2+ words."""
       if len(word_list) > 1:
           idx1, idx2 = random.sample(range(len(word_list)), 2)
           word_list[idx1], word_list[idx2] = word_list[idx2], word_list[idx1]
       return word_list

   def augment_text(self, text, num_swaps, debug=False):
       """Augment text by performing random word swaps."""
       tokens = text.split()
       for _ in range(num_swaps):
           self.swap_random_words(tokens)
       
       augmented_text = ' '.join(tokens)
       return augmented_text + " (rs)" if debug else augmented_text

   def augment_dataset(self, dataframe, num_swaps):
       """Apply random swapping augmentation to dataset."""
       dataframe['augmented_sentence'] = [
           self.augment_text(text, num_swaps) 
           for text in dataframe['sentence1']
       ]
       dataframe['method'] = f'rs{num_swaps}'
       return dataframe