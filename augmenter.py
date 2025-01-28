import random
from methods.synonym import BengaliSynonymReplacer
from
# Define augmentation functions
def aug1(text):
    return f"1 AUG {text}"

def aug2(text):
    return f"2 AUG {text}"

def aug3(text):
    return f"3 AUG {text}"

def aug4(text):
    return f"4 AUG {text.upper()}"

def bda_augmenter(value):
    """Randomly selects and applies an augmentation method"""
    aug_str = str(value)  # Convert to string for safety
    
    # Create a dispatch dictionary
    augmentations = {
        1: aug1,
        2: aug2,
        3: aug3,
        4: aug4
    }
    
    # Randomly select augmentation
    selected = random.randint(1, 4)
    augmented_text = augmentations[selected](aug_str)
    
    return augmented_text