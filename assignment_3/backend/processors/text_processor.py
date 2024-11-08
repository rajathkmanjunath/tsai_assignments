from .base_processor import BaseProcessor
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import random

class TextProcessor(BaseProcessor):
    def __init__(self):
        # Initialize NLTK resources
        self._download_nltk_resources()
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize augmenters
        self.augmenters = [
            naw.SynonymAug(aug_src='wordnet'),
            naw.ContextualWordEmbsAug(model_path='bert-base-uncased'),
            naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de', 
                to_model_name='facebook/wmt19-de-en'
            ),
            naw.RandomWordAug(action="swap"),
            nac.RandomCharAug(action="substitute")
        ]

    def _download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = [
            'punkt',
            'stopwords',
            'wordnet',
            'averaged_perceptron_tagger',
            'omw-1.4'  # Open Multilingual Wordnet
        ]
        
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Error downloading {resource}: {str(e)}")

    def preprocess(self, text):
        """Apply all preprocessing steps"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        
        # Join tokens back into text
        processed_text = " ".join(tokens)
        
        return processed_text

    def augment(self, text):
        """Randomly apply 1-3 augmentation techniques"""
        # Randomly select number of augmentations to apply (1-3)
        num_augs = random.randint(1, 3)
        
        # Randomly select augmenters
        selected_augmenters = random.sample(self.augmenters, num_augs)
        
        # Apply selected augmentations sequentially
        augmented_text = text
        applied_techniques = []
        
        for augmenter in selected_augmenters:
            try:
                augmented_text = augmenter.augment(augmented_text)[0]
                applied_techniques.append(augmenter.__class__.__name__)
            except Exception as e:
                print(f"Augmentation failed with {augmenter.__class__.__name__}: {str(e)}")
                continue
        
        return augmented_text, applied_techniques 