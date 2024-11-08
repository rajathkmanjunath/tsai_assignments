import nltk

def download_nltk_resources():
    resources = [
        'punkt',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)
        print(f"Downloaded {resource} successfully!")

if __name__ == "__main__":
    download_nltk_resources() 