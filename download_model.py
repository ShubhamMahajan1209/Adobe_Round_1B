from sentence_transformers import SentenceTransformer

# This script should be run once with an internet connection
# to download and cache the model for offline use.

def download_and_cache_model():
    """
    Downloads the specified Sentence Transformer model and saves it to a local directory.
    """
    model_name = 'all-mpnet-base-v2'
    # Define the local path where the model will be saved
    local_model_path = './model_cache/all-mpnet-base-v2'
    
    print(f"Downloading model: {model_name}")
    print(f"This may take a few minutes...")
    
    try:
        # SentenceTransformer library automatically handles caching, but we save it 
        # to a specific path to make it portable for the Docker image.
        model = SentenceTransformer(model_name)
        model.save(local_model_path)
        print(f"Model successfully downloaded and saved to '{local_model_path}'")
        print("You can now run the main script offline.")
    except Exception as e:
        print(f"An error occurred during model download: {e}")
        print("Please check your internet connection and try again.")

if __name__ == '__main__':
    download_and_cache_model()
