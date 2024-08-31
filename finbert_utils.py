# Import necessary libraries and modules
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # Hugging Face Transformers for FinBERT
import torch 
from typing import Tuple
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the FinBERT tokenizer and model from the pre-trained ProsusAI/finbert
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")  # Tokenizer converts text to tokens that the model can process
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)  # Load the pre-trained model and move it to the appropriate device (GPU/CPU)
labels = ["positive", "negative", "neutral"]  # Define sentiment labels corresponding to the model's output classes

# Function to estimate sentiment of a list of news headlines
def estimate_sentiment(news):
    if news:  # Check if the news list is not empty
        tokens = tokenizer(news, return_tensors="pt", padding=True).to(device)  # Tokenize the news headlines and move tensors to the appropriate device

        # Get model predictions by passing the tokens through the model
        result = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]

        # Apply softmax to the logits to get probabilities, then sum them up across the batch dimension
        result = torch.nn.functional.softmax(torch.sum(result, 0), dim=-1)

        # Get the probability and sentiment label with the highest score
        probability = result[torch.argmax(result)]  # Probability of the most likely sentiment
        sentiment = labels[torch.argmax(result)]  # Corresponding sentiment label
        return probability, sentiment  # Return the probability and sentiment label
    else:
        return 0, labels[-1]  # If no news is provided, return 0 probability and 'neutral' sentiment

# Main block to test the sentiment estimation function
if __name__ == "__main__":
    tensor, sentiment = estimate_sentiment(['markets responded negatively to the news!', 'traders were displeased!'])  
    # Example input: list of news headlines
    print(tensor, sentiment)  # Print the resulting sentiment probability and label
    print(torch.cuda.is_available())  # Print whether CUDA (GPU) is available
