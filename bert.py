from transformers import BertTokenizer, BertModel
from tokenizer import Tokenizer
import torch

# Path to the directory containing your pretrained BERT model files
pretrained_path = "C:\\Users\\appuv\\backend\\"


# Load pre-trained BERT model and tokenizer from local files
tokenizer = BertTokenizer.from_pretrained(pretrained_path)
model = BertModel.from_pretrained(pretrained_path)

# Define the token dictionary for your specific task
token_dict = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "biomedical": 4,
    "ner": 5,
    "protein": 6,
    "gene": 7,
    "disease": 8,
    "treatment": 9,
    # Add more biomedical tokens as needed
}

# Define a custom tokenizer based on your token dictionary
custom_tokenizer = Tokenizer(token_dict)

# Define a function to process input text using the custom tokenizer
def process_text(input_text):
    # Tokenize input text using the custom tokenizer
    tokenized_text = custom_tokenizer.tokenize(input_text)
    return tokenized_text

# Define function for performing inference
def run_inference(tokenized_text):
    # Pass tokenized input through BERT model
    with torch.no_grad():
        outputs = model(**tokenized_text)
    return outputs

# Main function for testing
def main():
    input_text = "input text goes here."

    # Process input text
    tokenized_text = process_text(input_text)

    # Perform inference
    outputs = run_inference(tokenized_text)

    print(outputs)

# Entry point of the script
if __name__ == "__main__":
    main()
