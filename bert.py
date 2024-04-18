from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-incased')
model = BertModel.from_pretrained('bert-base-uncased')


# Define Function for processing input text
def process_text(input_text):
    # Tokenize input text
    tokenized_text = tokenizer(input_text, return_tensors='pt')
    return tokenized_text

# Define function for performing inference
def run_inference(tokenized_text):
    #Pass tokenized input through BERT model
    with torch.no_grad():
        outputs = model(**tokenized_text)
    return outputs

# Main func for testing

def main():
    input_text = "input text goes here."

    #process i/p text
    tokkenized_text = process_text(input_text)


    #perfrom inference
    outputs = run_inference(tokkenized_text)

    print(outputs)


# Entry point of the script
if __name__ == "__main__":
    main()
    