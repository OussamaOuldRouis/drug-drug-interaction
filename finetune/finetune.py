# Drug-Drug Interaction Chatbot with BioGPT
# This code builds a fine-tuned model for drug-drug interactions using Microsoft's BioGPT

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os
from google.colab import drive
import re

# Mount Google Drive if running in Colab
try:
    drive.mount('/content/drive')
    print("Google Drive mounted successfully")
except:
    print("Not running in Google Colab or Drive already mounted")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare the dataset
def load_data(file_path="db_drug_interactions.csv"):
    """Load the drug-drug interaction dataset"""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {len(df)} entries")
    print(df.head())
    return df

# Create a custom dataset class
class DrugInteractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        drug1 = str(row['Drug 1'])
        drug2 = str(row['Drug 2'])
        interaction = str(row['Interaction Description'])

        # Format as a prompt for training
        prompt = f"What is the interaction between {drug1} and {drug2}?"
        completion = f" {interaction}"

        # For training, combine prompt and completion
        full_text = prompt + completion

        # Tokenize
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create labels (same as input_ids for causal language modeling)
        input_ids = encodings.input_ids.squeeze()
        attention_mask = encodings.attention_mask.squeeze()

        # Special handling for labels - we don't want to compute loss on the prompt part
        labels = input_ids.clone()

        # Find where the prompt ends and completion begins
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        prompt_length = len(prompt_tokens)

        # Set labels for prompt tokens to -100 (ignored in loss calculation)
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Function to load and prepare the model and tokenizer
def prepare_model_and_tokenizer():
    """Load the BioGPT model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
    model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer

# Training function
def train_model(model, tokenizer, train_df, val_df, output_dir="./biogpt_ddi_model"):
    """Fine-tune the model on our drug interaction data"""
    # Create datasets
    train_dataset = DrugInteractionDataset(train_df, tokenizer)
    val_dataset = DrugInteractionDataset(val_df, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),  # Use mixed precision if available
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer

# Inference function for the chatbot
def ddi_chatbot(question, model, tokenizer, max_length=128):
    """Generate a response for a drug interaction question"""
    # Prepare input
    inputs = tokenizer(question, return_tensors="pt").to(device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode and clean the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated part that follows the input question
    response = response[len(question):]

    # Clean up the response
    response = response.strip()

    return response

# Function to extract drug names from user input
def extract_drugs(question):
    """Extract drug names from a question"""
    # Simple pattern for "between drug1 and drug2" or "drug1 and drug2 interaction"
    pattern = r"between\s+([a-zA-Z0-9\s\-]+)\s+and\s+([a-zA-Z0-9\s\-]+)"
    match = re.search(pattern, question.lower())

    if match:
        drug1 = match.group(1).strip()
        drug2 = match.group(2).strip()
        # Remove any trailing characters like '?' or '.'
        drug2 = re.sub(r'[^\w\s\-]', '', drug2).strip()
        return drug1, drug2

    # Try alternative pattern
    pattern = r"([a-zA-Z0-9\s\-]+)\s+and\s+([a-zA-Z0-9\s\-]+)\s+interaction"
    match = re.search(pattern, question.lower())

    if match:
        drug1 = match.group(1).strip()
        drug2 = match.group(2).strip()
        return drug1, drug2

    return None, None

# Main function
def main():
    # Load data
    df = load_data()

    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Training set size: {len(train_df)}, Validation set size: {len(val_df)}")

    # Prepare model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer()

    # Fine-tune the model
    print("Starting model fine-tuning...")
    model, tokenizer = train_model(model, tokenizer, train_df, val_df)

    # Move model to device
    model = model.to(device)

    # Interactive chatbot loop
    print("\n\n=== Drug-Drug Interaction Chatbot ===")
    print("Ask about interactions between drugs. Type 'exit' to quit.")
    print("Example: What is the interaction between aspirin and warfarin?")

    while True:
        question = input("\nYou: ")
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        # Try to extract drug names for formatting
        drug1, drug2 = extract_drugs(question)

        if drug1 and drug2:
            # If drugs were successfully extracted, format the question consistently
            formatted_question = f"What is the interaction between {drug1} and {drug2}?"
            print(f"Formatted question: {formatted_question}")
            response = ddi_chatbot(formatted_question, model, tokenizer)
        else:
            # Use the original question if drug extraction failed
            response = ddi_chatbot(question, model, tokenizer)

        print(f"Chatbot: {response}")

# Example of how to load a saved model and use it
def load_and_use_saved_model(model_path="./biogpt_ddi_model"):
    """Load a saved model and use it for inference"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

    print("\n=== Drug-Drug Interaction Chatbot (Pre-trained) ===")
    print("Ask about interactions between drugs. Type 'exit' to quit.")

    while True:
        question = input("\nYou: ")
        if question.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break

        response = ddi_chatbot(question, model, tokenizer)
        print(f"Chatbot: {response}")

# If the script is run directly
if __name__ == "__main__":
    # Check if we have a saved model
    if os.path.exists("./biogpt_ddi_model"):
        print("Found existing model. Do you want to train a new model or use the existing one?")
        choice = input("Enter 'train' or 'use': ").strip().lower()
        if choice == 'train':
            main()
        else:
            load_and_use_saved_model()
    else:
        # No saved model found, train a new one
        main()