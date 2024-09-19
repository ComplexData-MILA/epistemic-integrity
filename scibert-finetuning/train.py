import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import get_scheduler
import wandb
import os
from tqdm import tqdm
from data_prep import AssertivenessDataset
from tuned_model import ModifiedAssertivenessModel
import pandas as pd

def train(model, dataloader, optimizer, scheduler, criterion, epoch, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        scores = batch['score'].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.squeeze(), scores)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
        # Log metrics to WandB
        wandb.log({"train_loss": loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")
    return avg_loss

def validate(model, dataloader, criterion, device, phase="Validation"):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{phase}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.squeeze(), scores)

            total_loss += loss.item()
            
            # Log validation loss to WandB
            wandb.log({f"{phase.lower()}_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    print(f"{phase} Loss: {avg_loss:.4f}")
    return avg_loss

def main():

    # Initialize wandb run
    wandb.init(project="assertiveness-finetuning", 
               config={
        "epochs": 40,
        "batch_size": 16,
        "learning_rate": 5e-3,
        "weight_decay": 1e-2,
        "dropout_rate": 0.01,
        "warmup_steps": 3
    })

    config = wandb.config

    epochs = config.epochs # type: ignore
    batch_size = config.batch_size  # type: ignore
    learning_rate = config.learning_rate # type: ignore
    weight_decay = config.weight_decay # type: ignore
    dropout_rate = config.dropout_rate # type: ignore
    warmup_steps = config.warmup_steps # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training data
    df = pd.read_csv('scibert-finetuning/data/train_data.csv')
    texts = df['text'].tolist()
    scores = df['assertiveness'].tolist()

    dataset = AssertivenessDataset(texts, scores)
    
    # Split the dataset into training and validation sets (e.g., 80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader for batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = ModifiedAssertivenessModel(add_activation=False).to(device)
    
    # Add dropout to the model
    model.base_model.dropout = nn.Dropout(dropout_rate)

    # Loss function (MSE for regression)
    criterion = nn.MSELoss()

    # Optimizer (AdamW) and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

    # Track the model with WandB
    wandb.watch(model, log="all") # type: ignore

    # Early stopping variables
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    # Training loop
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, scheduler, criterion, epoch, device)
        val_loss = validate(model, val_loader, criterion, device, phase="Validation")
        
        # Log average losses to WandB
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": train_loss,
            "avg_val_loss": val_loss
        })

        # set save_dir to be /$SCRATCH/scibert_finetuned
        save_dir = os.path.join(os.environ['SCRATCH'], 'scibert_finetuned')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter if validation loss improves
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth"))
            wandb.save(os.path.join(save_dir, f"best_model_epoch_{epoch+1}.pth"))
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
        
        # Stop training if patience is exceeded
        if patience_counter >= patience:
            print(f"Stopping early after {patience} epochs of no improvement.")
            break

    print("Training complete.")

    # Optionally, test the model on test set if early stopping occurs
    df = pd.read_csv('scibert-finetuning/data/test_data.csv')
    texts = df['text'].tolist()
    scores = df['assertiveness'].tolist()

    dataset = AssertivenessDataset(texts, scores)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_loss = validate(model, test_loader, criterion, device, phase="Test")
    print(f"Test Loss: {test_loss:.4f}")
    wandb.log({"test_loss": test_loss})

    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
