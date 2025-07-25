import torch
from tqdm import tqdm
import numpy as np
from optimizers import get_optimizer
import copy

def run_one_epoch(model, loader, criterion, optimizer=None, device="cuda"):
    if optimizer is not None:
        is_in_training_mode = True
    else:
        is_in_training_mode = False

    model.train(is_in_training_mode)
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_scores = []

    progress_bar = tqdm(loader, leave=True)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        with torch.set_grad_enabled(is_in_training_mode):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if is_in_training_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        scores = torch.softmax(outputs, dim=1)
        preds = torch.argmax(scores, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_scores.extend(scores.cpu().detach().numpy())
        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_scores)


def train_fold(model, train_loader, val_loader, criterion, args, device):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    if args.strategy == 'finetune':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        head_optimizer = get_optimizer(model.fc.parameters(), args.optimizer, args.lr)
        
        print(f"Stage 1 (Finetune): Training only the head for {args.head_epochs} epochs.")
        for epoch in range(args.head_epochs):
            run_one_epoch(model, train_loader, criterion, head_optimizer, device)
        
        for param in model.parameters():
            param.requires_grad = True

        main_lr = args.lr / 10.0
        main_optimizer = get_optimizer(model.parameters(), args.optimizer, main_lr)
        print(f"Stage 2 (Finetune): Training full network for {args.max_epochs} epochs.")

    elif args.strategy == 'baseline':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        main_lr = args.lr
        main_optimizer = get_optimizer(model.fc.parameters(), args.optimizer, main_lr)
        print(f"Strategy (FFE): Training only the head for {args.max_epochs} epochs.")

    else:
        for param in model.parameters():
            param.requires_grad = True
        
        main_lr = args.lr
        main_optimizer = get_optimizer(model.parameters(), args.optimizer, main_lr)
        print(f"Strategy ({args.strategy}): Training full network (no head warm-up) for {args.max_epochs} epochs.")

    for epoch in range(args.max_epochs):
        run_one_epoch(model, train_loader, criterion, main_optimizer, device)
        
        validation_loss, validation_labels, validation_predictions, validation_scores = run_one_epoch(model, val_loader, criterion, None, device)

        print(f"  Epoch {epoch+1}/{args.max_epochs} -> Validation Loss: {validation_loss:.4f}")

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= args.patience:
                print(f"Early stopped after {args.patience} epochs.")
                break
        
    return best_model_state