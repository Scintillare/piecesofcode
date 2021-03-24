import torch
from torch.utils.data import DataLoader, sampler, SubsetRandomSampler
from torchvision import transforms, datasets, models
import torch.nn as nn

# Data science tools
import numpy as np
# Visualizations
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# FIXME deprecated. I need another training function

def train_model(model, 
        dataloaders, 
        criterion, 
        optimizer, 
        device, 
        save_file_name,
        max_epochs_stop=3,
        n_epochs=20,
        print_every=2,
        is_inception=False):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        dataloaders (PyTorch dataloader): dictionary with training and validation dataloaders 
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        device: torch.device - cuda or cpu
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats
        is_inception: True if model is inception

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = {'acc': {'train': [], 'val': []},
                'loss': {'train': [], 'val': []}}

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):
        start = timer()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            phase_loss = 0.0
            phase_corrects = 0

            # Iterate over data.
            for ii, (data, labels) in enumerate(dataloaders[phase]):
                data = data.to(device)
                labels = labels.to(device)

                # zero the parameter gradients # Clear gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(data)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(data)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    # Track loss by multiplying average loss by number of examples in batch
                    phase_loss += loss.item() * data.size(0)
                    phase_corrects += torch.sum(preds == labels.data)
                    
                    # Track training progress
                    print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(dataloaders[phase]):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.', end='\r')
            
            epoch_loss = phase_loss / len(dataloaders[phase].sampler)
            epoch_acc = phase_corrects.double() / len(dataloaders[phase].sampler)

            history['loss'][phase].append(epoch_loss)
            history['acc'][phase].append(epoch_acc)

            #  Print training and validation results
            if phase == 'val' and (epoch + 1) % print_every == 0:
                train_loss, valid_loss = history['loss']['train'][-1], history['loss']['val'][-1]
                train_acc, valid_acc = history['acc']['train'][-1], history['acc']['val'][-1]
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )
                
                plot_training_stats(history)
                # num_correct = 0
                # num_ex = 0
                # for data, labels in dataloaders['test']:
                #     data, labels = data.to(device), labels.to(device)
                #     probs = model(data)
                #     num_correct += sum((torch.argmax(probs, 1) == labels).float())
                #     num_ex += len(labels)
                # acc = num_correct/num_ex
                # print(acc)

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            
            # Save the model if validation loss decreases
            if phase == 'val' and epoch_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), r'./classifier/'+save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = epoch_loss
                valid_max_acc = epoch_acc
                best_epoch = epoch
                plot_training_stats(history)
            # Otherwise increment count of epochs with no improvement
            elif phase == 'val':
                epochs_no_improve += 1
                torch.save(model.state_dict(), r'./classifier/'+save_file_name+str(epoch))
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_max_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )


                    # Load the best state dict
                    model.load_state_dict(torch.load(r'./classifier/'+save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer
                    torch.save(model.state_dict(), r'./classifier/'+save_file_name)  # !!
                    return model, history
                             
        print()

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    torch.save(model.state_dict(), r'./classifier/'+save_file_name) 
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_max_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (n_epochs):.2f} seconds per epoch.'
    )
    
    return model, history


def plot_training_stats(history: dict):
    plt.figure(figsize=(8, 6))
    for c in ['train', 'val']:
        plt.plot(
            history['loss'][c], label='loss '+c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Losses')
    # plt.show()
    plt.savefig("loss.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    for c in ['train', 'val']:
        plt.plot(
             history['acc'][c], label='acc ' + c)#100 *
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    # plt.show()
    plt.savefig("acc.png")
    plt.close()
            
