import torch
from torch.utils.data import DataLoader, sampler, SubsetRandomSampler
from torchvision import transforms, datasets, models
import torch.nn as nn

import numpy as np
import os
import gc

# Image manipulations
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_resnet(num_classes):
    model = models.resnet50(pretrained=False)
    n_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, num_classes))#, nn.LogSoftmax(dim=1))
    return model


# From repository confidence-calibrated adversarial training
def find_last_checkpoint(model_file):
    """
    Name of model should be in format [somename].pth.tar.epoch_number(digit).
    Final model should be saved as [somename].pth.tar
    State file.

    :param model_file: base state file
    :type model_file: str
    :return: state file of ongoing training
    :rtype: str
    """

    base_directory = os.path.dirname(os.path.realpath(model_file))
    file_name = os.path.basename(model_file)

    if os.path.exists(base_directory):
        state_files = []
        files = [os.path.basename(f) for f in os.listdir(base_directory) if os.path.isfile(os.path.join(base_directory, f))]

        for file in files:
            if file.find(file_name) >= 0 and file != file_name:
                state_files.append(file)

        if len(state_files) > 0:
            epochs = [state_files[i].replace(file_name, '').replace('.pth.tar', '').replace('.', '') for i in range(len(state_files))]
            epochs = [epoch for epoch in epochs if epoch.isdigit()]
            epochs = list(map(int, epochs))
            epochs = [epoch for epoch in epochs if epoch >= 0]

            if len(epochs) > 0:
                # list is not ordered by epochs!
                i = np.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i])

#common.torch.classification_error confidence-calibrated-adv-tr
def classification_error(logits, targets, reduction='mean'):
    """
    Accuracy.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduce: reduce to number or keep per element
    :type reduce: bool
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        values, indices = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)
    else:
        indices = torch.round(torch.nn.functional.sigmoid(logits)).view(-1)

    errors = torch.clamp(torch.abs(indices.long() - targets.long()), max=1)
    if reduction == 'mean':
        return torch.mean(errors.float())
    elif reduction == 'sum':
        return torch.sum(errors.float())
    else:
        return errors


#common.torch.classification_loss confidence-calibrated-adv-tr
def classification_loss(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        return torch.nn.functional.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        # probability 1 is class 1
        # probability 0 is class 0
        return torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(logits).view(-1), targets.float(), reduction=reduction)


#common.torch.as_variable confidence-calibrated-adv-tr
def as_torch_variable(mixed, cuda=False, requires_grad=False):
    """
    Get a tensor or numpy array as variable.

    :param mixed: input tensor
    :type mixed: torch.Tensor or numpy.ndarray
    :param device: gpu or not
    :type device: bool
    :param requires_grad: gradients
    :type requires_grad: bool
    :return: variable
    :rtype: torch.autograd.Variable
    """

    assert isinstance(mixed, np.ndarray) or isinstance(mixed, torch.Tensor), 'input needs to be numpy.ndarray or torch.Tensor'

    if isinstance(mixed, np.ndarray):
        mixed = torch.from_numpy(mixed)

    if cuda:
        mixed = mixed.cuda()
    return torch.autograd.Variable(mixed, requires_grad)

#common.numpy.concatenate confidence-calibrated-adv-tr
def np_concatenate(array1, array2, axis=0):
    """
    Basically a wrapper for numpy.concatenate, with the exception
    that the array itself is returned if its None or evaluates to False.

    :param array1: input array or None
    :type array1: mixed
    :param array2: input array
    :type array2: numpy.ndarray
    :param axis: axis to concatenate
    :type axis: int
    :return: concatenated array
    :rtype: numpy.ndarray
    """

    assert isinstance(array2, np.ndarray)
    if array1 is not None:
        assert isinstance(array1, np.ndarray)
        return np.concatenate((array1, array2), axis=axis)
    else:
        return array2

#CCAT
def is_cuda(mixed):
    """
    Check if model/tensor is on CUDA.

    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.autograd.Variable or torch.Tensor
    :return: on cuda
    :rtype: bool
    """

    assert isinstance(mixed, torch.nn.Module) or isinstance(mixed, torch.autograd.Variable) \
        or isinstance(mixed, torch.Tensor), 'mixed has to be torch.nn.Module, torch.autograd.Variable or torch.Tensor'

    is_cuda = False
    if isinstance(mixed, torch.nn.Module):
        is_cuda = True
        for parameters in list(mixed.parameters()):
            is_cuda = is_cuda and parameters.is_cuda
    if isinstance(mixed, torch.autograd.Variable):
        is_cuda = mixed.is_cuda
    if isinstance(mixed, torch.Tensor):
        is_cuda = mixed.is_cuda

    return is_cuda

#CCAT
def test(model, testset, cuda=False, return_labels=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    :param return_labels: whether to additionally return labels
    :type return_labels: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    # assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and is_cuda(model)) or (not cuda and not is_cuda(model))

    probabilities = None
    labels = None

    # should work with and without labels
    for b, data in enumerate(testset):
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
        else:
            inputs = data

        if return_labels is True:
            assert len(data) >= 2
            labels = np_concatenate(labels, data[1].numpy())

        assert isinstance(inputs, torch.Tensor)

        inputs = as_torch_variable(inputs, cuda)
        # inputs = inputs.permute(0, 3, 1, 2)

        logits = model(inputs)
        probabilities_ = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = np_concatenate(probabilities, probabilities_)
        gc.collect() #XXX


    if return_labels:
        return probabilities, labels
    else:
        return probabilities