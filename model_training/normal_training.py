import time
import torch
import numpy
import random
from from_common import *
from tqdm import tqdm
import gc


class NormalTraining():
    """
    Code adopted from confidence-calibrated-adversarial-training
    Normal training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, writer, cuda=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert isinstance(model, torch.nn.Module)
        assert len(trainset) > 0
        assert len(testset) > 0
        assert isinstance(trainset, torch.utils.data.DataLoader)
        assert isinstance(testset, torch.utils.data.DataLoader)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert (cuda and is_cuda(model)) or (not cuda and not is_cuda(model))

        self.writer = writer
        """ (torch.util.tensorboardSummarWriter or equivalent) Summary writer. """

        self.model = model
        """ (torch.nn.Module) Model. """

        self.trainset = trainset
        """ (torch.utils.data.DatLoader) Taining set. """

        self.testset = testset
        """ (torch.utils.data.DatLoader) Test set. """

        self.optimizer = optimizer
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = scheduler
        """ (torch.optim.LRScheduler) Scheduler. """

        self.cuda = cuda
        """ (bool) Run on CUDA. """

        self.loss = classification_loss
        """ (callable) Classificaiton loss. """

        self.summary_gradients = False
        """ (bool) Summary for gradients. """

        self.writer.add_text('config/model', self.model.__class__.__name__)
        self.writer.add_text('config/model_details', str(self.model))
        self.writer.add_text('config/optimizer', self.optimizer.__class__.__name__)
        self.writer.add_text('config/scheduler', self.scheduler.__class__.__name__)
        self.writer.add_text('config/cuda', str(self.cuda))

        seed = int(time.time()) # time() is float
        self.writer.add_text('config/seed', str(seed))
        torch.manual_seed(seed)
        numpy.random.seed(seed)
        random.seed(seed)


    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.train()
        assert self.model.training is True
        batches = len(self.trainset)

        losses = None
        errors = None
        logits = None
        confidences = None
        accs = []

        b = 0
        for (inputs, targets) in tqdm(self.trainset):
            gc.collect()

            inputs = as_torch_variable(inputs, self.cuda)
            assert len(targets.shape) == 1
            targets = as_torch_variable(targets, self.cuda)
            targets = targets.long()
            assert len(list(targets.size())) == 1

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, targets)
            
            errors = np_concatenate(errors, classification_error(outputs, targets, reduction='none').detach().cpu().numpy())
            losses = np_concatenate(losses, self.loss(outputs, targets, reduction='none').detach().cpu().numpy())
            logits = np_concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            confidences = np_concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
            accs.append((torch.sum((torch.argmax(outputs, axis=1) == targets)) / float(len(targets))).detach().cpu().numpy())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            
            if b == batches - 1:
                global_step = epoch
                self.writer.add_scalar('train/loss', numpy.mean(losses), global_step=global_step)
                self.writer.add_scalar('train/error', numpy.mean(errors), global_step=global_step)
                self.writer.add_scalar('train/confidence', numpy.mean(confidences), global_step=global_step)
                self.writer.add_scalar('train/accuracy', numpy.mean(accs), global_step=global_step)
                # print(loss.item(), error.item())
 
                # self.writer.add_images('train/images', inputs[:8], global_step=global_step)
            b += 1

        

    def test(self, epoch):
        """
        Test step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.eval()
        assert self.model.training is False

        # reason to repeat this here: use correct loss for statistics
        losses = None
        errors = None
        logits = None
        confidences = None

        for (inputs, targets) in tqdm(self.testset):
            gc.collect()
            inputs = as_torch_variable(inputs, self.cuda)
            # inputs = inputs.permute(0, 3, 1, 2)
            targets = as_torch_variable(targets, self.cuda)
            targets = targets.long()

            outputs = self.model(inputs)
            losses = np_concatenate(losses, self.loss(outputs, targets, reduction='none').detach().cpu().numpy())
            errors = np_concatenate(errors, classification_error(outputs, targets, reduction='none').detach().cpu().numpy())
            logits = np_concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            confidences = np_concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())

        global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1
        self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)

    def step(self, epoch):
        """
        Training + test step.

        :param epoch: epoch
        :type epoch: int
        """

        self.train(epoch)
        self.test(epoch)