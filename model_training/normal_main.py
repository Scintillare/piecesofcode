# python examples/pretnormal_training_robustness.py --dataset mushrooms --tensorboard

import os
import math
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from normal_training import *

from train_utils import *


DATA_DIR_TRAIN = r".\dataset\STLmod\train"
DATA_DIR_TEST = r".\dataset\STLmod\test"

class Main:
    def __init__(self, args=None):
        """
        Initialize.

        :param args: optional arguments if not to use sys.argv
        :type args: [str]
        """

        self.directory = './.assets/checkpoints'
        """ Arguments of program. """

        self.trainloader = None
        """ (torch.utils.data.DataLoader) Training loader. """

        self.testloader = None
        """ (torch.utils.data.DataLoader) Test loader. """


    def setup(self):
        """
        Set dataloaders.
        """

        batch_size = 12
        image_transforms = {
            'train':
            transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        #     # Validation does not use augmentation
            'val':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }
        data = {
            'train':
            datasets.ImageFolder(DATA_DIR_TRAIN, image_transforms['train']),
            'test':
            datasets.ImageFolder(DATA_DIR_TEST, image_transforms['val']),
        }
        self.trainloader = DataLoader(data['train'],  batch_size=batch_size, shuffle=True, drop_last=True)
        self.testloader = DataLoader(data['test'],  batch_size=batch_size, shuffle=False, drop_last=True)

        os.makedirs(self.directory, exist_ok=True)

    def train(self):
        """
        Training configuration.
        """

        writer = SummaryWriter('%s/logs/' % self.directory, max_queue=100)

        epochs = 20
        snapshot = 1
        start_epoch = 0
        num_classes = 10

        self.model = get_resnet(num_classes) 
        self.model = self.model.cuda()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=10**(-4), betas=(0, 0.99))
        scheduler = None

        model_file = '%s/classifier.pth.tar' % self.directory
        optim_file = '%s/optimizer.pth.tar' % self.directory
        incomplete_model_file = find_last_checkpoint(model_file)
        incomplete_optim_file = find_last_checkpoint(optim_file)

        if incomplete_model_file is not None:
            self.model.load_state_dict(torch.load(incomplete_model_file))
            optimizer.load_state_dict(torch.load(incomplete_optim_file))
            *_, start_epoch = incomplete_optim_file.split('.')
            start_epoch = int(start_epoch)+1
            print('loaded %s' % incomplete_model_file)
            print('loaded %s' % incomplete_optim_file)

        trainer = NormalTraining(self.model, self.trainloader, self.testloader, optimizer, scheduler,
                                             writer=writer, cuda=True)

        self.model.train()
        for epoch in tqdm(range(start_epoch, epochs)):
            trainer.step(epoch)
            writer.flush()

            snapshot_model_file = '%s/classifier.pth.tar.%d' % (self.directory, epoch)
            snapshot_optim_file = '%s/optimizer.pth.tar.%d' % (self.directory, epoch)
            torch.save(self.model.state_dict(), snapshot_model_file)
            torch.save(optimizer.state_dict(), snapshot_optim_file)

            previous_model_file = '%s/classifier.pth.tar.%d' % (self.directory, epoch - 1)
            previous_optim_file = '%s/optimizer.pth.tar.%d' % (self.directory, epoch - 1)
            if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
                os.unlink(previous_model_file)
                os.unlink(previous_optim_file)

        previous_model_file = '%s/classifier.pth.tar.%d' % (self.directory, epoch - 1)
        previous_optim_file = '%s/optimizer.pth.tar.%d' % (self.directory, epoch - 1)
        if os.path.exists(previous_model_file) and (epoch - 1) % snapshot > 0:
            os.unlink(previous_model_file)
            os.unlink(previous_optim_file)
        
        torch.save(self.model.state_dict(), model_file)
        torch.save(optimizer.state_dict(), optim_file)


    def evaluate(self):
        """
        Evaluate.
        """
        self.model.eval()
        clean_probabilities, labels = test(self.model, self.testloader, cuda=True, return_labels=True)
        acc = numpy.sum((numpy.argmax(clean_probabilities, axis=1) == labels)) / float(len(labels))
        print(f'Accuracy: {acc}')


    def main(self):
        """
        Main.
        """

        self.setup()

        model_file = '%s/classifier.pth.tar' % self.directory        
        num_classes = 10        
        if not os.path.exists(model_file):
            self.train()
        else:
            self.model = get_resnet(num_classes)    
            self.model.load_state_dict(torch.load(model_file))
            self.model = self.model.cuda()
        self.evaluate()


if __name__ == '__main__':
    program = Main()
    program.main()