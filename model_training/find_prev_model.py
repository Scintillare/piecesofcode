
import os
import numpy

# From repository confidence-calibrated adversarial training
def find_incomplete_state_file(model_file):
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
                i = numpy.argmax(epochs)
                return os.path.join(base_directory, file_name + '.%d' % epochs[i])

def main():
    directory = r''
    model_file = '%s/classifier.pth.tar' % directory
    incomplete_model_file = find_incomplete_state_file(model_file)
    load_file = model_file
    if incomplete_model_file is not None:
        load_file = incomplete_model_file
    
    start_epoch = 0
    if os.path.exists(load_file):
        #load model in some way
        # checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
        # model = some_model_class()
        # model.load_state_dict(checkpoint['model'])
        # model = state.model
        # del checkpoint
        # torch.cuda.empty_cache()
        # start_epoch = state.epoch + 1
        # epoch = start_epoch

        # *_, self.start_epoch = last_netG.split('.')
        # self.iteration = None
        # self.start_epoch = int(self.start_epoch)+1
        
        print('loaded %s' % load_file)
    else:
        # model = some_model_class()
        print("Start from scratch")
    # model = model.cuda()


    # save generator
        if epoch%1==0:
            netG_file_name = self.netG_file_name + '.' + str(epoch) 
            torch.save(self.netG.state_dict(), netG_file_name)
            netD_file_name = self.netDisc_file_name + '.' + str(epoch) 
            torch.save(self.netDisc.state_dict(), netD_file_name)
            optG_file_name = self.optG_file_name + '.' + str(epoch) 
            torch.save(self.optimizer_G.state_dict(), optG_file_name)
            optD_file_name = self.optD_file_name + '.' + str(epoch) 
            torch.save(self.optimizer_D.state_dict(), optD_file_name)
        
    #save final model
    torch.save(self.netG.state_dict(), self.netG_file_name )
    torch.save(self.netDisc.state_dict(), self.netDisc_file_name)
    torch.save(self.optimizer_G.state_dict(), self.optG_file_name)
    torch.save(self.optimizer_D.state_dict(), self.optD_file_name)