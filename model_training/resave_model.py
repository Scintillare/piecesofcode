import os

directory = r''
save_every = 3
for epoch in tqdm(range(start_epoch, epochs)):
    # trainer.step(epoch)
    # writer.flush()

    snapshot_model_file = '%s/classifier.pth.tar.%d' % (directory, epoch)
    #dsve model in some way
    # torch.save({
    #         'model': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #         'scheduler': scheduler.state_dict(),
    #         'epoch': epoch,
    #     }, filepath)

    previous_model_file = '%s/classifier.pth.tar.%d' % (directory, epoch - 1)
    if os.path.exists(previous_model_file) and (epoch - 1) % save_every > 0:
        os.unlink(previous_model_file)

previous_model_file = '%s/classifier.pth.tar.%d' % (directory, epoch - 1)
if os.path.exists(previous_model_file) and (epoch - 1) % save_every > 0:
    os.unlink(previous_model_file)
#save in come way