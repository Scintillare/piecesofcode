from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./assets/logs/', max_queue=100)

writer.add_scalar('train/loss', loss.item(), global_step=epoch)
writer.add_scalar('train/accuracy', 100. * correct / len(train_loader.dataset), global_step=epoch)

writer.add_scalar('test/loss', test_loss, global_step=epoch)
writer.add_scalar('test/accuracy', 100. * correct / len(test_loader.dataset), global_step=epoch)

self.writer.add_images('train/images', inputs[:8], global_step=global_step)
    
writer.flush()