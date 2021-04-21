tens = transforms.ToTensor()
pil = transforms.ToPILImage()
images = [train_data.loader(train_data.imgs[i][0]) for i in range(5015, 5900)]
images = torch.stack(list(map(tens, images)))
mi = images.mean(dim=0)
pil(mi)