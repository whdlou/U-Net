img_type = '.jpg'
mask_type = '.png'
# palette = ((0, 0, 0),
#            (0, 0, 255),
#            (0, 255, 0),
#            (0, 255, 255),
#            (255, 0, 0),
#            (255, 0, 255),
#            (255, 255, 0),
#            (255, 255, 255))
palette = ((0, 0, 0), (255, 0, 0), (0, 0, 255), (255, 0, 255))
num_classes = 4
size = (256, 256)
batch_size = 16
start_epoch = 0
num_epochs = 250
lr = 1e-3
weight = 'output/weights/epoch171_loss0.012811.pth'