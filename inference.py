from torch.utils.data import DataLoader
import torch
from PIL import Image
import cfg
import numpy as np

def inference(model,
              test_set,
              device):
    model.eval()
    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False)
    for batch_id, data in enumerate(test_loader):
        imgs = data['image'].to(device)
        out = model(imgs)
        preds = torch.argmax(out, dim=1).cpu()
        palette = torch.tensor(cfg.palette)
        result = palette[preds].squeeze().numpy().astype(np.uint8)
        Image.fromarray(result).save('output\\results\\{}.png'.format(data['name'][0]))
        print('Save succesfully.')