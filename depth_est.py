import numpy as np
from torchvision import transforms
import torch
from network import depth8sig
import matplotlib.pyplot as plt

def est_depth(x, pilImage_path):

	# reshape image
	x = np.array(x)
	x = transforms.ToTensor()(x)
	x = torch.unsqueeze(x, 0)
	print('---- image reshaped')

	# init model
	model = depth8sig(8)
	model.load_model('Depth8_cpu.pt')

	print('---- model loaded')

	# get prediction
	prediction = model.forward(x)
	prediction = torch.squeeze(prediction, 0)
	prediction = np.array(prediction.detach().cpu())
	prediction = prediction.reshape((480, 640))

	print('---- prediction gotten')

	# save prediction (for send_file)
	plt.imshow(prediction, cmap='gray')
	plt.axis('off')
	plt.savefig(pilImage_path)
	print('---- prediction saved')

	return True