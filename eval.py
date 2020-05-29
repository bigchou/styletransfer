import torch
import torch.nn as nn
import torch.optim as optim

import copy, os, cv2, copy, argparse
import numpy as np
from torchvision import models, transforms
from lib import *




parser = argparse.ArgumentParser()
parser.add_argument("--content-image", default="images/content/market.jpeg",type=str, help="path to content image you want to stylize")
parser.add_argument("--style-image",   default="images/style/sketch_market.jpg",type=str,help="path to style-image")
parser.add_argument("--model", default="models/vgg19-d01eb7cb.pth")
parser.add_argument("--MAX_IMAGE_SIZE", type=int, default=512)
parser.add_argument('--OPTIMIZER', type=str, default='adam',choices=["adam", "lbfgs"])
parser.add_argument('--NUM_ITER',type=int,default=200)
parser.add_argument("--outdir",type=str,default="result")
args = parser.parse_args()

# GPU
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Load File
style_img = cv2.imread(args.style_image)# Images loaded as BGR
style_tensor = itot(style_img,MAX_IMAGE_SIZE=args.MAX_IMAGE_SIZE).to(device)# Convert Images to Tensor
content_img = cv2.imread(args.content_image)# Images loaded as BGR
content_tensor = itot(content_img,MAX_IMAGE_SIZE=args.MAX_IMAGE_SIZE).to(device)# Convert Images to Tensor
g = initial(content_tensor, init_image='content')
g = g.to(device).requires_grad_(True)
print("content file: ",args.content_image, "style_file:", args.style_image)

# MODEL CONFIG
vgg = models.vgg19(pretrained=False)
VGG19_PATH = args.model
vgg.load_state_dict(torch.load(VGG19_PATH), strict=False)
model = copy.deepcopy(vgg.features)
model.to(device)
for param in model.parameters():
	param.requires_grad = False# Turn-off unnecessary gradient tracking

# Stylize
prefix = args.outdir#"result"
if not os.path.exists(prefix): os.mkdir(prefix)
contentweight = range(1,5)
styleweight = range(1,5)
for cw in contentweight:
	for sw in styleweight:
		cw = 15e0
		sw = 1e2#sw *= 10
		out = stylize(model,g,content_tensor,style_tensor,iteration=args.NUM_ITER,OPTIMIZER=args.OPTIMIZER,STYLE_WEIGHT=sw,CONTENT_WEIGHT=cw)
		# Save the final output
		img = ttoi(g.clone().detach())# Convert Tensor to Image
		name = "cw"+str(cw)+"_sw"+str(sw)+".png"
		name = os.path.join(prefix,name)
		print("save to %s"%(name))
		saveimg(img, name)
		exit()