package.path = './?.lua;' .. package.path
require 'torch'
require 'nn'
require 'image'
require 'inspect'

require 'densecap.DenseCapModel'
utils = require 'densecap.utils'
box_utils = require 'densecap.box_utils'
vis_utils = require 'densecap.vis_utils'
inspect = require('inspect')

-- options
checkpoint = './data/models/densecap/densecap-pretrained-vgg16.t7'
-- load model
dtype, use_cudnn = utils.setup_gpus(0, 1)
-- dtype, use_cudnn = utils.setup_gpus(-1, 0)
print(dtype, use_cudnn)
model = torch.load(checkpoint).model
model:convert(dtype, use_cudnn)
model:evaluate()

image_size = 250
img_path = './imgs/elephant.jpg'

-- read image and box
img = image.load(img_path, 3)
box1 = torch.Tensor{400, 260, 400, 300}
box2 = torch.Tensor{400, 250, 400, 450}
boxes = torch.cat({box1, box2},1):view(2, -1)

-- rescale box and img
boxes = boxes:mul(image_size/math.max(img:size(2), img:size(3)))


img = image.scale(img, image_size):float()
H, W = img:size(2), img:size(3)

-- convert img
img_caffe = img:view(1, 3, H, W)
img_caffe = img_caffe:index(2, torch.LongTensor{3,2,1}):mul(255)
vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
img_caffe:add(-1, vgg_mean)

-- get ONE caption for the target boxes
out = model:forward_boxes(img_caffe:type(dtype), boxes:type(dtype))
obj_scores, seqs, roi_codes, hidden_codes, captions = unpack(out)
print(inspect(captions))

print (hidden_codes:size())

-- -- get BEAM captions for the target boxes
beam_size = 10
output = model:forward_boxes_beams(img_caffe:type(dtype), boxes:type(dtype), beam_size)
beam_captions = output[5]
print(inspect(beam_captions))