local ros = require 'ros'
require 'ros.actionlib.ActionServer'
local actionlib = ros.actionlib

local image = require 'image'
gm = require 'graphicsmagick'

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
require 'camera'
require 'qt'
require 'qttorch'
require 'qtwidget'

require 'densecap.DenseCapModel'
require 'densecap.modules.BoxIoU'

local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'

cmd = torch.CmdLine()
cmd:option('-checkpoint',
  'data/models/densecap/densecap-pretrained-vgg16.t7')
cmd:option('-display_image_height', 640)
cmd:option('-display_image_width', 480)
cmd:option('-model_image_size', 240)
cmd:option('-num_proposals', 100)
cmd:option('-boxes_to_show', 100)
cmd:option('-webcam_fps', 1)
cmd:option('-gpu', 0)
cmd:option('-timing', 1)
cmd:option('-detailed_timing', 0)
cmd:option('-text_size', 2)
cmd:option('-box_width', 2)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-use_cudnn', 1)

ros.init('localize_actionserver')
nh = ros.NodeHandle()

spinner = ros.AsyncSpinner()
spinner:start()

local function grab_frame(opt, img_orig)

  -- local img_orig = img
  local img = image.scale(img_orig, opt.model_image_size)
  local img_caffe = img:index(1, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.Tensor{103.939, 116.779, 123.68}
  img_caffe:add(-1, vgg_mean:view(3, 1, 1):expandAs(img_caffe))
  local H, W = img_caffe:size(2), img_caffe:size(3)
  img_caffe = img_caffe:view(1, 3, H, W)

  return img_orig, img_caffe
end
  
local function Localize_Action_Server(goal_handle)
  ros.INFO("Localize_Action_Server")
  local g = goal_handle:getGoal().goal

  -- Convert to torch image tensor
  local img_tensor = torch.reshape(g.input.data, torch.LongStorage{g.input.height, g.input.width, 3})
  local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  -- local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  local img = img_gm:toTensor('double','RGB', 'DHW')

  goal_handle:setAccepted('yip')
  
  local img_orig, img_caffe = grab_frame(opt, img)
  local boxes_xcycwh, scores, captions, feats = model:forward_test(img_caffe:cuda())

  boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
  local scale = img_orig:size(2) / img_caffe:size(3)
  boxes_xywh = box_utils.scale_boxes_xywh(boxes_xywh, scale)

  local r = goal_handle:createResult()
  r.fc7_vecs = feats:reshape(feats:size(1) * feats:size(2)):float()
  r.boxes = boxes_xywh:reshape(boxes_xywh:size(1) * boxes_xywh:size(2)):float()
  r.scores = scores:reshape(scores:size(1)):float()
  r.captions = captions

  goal_handle:setSucceeded(r, 'done')
end

opt = cmd:parse(arg)

-- Setup Localization Server
local as_localize_server = actionlib.ActionServer(nh, 'dense_localize', 'action_controller/Localize')
as_localize_server:registerGoalCallback(Localize_Action_Server)


print('Starting Dense Localization action server...')
as_localize_server:start()

opt = cmd:parse(arg)
dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)

-- Load the checkpoint
print('loading checkpoint from ' .. opt.checkpoint)
checkpoint = torch.load(opt.checkpoint)
model = checkpoint.model
print('done loading checkpoint')

-- Ship checkpoint to GPU and convert to cuDNN
model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh = opt.rpn_nms_thresh,
  final_nms_thresh = opt.final_nms_thresh,
  num_proposals = opt.num_proposals,
}
model:evaluate()



timer = torch.Timer()

local s = ros.Duration(0.001)
while ros.ok() do
  s:sleep()
  ros.spinOnce()
end

as_localize_server:shutdown()
nh:shutdown()
server:shutdown()
ros.shutdown()