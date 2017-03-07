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
cmd:option('-model_image_size', 720)
cmd:option('-num_proposals', 1000)
cmd:option('-boxes_to_show', 40)
cmd:option('-webcam_fps', 1)
cmd:option('-gpu', 0)
cmd:option('-timing', 1)
cmd:option('-detailed_timing', 0)
cmd:option('-text_size', 2)
cmd:option('-box_width', 2)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.05)
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

  print (g)

  -- Convert to torch image tensor
  local img_tensor = torch.reshape(g.input.data, torch.LongStorage{g.input.height, g.input.width, 3})
  local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  -- local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  local img = img_gm:toTensor('double','RGB', 'DHW')

  goal_handle:setAccepted('yip')
  
  -- compute boxes
  local img_orig, img_caffe = grab_frame(opt, img)
  local boxes_xcycwh, scores, captions, feats = model:forward_test(img_caffe:float())

  -- compute fc7 features for whole image
  local whole_img_roi = torch.FloatTensor{{1.0, 1.0, g.input.width*1.0, g.input.height*1.0}}
  local out = model:forward_boxes(img_caffe:float(), whole_img_roi)
  local f_objectness_scores, f_seqs, f_roi_codes, f_hidden_codes, f_captions = unpack(out)

  -- scale boxes to image size
  boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
  local scale = img_orig:size(2) / img_caffe:size(3)
  boxes_xywh = box_utils.scale_boxes_xywh(boxes_xywh, scale)

  -- store results for future queries
  history_feats[g.frame_id] = feats:type('torch.FloatTensor')
  history_captions[g.frame_id] = captions
  history_boxes_xcycwh[g.frame_id] = boxes_xcycwh:type('torch.FloatTensor')
  history_boxes_xywh[g.frame_id] = boxes_xywh:type('torch.FloatTensor')

  -- return results
  local r = goal_handle:createResult()
  r.fc7_img = f_roi_codes:reshape(f_roi_codes:size(2)):float()
  r.fc7_vecs = feats:reshape(feats:size(1) * feats:size(2)):float()
  r.boxes = boxes_xywh:reshape(boxes_xywh:size(1) * boxes_xywh:size(2)):float()
  r.scores = scores:reshape(scores:size(1)):float()
  r.captions = captions

  goal_handle:setSucceeded(r, 'done')
end


local function Query_Action_Goal(goal_handle)
  ros.INFO("Query_Action_Goal")
  local g = goal_handle:getGoal().goal

  print (g)

  -- TODO IMPORTANT: check if history is available, otherwise reject the goal
  goal_handle:setAccepted('yip')

  -- local top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time = search(g.query, g.min_loss_threshold)
  local top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time, top_k_feats, top_k_orig_idx = model:language_query(history_feats, history_captions, history_boxes_xcycwh, history_boxes_xywh, g.query, g.min_loss_threshold, g.k)

  local r = goal_handle:createResult()
  r.frame_ids = top_k_ids:reshape(top_k_ids:size(1)):int()
  r.captioning_losses = top_k_losses:reshape(top_k_losses:size(1)):float()
  r.boxes = top_k_boxes:reshape(top_k_boxes:size(1) * top_k_boxes:size(2)):float()
  r.meteor_ranks = top_k_meteor_ranks:reshape(top_k_meteor_ranks:size(1)):int()
  r.search_time = search_time
  r.fc7_vecs = top_k_feats:reshape(top_k_feats:size(1) * top_k_feats:size(2)):float()
  r.orig_idx = top_k_orig_idx

  goal_handle:setSucceeded(r, 'done')
end


opt = cmd:parse(arg)

-- Setup Localization Server
local as_localize_server = actionlib.ActionServer(nh, 'dense_localize', 'action_controller/Localize')
local as_query_server = actionlib.ActionServer(nh, 'localize_query', 'action_controller/LocalizeQuery')

as_localize_server:registerGoalCallback(Localize_Action_Server)
as_query_server:registerGoalCallback(Query_Action_Goal)

print('Starting Dense Localization and Query action server...')
as_localize_server:start()
as_query_server:start()

opt = cmd:parse(arg)
-- dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
dtype, use_cudnn = utils.setup_gpus(-1, 0)


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

-- NOTE: linear space complexity
history_feats = {}
history_captions = {}
history_boxes_xcycwh = {}
history_boxes_xywh = {}

timer = torch.Timer()

local s = ros.Duration(0.001)
while ros.ok() do
  s:sleep()
  ros.spinOnce()
end

as_localize_server:shutdown()
as_query_server:shutdown()
nh:shutdown()
server:shutdown()
ros.shutdown()