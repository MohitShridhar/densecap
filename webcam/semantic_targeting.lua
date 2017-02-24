local ros = require 'ros'
require 'ros.actionlib.ActionServer'
local actionlib = ros.actionlib

require 'image'
gm = require 'graphicsmagick'

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
require 'image'
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
cmd:option('-model_image_size', 150)
cmd:option('-num_proposals', 100)
cmd:option('-boxes_to_show', 10)
cmd:option('-webcam_fps', 1)
cmd:option('-gpu', 0)
cmd:option('-timing', 1)
cmd:option('-detailed_timing', 0)
cmd:option('-text_size', 2)
cmd:option('-box_width', 2)
cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.3)
cmd:option('-use_cudnn', 1)

ros.init('densecap_actionserver')
nh = ros.NodeHandle()

spinner = ros.AsyncSpinner()
spinner:start()

local function grab_frame(opt, img_orig)
  local timer = nil
  if opt.timing == 1 then
    cutorch.synchronize()
    timer = torch.Timer()
  end

  -- local img_orig = img
  local img = image.scale(img_orig, opt.model_image_size)
  local img_caffe = img:index(1, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.Tensor{103.939, 116.779, 123.68}
  img_caffe:add(-1, vgg_mean:view(3, 1, 1):expandAs(img_caffe))
  local H, W = img_caffe:size(2), img_caffe:size(3)
  img_caffe = img_caffe:view(1, 3, H, W)

  if opt.timing == 1 then
    cutorch.synchronize()
    print('grab_frame took ', timer:time().real)
  end

  return img_orig, img_caffe
end


local function run_model(opt, info, model, img_caffe, frame_id, store_features)
  local timer = nil
  if opt.timing == 1 then
    cutorch.synchronize()
    timer = torch.Timer()
  end

  local model_timer = nil
  if opt.timing == 1 then
    model_timer = torch.Timer()
    cutorch.synchronize()
    model.timing = opt.detailed_timing
  end

  local boxes_xcycwh, scores, captions, feats = model:forward_test(img_caffe:cuda())

  if store_features then
    history_feats[frame_id] = feats:type('torch.LongTensor')
    history_captions[frame_id] = captions
    history_boxes_xcycwh[frame_id] = boxes_xcycwh:type('torch.LongTensor')
  end

  if opt.timing == 1 then
    cutorch.synchronize()
    print(string.format('    model_forward took %f', model_timer:time().real))
    if opt.detailed_timing == 1 then
      for k, v in pairs(model.stats.times) do
        print(string.format('    %s took %f', k, v))
      end
    end
  end
  local num_boxes = math.min(boxes_xcycwh:size(1), opt.boxes_to_show)
  boxes_xcycwh = boxes_xcycwh[{{1, num_boxes}}]

  if opt.timing == 1 then
    cutorch.synchronize()
    print('run_model took ', timer:time().real)
  end

  return boxes_xcycwh:float(), captions, scores
end


local function image_query(opt, info, model, query, min_loss_threshold)
  local timer = nil
  if opt.timing == 1 then
    cutorch.synchronize()
    timer = torch.Timer()
  end

  local model_timer = nil
  if opt.timing == 1 then
    model_timer = torch.Timer()
    cutorch.synchronize()
    model.timing = opt.detailed_timing
  end

  local top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time = model:language_query(history_feats, history_captions, history_boxes_xcycwh, history_boxes_xywh, query, min_loss_threshold)

  if opt.timing == 1 then
    cutorch.synchronize()
    print(string.format('    model_forward took %f', model_timer:time().real))
    if opt.detailed_timing == 1 then
      for k, v in pairs(model.stats.times) do
        print(string.format('    %s took %f', k, v))
      end
    end
  end
  -- local num_boxes = math.min(boxes_xcycwh:size(1), opt.boxes_to_show)
  -- boxes_xcycwh = boxes_xcycwh[{{1, num_boxes}}]

  if opt.timing == 1 then
    cutorch.synchronize()
    print('run_model took ', timer:time().real)
  end

  return top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time

end


local function show_results(opt, img_orig, img_caffe, win, boxes_xywh, captions)
  local timer = nil
  if opt.timing == 1 then
    cutorch.synchronize()
    timer = torch.Timer()
  end

  -- Draw boxes in the image
  local draw_options = {text_size=opt.text_size, box_width=opt.box_width}
  local img_disp = vis_utils.densecap_draw(img_orig, boxes_xywh, captions, draw_options)
  img_disp:clamp(0, 1)

  -- Show the image
  if not win then
    -- On the first call just use image.display
    win = image.display{image=img_disp, win=win}
  else
    -- Re-calling image.display for the same window invokes the garbage
    -- collector, which kills the framerate. Therefore we dip into qt
    -- and update it ourselves.
    win.image = img_disp
    local size = win.window.size:totable()
    local qtimg = qt.QImage.fromTensor(img_disp)
    win.painter:image(0, 0, size.width, size.height, qtimg)
  end

  if opt.timing == 1 then
    cutorch.synchronize()
    print('show_results took ', timer:time().real)
  end

  return win
end


local function temporal_smoothing(prev_boxes, prev_captions, cur_boxes, cur_captions)
  -- Reorder the current boxes to match the order of boxes from the
  -- previous frame; this prevents colors of stable detections from bouncing
  -- around too much.
  -- TODO: Do something fancier?

  local ious = nn.BoxIoU():float():forward{
                  prev_boxes:view(1, -1, 4),
                  cur_boxes:view(1, -1, 4)}[1]
  local num_cur = cur_boxes:size(1)
  local num_prev = prev_boxes:size(1)
  local idx = torch.LongTensor(num_cur)
  for i = 1, math.min(num_prev, num_cur) do
    local _, j = ious[i]:max(1)
    j = j[1]
    idx[i] = j
    ious[{{}, j}] = -1
  end
  if num_cur > num_prev then
    for i = num_prev + 1, num_cur do
      local _, j = ious:max(1):max(2)
      j = j[1]
      idx[i] = j
      ious[{{}, j}] = -1
    end
  end

  local new_boxes = cur_boxes:index(1, idx)
  local new_captions = {}
  for i = 1, num_cur do
    new_captions[i] = cur_captions[idx[i]]
  end

  return new_boxes, new_captions
end


local function process(img, frame_id, store_features)
  if opt.timing == 1 then
    cutorch.synchronize()
    timer:reset()
  end

  local boxes_xywh, captions, scores
  local boxes_xcycwh
  if not paused then
    local img_orig, img_caffe = grab_frame(opt, img)
    boxes_xcycwh, captions, scores = run_model(opt, info, model, img_caffe, frame_id, store_features)

    -- if prev_boxes then
    --   boxes_xcycwh, captions = temporal_smoothing(prev_boxes, prev_captions,
    --                                               boxes_xcycwh, captions)
    -- end

    boxes_xywh = box_utils.xcycwh_to_xywh(boxes_xcycwh)
    local scale = img_orig:size(2) / img_caffe:size(3)
    boxes_xywh = box_utils.scale_boxes_xywh(boxes_xywh, scale)

    if (store_features) then
      history_boxes_xywh[frame_id] = boxes_xywh:type('torch.LongTensor')
    end

    win = show_results(opt, img_orig, img_caffe, win, boxes_xywh, captions)

    prev_boxes = boxes_xcycwh
    prev_captions = captions
  end

  if opt.timing == 1 then
    cutorch.synchronize()
    local time = timer:time().real
    local fps = 1.0 / time
    local msg = 'Iteration took %.3f (%.2f FPS)'
    print(string.format(msg, time, fps))
    print ''
  end

  return boxes_xywh, captions, scores
end


local function search(query, min_loss_threshold)
  if opt.timing == 1 then
    cutorch.synchronize()
    timer:reset()
  end

  local boxes_xcycwh, captions, scores
  if not paused then
    top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time = image_query(opt, info, model, query, min_loss_threshold)
  end

  if opt.timing == 1 then
    cutorch.synchronize()
    local time = timer:time().real
    local fps = 1.0 / time
    local msg = 'Iteration took %.3f (%.2f FPS)'
    print(string.format(msg, time, fps))
    print ''
  end

  return top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time
end


local function Caption_ActionServer_Goal(goal_handle)
  ros.INFO("Caption_ActionServer_Goal")
  local g = goal_handle:getGoal().goal

  print (g)

  -- Convert to torch image tensor
  local img_tensor = torch.reshape(g.input.data, torch.LongStorage{g.input.height, g.input.width, 3})
  -- local img_gm = gm.Image(img_tensor, 'RGB', 'DWH')
  local img_gm = gm.Image(img_tensor, 'BGR', 'DWH')

  local img = img_gm:toTensor('double','RGB', 'DHW')

  -- Loading specified settings (TODO: not reinitialized)
  opt.model_image_size = g.model_image_size
  opt.num_proposals = g.num_proposals
  opt.boxes_to_show = g.boxes_to_show
  opt.rpn_nms_thresh = g.rpn_nms_thresh
  opt.final_nms_thresh = g.final_nms_thresh

  goal_handle:setAccepted('yip')
  
  local boxes_xywh, captions, scores = process(img, g.frame_id, g.store_features)

  local r = goal_handle:createResult()
  r.n = scores:size(1)
  r.captions = captions
  r.boxes = boxes_xywh:reshape(boxes_xywh:size(1) * boxes_xywh:size(2)):float()
  r.scores = scores:reshape(scores:size(1)):float()

  goal_handle:setSucceeded(r, 'done')
end


local function Query_ActionServer_Goal(goal_handle)
  ros.INFO("Query_ActionServer_Goal")
  local g = goal_handle:getGoal().goal

  print (g)

  -- IMPORTANT: check if history is available, otherwise reject the goal
  goal_handle:setAccepted('yip')

  local top_k_ids, top_k_boxes, top_k_losses, top_k_meteor_ranks, search_time = search(g.query, g.min_loss_threshold)

  local r = goal_handle:createResult()
  r.frame_ids = top_k_ids:reshape(top_k_ids:size(1)):int()
  r.captioning_losses = top_k_losses:reshape(top_k_losses:size(1)):float()
  r.boxes = top_k_boxes:reshape(top_k_boxes:size(1) * top_k_boxes:size(2)):float()
  r.meteor_ranks = top_k_meteor_ranks:reshape(top_k_meteor_ranks:size(1)):int()
  r.search_time = search_time

  goal_handle:setSucceeded(r, 'done')
end

-- Action server initialization
local as_caption = actionlib.ActionServer(nh, 'dense_caption', 'action_controller/DenseCaption')
local as_query = actionlib.ActionServer(nh, 'dense_query', 'action_controller/DenseImageQuery')

as_caption:registerGoalCallback(Caption_ActionServer_Goal)
as_query:registerGoalCallback(Query_ActionServer_Goal)

print('Starting Dense Caption & Query action server...')
as_caption:start()
as_query:start()

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

win = nil
camera_opt = {
  fps=opt.webcam_fps,
  height=opt.display_image_height,
  width=opt.display_image_width,
}

-- Some variables for managing state
attached_handler = false
paused = false
prev_boxes, prev_captions = nil, nil

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

as_caption:shutdown()
as_query:shutdown()
nh:shutdown()
server:shutdown()
ros.shutdown()