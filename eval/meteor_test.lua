require 'torch'

local eval_utils = require 'eval.eval_utils'
local inspect = require('inspect')

records = {}

-- record 1
local record = {}
record.candidate = 'a man riding a bicycle'
record.references = {'trees in the background', 'trees in the background'}
table.insert(records, record)

-- record 2
local record2 = {}
record2.candidate = 'the box next to the red clock'
record2.references = {'the quick brown fox', 'the quick brown fox'}
table.insert(records, record2)

local blob = eval_utils.score_captions(records, 0)
print (inspect(blob))