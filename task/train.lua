local cmdio = require 'cmdio'
local sig   = require 'signal'

local argcheck = require 'argcheck'
local image    = require 'image'
local nn       = require 'nn'
local optim    = require 'optim'
local tnt      = require 'torchnet'
local nngraph  = require 'nngraph'


require 'dropconnect'
require 'early_stopper'
require 'exactmatchmeter'

local function parse_args(args)
   local op = xlua.OptionParser("train.lua --train TRAIN_DATASET"
      .. " --test TEST_DATASET --shared SHARED_NN --secific SPECIFIC_NN"
      .. " [OPTIONS...] PREVIOUSLY_TRAINED_SPECIFIC_NNs...\n\n"
      .. [[
Further description to fill in...]])

   op:option{
      "--train",
      dest = "train_path",
      help = "a training datafile",
   }

   op:option{
      "--test",
      dest = "test_path",
      help = "a testing datafile",
      req  = true,
   }

   op:option{
      "--shared",
      dest = "shared_path",
      help = "a previously created shared part of the nn",
      req = true,
   }

   op:option{
      "--specific",
      dest = "specific_path",
      help = "a previously created specific part of the nn",
      req = true,
   }

   op:option{
      "--split",
      default = 0.8,
      dest    = "split",
      help    = "keep this percentage for training and the rest for validation"
   }

   op:option{
      "--try-epochs",
      default = 5,
      dest    = "try_epochs",
      help    = "keep trying for # epochs to find better parameters (early stopping)",
   }

   op:option{
      "--N",
      default = 2,
      dest    = "n",
      help    = "N parameter for softmax lowering",
   }

   op:option{
      "--optim",
      default = "adam",
      dest    = "optim",
      help    = "optimization method: sgd | adam | adadelta | adagrad | adamax",
   }

   op:option{
      "--batch-size",
      default = 100,
      dest    = "batch_size",
      help    = "the batch size to use for training",
   }

   op:option{
      "--weight-init",
      default = "xavier",
      dest    = "weight_init",
      help    = "weight initialization mode: heuristic | xavier | xavier_caffe | kaiming",
   }

   op:option{
      "--weight-decay",
      default = 0.003,
      dest    = "weight_decay",
      help    = "L2 weight decay",
   }

   op:option{
      "--visualize",
      action  = "store_true",
      dest    = "visualize",
      help    = "visualize the first hidden layer's weights",
   }

   op:option{
      "--visual-check",
      action = "store_true",
      dest   = "visual_check",
      help   = "after training show input images one by one + the net's responses",
   }

   op:option{
      "--seed",
      dest = "seed",
      help = "manual seed for experiment repeatability",
   }

   local opts, args = op:parse()

   local function check(path)
      if path and not paths.filep(path) then
         op:fail("The '" .. path .. "' file does not exist!")
      end
   end

   if opts.seed then
      torch.manualSeed(tonumber(opts.seed))
   end

   check(opts.train_path)
   check(opts.test_path)
   check(opts.shared_path)
   check(opts.specific_path)

   opts.split      = tonumber(opts.split)
   opts.try_epochs = tonumber(opts.try_epochs)
   opts.batch_size = tonumber(opts.batch_size)
   opts.n          = tonumber(opts.n)

   return opts, args
end

local lossfmt = '%10.8f'
local accfmt  = '%7.3f%%'
local create_log = argcheck{
   {name='path', type='string', default='log.txt'},
   call =
      function(path)
         local logkeys = {
            'epoch',
            'train_loss',
            'train_acc',
            'valid_loss',
            'valid_acc',
            'learn_rate',
            'epoch_time',
            'new_best',
         }

         local logtext   = require 'torchnet.log.view.text'
         local logstatus = require 'torchnet.log.view.status'

         local format = {'%4d', lossfmt, accfmt, lossfmt, accfmt, '%9.7f', '%7.3fs', '%s'}

         return tnt.Log{
            keys = logkeys,
            onSet = {
               logstatus{}
            },
            onFlush = {
               logtext{
                  keys   = logkeys,
                  format = format,
               },
               logtext{
                  filename = path,
                  keys     = logkeys,
                  format   = format,
               },
            },
         }
      end
}

local visualize_layer = argcheck{
   {name='modules', type='table'},
   {name='window',  type='table', opt=true},
   {name='per_row', type='number', default=10},

   call =
      function(modules, window, per_row)
         local parameters
         for _, module in ipairs(modules) do
            if module.__typename == "nn.Linear"
               or module.__typename == "nn.LinearDropconnect" then
               local nunits = module.weight:size(1)
               parameters   = module.weight:view(nunits, 64, 64)
               break
            end
         end

         return image.display{
            image = image.toDisplayTensor{
               input   = parameters,
               nrow    = per_row,
               padding = 1},
            zoom = 2,
            win  = window
         }
      end
}

local visual_check = argcheck{
   {name='net', type='nn.Container'},
   {name='dataset', type='tnt.Dataset'},
   call =
      function(net, dataset)
         local w
         _G.interrupted = nil
         for data in test_dataset:iterator()() do
            w = image.display{image=data.input:view(1, 64, 64), win = w}
            local a = net:forward(data.input):squeeze()
            local b = data.target
            a = torch.cat(a, a:ge(0.5):double(), 2)
            print(torch.cat(a, b, 2):t())
            print("Press enter to load next example...")
            io.read()

            if _G.interrupted then
               if w then w.window:close() end
               break
            end
         end
      end
}

local extend_dataset = argcheck{
   {name='dataset', type='tnt.Dataset'},
   {name='net', type='nn.Container'},
   call =
      function(dataset, net)
         local input  = {}
         local target = {}

         net:evaluate()

         -- TODO MIGHT NOT WORK FOR SMALLER BATCH SIZES AS WELL???
         for data in dataset:batch(dataset:size()):iterator()() do
            local sm = net:forward(data.input)

            if type(sm) ~= 'table' then sm = {sm} end

            for i = 1, data.target:size(1) do
               local tmp = {data.target[i]}

               for j = 1, #sm do
                  table.insert(tmp, sm[j][i])
               end
               -- tnt.utils.table.foreach(
               --    sm,
               --    function(item)
               --       table.insert(tmp, item[i])
               --    end
               -- )

               table.insert(input, data.input[i])
               table.insert(target, tmp)
            end
         end

         input = tnt.utils.table.mergetensor(input)

         return tnt.ListDataset{
            list = torch.range(1, dataset:size()):long(),
            load =
               function(idx)
                  return {
                     input = input[idx],
                     target = target[idx],
                  }
               end,
         }
      end
}


-- [[MAIN BEGINS HERE ]]----

sig.signal(sig.SIGINT, sig.signal_handler)

local opts, args = parse_args(_G.arg)

local log      = create_log()
local engine   = tnt.OptimEngine()
local avgloss  = tnt.AverageValueMeter()
local emmeter = tnt.EMMeter()
local timer    = tnt.TimeMeter()
local stopper  = EarlyStopper(opts.try_epochs)

engine.hooks.onStartEpoch = function(state)
   avgloss:reset()
   emmeter:reset()
   timer:reset()
end

local visualize_window
engine.hooks.onStart = function(state)
   if state.training and opts.visualize then
      visualize_window = visualize_layer(state.network.modules, visualize_window)
   end
end

engine.hooks.onForwardCriterion = function(state)
   avgloss:add(state.criterion.output)
   if type(state.network.output) == 'table' then
      emmeter:add(state.network.output[1], state.sample.target[1])
   else
      emmeter:add(state.network.output, state.sample.target)
   end
end

engine.hooks.onEndEpoch = function(state)
   log:set{
      epoch      = state.epoch,
      train_loss = avgloss:value(),
      train_acc  = emmeter:value() * 100,
   }
   avgloss:reset()
   emmeter:reset()

   -- train_dataset:select('valid')
   state.iterator:exec('select', 'valid')
   state.iterator:exec('resample')
   engine:test{
      network   = state.network,
      -- iterator  = train_dataset:batch(train_dataset:size()):iterator(),
      iterator  = state.iterator,
      criterion = state.criterion,
   }
   -- train_dataset:select('train')
   state.iterator:exec('select', 'train')
   state.iterator:exec('resample') -- call :resample() on the underlying dataset

   stopper:epoch(emmeter:value(), state.network)

   log:set{
      valid_loss = avgloss:value(),
      valid_acc  = emmeter:value() * 100,
      learn_rate = state.config.learningRate,
      epoch_time = timer:value(),
      new_best   = stopper:improved() and '<--' or '',
   }
   log:flush()


   if opts.visualize then
      visualize_window = visualize_layer(state.network.modules, visualize_window)
   end

   if stopper:shouldStop() then
      if not tried_lowering then
         stopper:resetEpochs()
         state.config.learningRate = state.config.learningRate / 3
         tried_lowering = true
      end
   elseif stopper:improved() then
      tried_lowering = false
   end

   if stopper:shouldStop() or _G.interrupted then
      if visualize_window then visualize_window.window:close() end
      state.maxepoch = 0 -- end training
   end
end

-- local criterion
local net
local shared    = torch.load(opts.shared_path)
local specific  = torch.load(opts.specific_path)

local test_dataset  = torch.load(opts.test_path)

if #args == 0 then -- only the first specific + shared parameters to train
   criterion = nn.BCECriterion()

   net = nn.Sequential():add(shared):add(specific)

   print(net)

   if opts.train_path then
      local train_dataset = torch.load(opts.train_path):shuffle()
      net = require('weight-init')(net, opts.weight_init)
      engine:train{
         network     = net,
         iterator    =
            train_dataset
            :split({train = opts.split, valid = 1-opts.split}, 'train')
            :shuffle()
            :batch(opts.batch_size)
            :iterator(),
         criterion   = criterion,
         optimMethod = optim[opts.optim],
         maxepoch    = math.huge,
         config      = {
            learningRate = 0.001,
            weightDecay  = opts.weight_decay,
         }
      }

      net = stopper:getBestNet()
   end

   engine:test{
      network   = net,
      iterator  = test_dataset:batch(test_dataset:size()):iterator(),
      criterion = criterion,
   }

   print("Stats on the test set:")
   print(string.format("Loss: " .. lossfmt, avgloss:value()))
   print(string.format("Acc: " .. accfmt, emmeter:value() * 100))

   if opts.visual_check then
      visual_check(net, test_dataset)
   end

   if opts.train_path then
      if cmdio.check_useragrees("Overwrite shared net") then
         torch.save(opts.shared_path, net.modules[1])
         print("File '" .. opts.shared_path .. "' saved.")
      end
      if cmdio.check_useragrees("Overwrite specific net") then
         torch.save(opts.specific_path, net.modules[2])
         print("File '" .. opts.specific_path .. "' saved.")
      end
   end
else
   local train_dataset = torch.load(opts.train_path):shuffle()
   criterion = nn.BCECriterion()
   specific  = require('weight-init')(specific, opts.weight_init)
   print(specific)

   local preprocessed_input = {}
   local target = {}

   shared:evaluate()

   print("Pre-processing dataset for fine-tuning...")

   -- TODO DOES NOT WORK FOR SMALLER BATCH SIZE?!?!?!?!?!?!?
   for data in train_dataset:batch(train_dataset:size()):iterator()() do
      local a = shared:forward(data.input)
      -- local a = data.input
      for i = 1, data.input:size(1) do
         table.insert(preprocessed_input, a[i])
         table.insert(target, data.target[i])
      end
   end
   preprocessed_input = tnt.utils.table.mergetensor(preprocessed_input)
   target = tnt.utils.table.mergetensor(target)
   local preprocessed_dataset = tnt.ListDataset{
      list = torch.range(1, train_dataset:size()):long(),
      load =
         function(idx)
            return {
               input  = preprocessed_input[idx],
               target = target[idx],
            }
         end,
   }

   -- local pre2 = train_dataset:transform(function(input) return shared:forward(input) end, 'input')

   -- for i = 1, pre2:size() do
   --    local ai = preprocessed_dataset:get(i).input
   --    local bi = pre2:get(i).input
   --    local at = preprocessed_dataset:get(i).target
   --    local bt = pre2:get(i).target

   --    if ai:eq(bi):min() == 0 then
   --       print(ai)
   --       print(bi)
   --       error('inputs mismatch')
   --    end
   --    if at:eq(bt):min() == 0 then
   --       print(at)
   --       print(bt)
   --       error('targets mismatch')
   --    end
   -- end

   print("Fine-tuning new specific net...")

   engine:train{
      network  = specific,
      iterator =
         preprocessed_dataset
         :split({train = opts.split, valid = 1-opts.split}, 'train')
         :shuffle()
         :batch(opts.batch_size)
         -- :transform(function(input) return shared:forward(input) end, 'input')
         :iterator(),
      criterion   = criterion,
      optimMethod = optim[opts.optim],
      maxepoch    = math.huge,
      config      = {
         learningRate = 0.001,
         weightDecay  = opts.weight_decay,
      }
   }

   specific = stopper:getBestNet()

   engine:test{
      network   = nn.Sequential():add(shared):add(specific),
      iterator  = test_dataset:batch(test_dataset:size()):iterator(),
      criterion = criterion,
   }

   print("Stats on the test set:")
   print(string.format("Loss: " .. lossfmt, avgloss:value()))
   print(string.format("Acc: " .. accfmt, emmeter:value() * 100))

   criterion = nn.ParallelCriterion():add(nn.BCECriterion()) -- for the new spec. net

   local specific_old = tnt.utils.table.foreach(
      args,
      function(item)
         criterion:add(nn.DistKLDivCriterion()) -- for each old spec. net
         return torch.load(item)
      end
   )

   -- create copy of specific nets with shared parametes for later saving
   local for_saving = {
      shared        = shared,
      specific      = specific,
      specific_old  = tnt.utils.table.foreach(
         specific_old,
         function(item)
            return item:clone('weight', 'bias', 'running_mean', 'running_var')
         end
      )
   }

   -- modify old specific nets to output temperatured SoftMax
   tnt.utils.table.foreach(
      specific_old,
      function(item)
         item:remove() -- remove last module
         if opts.n ~= 1 then item:add(nn.MulConstant(1/opts.n)) end
         item:add(nn.LogSoftMax())
      end
   )

   stopper:setClosure(
      function()
         return {
            shared = for_saving.shared:clone():clearState(),
            specific = for_saving.specific:clone():clearState(),
            specific_old = tnt.utils.table.foreach(
               for_saving.specific_old,
               function(item)
                  return item:clone():clearState()
               end
            )
         }
      end
   )

   gshared = shared()
   local preprocess_net = nn.gModule(
      {gshared},
      tnt.utils.table.foreach(
         specific_old,
         function(item)
            return item(gshared)
         end
      )
   )

   print("Saving old specific nets' outputs...")

   train_dataset = extend_dataset(train_dataset, preprocess_net)
   test_dataset  = extend_dataset(test_dataset, preprocess_net)

   print("INCREMENTAL TRAINING...")

   _G.interrupted = false
   gshared = shared()
   gspecific = {specific(gshared)}
   tnt.utils.table.foreach(
      specific_old,
      function(item)
         table.insert(gspecific, item(gshared))
      end
   )

   engine:train{
      network  = nn.gModule({gshared}, gspecific),
      iterator =
         train_dataset
         :split({train = opts.split, valid = 1-opts.split}, 'train')
         :shuffle()
         :batch(opts.batch_size,
               function(idx, size) return idx end,
               function(table)
                  table.input  = tnt.utils.table.mergetensor(table.input)
                  table.target = tnt.transform.tablemergekeys()(table.target)
                  table.target = tnt.transform.tableapply(
                     function(field)
                        if tnt.utils.table.canmergetensor(field) then
                           return tnt.utils.table.mergetensor(field)
                        else
                           return field
                        end
                     end
                  )(table.target)
                  return table
               end)
         :iterator(),
      criterion   = criterion,
      optimMethod = optim[opts.optim],
      maxepoch    = math.huge,
      config      = {
         learningRate = 0.001,
         weightDecay  = opts.weight_decay,
      }
   }

   for_saving = stopper:getBestNet()

   if cmdio.check_useragrees("Overwrite shared net '"
         .. opts.shared_path .. "'") then
      torch.save(opts.shared_path, for_saving.shared)
      print("File saved.")
   end
   if cmdio.check_useragrees("Overwrite specific net '"
         .. opts.specific_path .. "'") then
      torch.save(opts.specific_path, for_saving.specific)
      print("File saved.")
   end

   for i = 1, #args do
      if cmdio.check_useragrees("Overwrite old specific net '"
            .. args[i] .. "'") then
         torch.save(args[i], for_saving.specific_old[i])
         print("File saved.")
      end
   end

   -- gshared = shared()
   -- gspecific = {specific(gshared)}
   -- tnt.utils.table.foreach(
   --    specific_old,
   --    function(item)
   --       table.insert(gspecific, item(gshared))
   --    end
   -- )
   -- engine:test{
   --    network   = nn.Sequential():add(shared):add(specific),
   --    iterator  = test_dataset:batch(test_dataset:size()):iterator(),
   --    criterion = criterion,
   -- }

   -- print("Stats on the test set:")
   -- print(string.format("Loss: " .. lossfmt, avgloss:value()))
   -- print(string.format("Acc: " .. accfmt, emmeter:value() * 100))

end
