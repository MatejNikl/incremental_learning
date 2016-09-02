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
      "--lambda",
      default = 10,
      dest    = "lambda",
      help    = "weight on soft target (hard target's weight = 1)",
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
      "--learning-rate",
      default = 0.001,
      dest    = "learning_rate",
      help    = "learning rate to start with",
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

   opts.split         = tonumber(opts.split)
   opts.try_epochs    = tonumber(opts.try_epochs)
   opts.batch_size    = tonumber(opts.batch_size)
   opts.n             = tonumber(opts.n)
   opts.lambda        = tonumber(opts.lambda)
   opts.weight_decay  = tonumber(opts.weight_decay)
   opts.learning_rate = tonumber(opts.learning_rate)

   return opts, args
end

local lossfmt = '%8.6f'
local accfmt  = '%6.2f%%'
local create_log = argcheck{
   {name='path', type='string', default='log.txt'},
   call =
      function(path)
         local logkeys = {
            'epoch',
            'train_hardloss',
            'train_softloss',
            'train_acc',
            'valid_hardloss',
            'valid_softloss',
            'valid_acc',
            'learn_rate',
            'epoch_time',
            'new_best',
         }

         local logtext   = require 'torchnet.log.view.text'
         local logstatus = require 'torchnet.log.view.status'

         local format = {
            '%3d',
            lossfmt,
            lossfmt,
            accfmt,
            lossfmt,
            lossfmt,
            accfmt,
            '%8.6f',
            '%5.2fs',
            '%s'}

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
         for data in dataset:iterator()() do
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

local preprocess_dataset = argcheck{
   {name='dataset', type='tnt.Dataset'},
   {name='net', type='nn.Container'},
   {name='save_as', type='string'},
   {name='batch_size', type='number', opt=true},
   call =
      function(dataset, net, save_as, batch_size)
         assert(save_as == 'input' or save_as == 'target',
            "Only can save as input or as target")

         batch_size = batch_size or dataset:size()
         net:evaluate()

         local input  = {}
         local target = {}

         for batch in dataset:batch(batch_size):iterator()() do
            local net_output = net:forward(batch.input)

            if save_as == 'input' then
               assert(type(net_output) == 'userdata')

               for i = 1, batch.target:size(1) do
                  table.insert(input, net_output[i]:clone())
                  table.insert(target, batch.target[i])
               end
            else
               if type(net_output) ~= 'table' then net_output = {net_output} end
               for i = 1, batch.target:size(1) do
                  local tmp = {batch.target[i]}

                  for j = 1, #net_output do
                     table.insert(tmp, net_output[j][i]:clone())
                  end

                  table.insert(input, batch.input[i])
                  table.insert(target, tmp)
               end
            end
         end

         if tnt.utils.table.canmergetensor(input) then
            input = tnt.utils.table.mergetensor(input)
         end

         if tnt.utils.table.canmergetensor(target) then
            target = tnt.utils.table.mergetensor(target)
         end

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

local log       = create_log()
local engine    = tnt.OptimEngine()
local hard_loss = tnt.AverageValueMeter()
local soft_loss = tnt.AverageValueMeter()
local emmeter   = tnt.EMMeter()
local timer     = tnt.TimeMeter()
local stopper   = EarlyStopper(opts.try_epochs)

engine.hooks.onStartEpoch = function(state)
   hard_loss:reset()
   soft_loss:reset()
   emmeter:reset()
   timer:reset()
end

local visualize_window
engine.hooks.onStart = function(state)
   if state.training then
      stopper:resetEpochs()
      hard_loss:reset()
      soft_loss:reset()
      emmeter:reset()
      timer:reset()

      engine:test{
         network   = state.network,
         iterator  = state.iterator,
         criterion = state.criterion,
      }

      log:set{
         epoch          = state.epoch,
         train_hardloss = hard_loss:value(),
         train_softloss = soft_loss:value(),
         train_acc      = emmeter:value() * 100,
      }
      hard_loss:reset()
      soft_loss:reset()
      emmeter:reset()

      state.iterator:exec('select', 'valid')
      state.iterator:exec('resample')
      engine:test{
         network   = state.network,
         iterator  = state.iterator,
         criterion = state.criterion,
      }
      state.iterator:exec('select', 'train')
      state.iterator:exec('resample') -- call :resample() on the underlying dataset

      log:set{
         valid_hardloss = hard_loss:value(),
         valid_softloss = soft_loss:value(),
         valid_acc      = emmeter:value() * 100,
         learn_rate     = 0/0,
         epoch_time     = timer:value(),
         new_best       = stopper:improved() and '*' or '',
      }
      log:flush()

      if opts.visualize then
         visualize_window = visualize_layer(state.network.modules, visualize_window)
      end
   end
end

engine.hooks.onForwardCriterion = function(state)
   if type(state.network.output) == 'table' then
      local total_loss = 0
      local crits   = state.criterion.criterions
      local weights = state.criterion.weights
      for i = 2, #crits do
         total_loss = total_loss + crits[i].output * weights[i]
      end

      hard_loss:add(crits[1].output)
      soft_loss:add(total_loss)
      emmeter:add(state.network.output[1], state.sample.target[1])
   else
      hard_loss:add(state.criterion.output)
      soft_loss:add(0/0)
      emmeter:add(state.network.output, state.sample.target)
   end
end

engine.hooks.onEndEpoch = function(state)
   log:set{
      epoch          = state.epoch,
      train_hardloss = hard_loss:value(),
      train_softloss = soft_loss:value(),
      train_acc      = emmeter:value() * 100,
   }

   hard_loss:reset()
   soft_loss:reset()
   emmeter:reset()

   -- train_dataset:select('valid')
   state.iterator:exec('select', 'valid')
   state.iterator:exec('resample')
   engine:test{
      network   = state.network,
      iterator  = state.iterator,
      criterion = state.criterion,
   }
   -- train_dataset:select('train')
   state.iterator:exec('select', 'train')
   state.iterator:exec('resample') -- call :resample() on the underlying dataset

   local sl_val = soft_loss:value()
   stopper:epoch(
      emmeter:value(),
      sl_val ~= sl_val and hard_loss:value() or sl_val,
      state.network
   )

   log:set{
      valid_hardloss = hard_loss:value(),
      valid_softloss = soft_loss:value(),
      valid_acc      = emmeter:value() * 100,
      learn_rate     = state.config.learningRate,
      epoch_time     = timer:value(),
      new_best       = stopper:improved() and '*' or '',
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
            learningRate = opts.learning_rate,
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
   print(string.format("Loss: " .. lossfmt, hard_loss:value()))
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


   print("Pre-processing dataset for fine-tuning...")
   preprocessed_dataset = preprocess_dataset(train_dataset, shared, 'input')


   print("Fine-tuning new specific net...")

   engine:train{
      network  = specific,
      iterator =
         preprocessed_dataset
         :split({train = opts.split, valid = 1-opts.split}, 'train')
         :shuffle()
         :batch(opts.batch_size)
         :iterator(),
      criterion   = criterion,
      optimMethod = optim[opts.optim],
      maxepoch    = math.huge,
      config      = {
         learningRate = opts.learning_rate,
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
   print(string.format("Loss: " .. lossfmt, hard_loss:value()))
   print(string.format("Acc: " .. accfmt, emmeter:value() * 100))

   criterion = nn.ParallelCriterion():add(nn.BCECriterion()) -- for the new spec. net

   local specific_old = tnt.utils.table.foreach(
      args,
      function(item)
         -- for each old spec. net add criterion
         criterion:add(nn.DistKLDivCriterion(), opts.lambda/#args)
         return torch.load(item)
      end
   )

   -- modify old specific nets to output temperatured SoftMax
   tnt.utils.table.foreach(
      specific_old,
      function(item)
         item:remove() -- remove last module
         if opts.n ~= 1 then item:add(nn.MulConstant(1/opts.n)) end
         item:add(nn.SoftMax())
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

   train_dataset = preprocess_dataset(train_dataset, preprocess_net, 'target')
   test_dataset  = preprocess_dataset(test_dataset,  preprocess_net, 'target')

   print("INCREMENTAL TRAINING...")

   -- modify old specific nets to output temperatured LogSoftMax
   tnt.utils.table.foreach(
      specific_old,
      function(item)
         item:remove() -- remove last module
         item:add(nn.LogSoftMax())
      end
   )


   _G.interrupted = false
   gshared = {shared()}
   gspecific = {specific(gshared)}
   tnt.utils.table.foreach(
      specific_old,
      function(item)
         table.insert(gspecific, item(gshared))
      end
   )

   local net = nn.gModule(gshared, gspecific)

   engine:train{
      network  = net,
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
         learningRate = opts.learning_rate,
         weightDecay  = opts.weight_decay,
      }
   }

   net = stopper:getBestNet()

   engine:test{
      network   = net,
      iterator  = test_dataset
         :batch(test_dataset:size(),
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
      criterion = criterion,
   }

   print("Stats on the test set:")
   print(string.format("hard loss: " .. lossfmt, hard_loss:value()))
   print(string.format("soft loss: " .. lossfmt, soft_loss:value()))
   print(string.format("Acc: " .. accfmt, emmeter:value() * 100))

   if cmdio.check_useragrees("Overwrite shared net '"
         .. opts.shared_path .. "'") then
      torch.save(opts.shared_path, net.modules[1])
      print("File saved.")
   end
   if cmdio.check_useragrees("Overwrite specific net '"
         .. opts.specific_path .. "'") then
      torch.save(opts.specific_path, net.modules[2])
      print("File saved.")
   end

   for i = 1, #args do
      if cmdio.check_useragrees("Overwrite old specific net '"
            .. args[i] .. "'") then
         local module = net.modules[i+2]
         module:remove()
         if opts.n ~= 1 then module:remove() end
         module:add(nn.Sigmoid())

         torch.save(args[i], module)
         print("File saved.")
      end
   end

end
