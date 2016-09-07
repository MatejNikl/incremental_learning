local cmdio = require 'cmdio'
local sig   = require 'signal'

local argcheck = require 'argcheck'
local image    = require 'image'
local nn       = require 'nn'
local optim    = require 'optim'
local tnt      = require 'torchnet'
local nngraph  = require 'nngraph'


require 'dropconnect'
require 'earlystopper'
require 'exactmatchmeter'

local function parse_args(args)
   local function print_settings(op)
      local lines = {}
      string.gsub(op:tostring(), "(%S* = %S*)", function(a) table.insert(lines, a) end)
      table.sort(lines)

      print("\nSettings for this experiment:")
      for _, line in ipairs(lines) do
         print("+ " .. line)
      end
      print()
   end

   local op = xlua.OptionParser("train.lua --task SCT[1-6] --net-dir .../net_dir"
      .. " [--train-dir .../train_dir] --test-dir .../test_dir"
      .. " [OPTIONS...] ALREADY_TRAINED_TASKS...\n\n"
      .. [[
Further description to fill in...]])

   op:option{
      "--task",
      dest = "task",
      help = "a name of the new task to learn [or the task to test if"
         .. " --train-dir is not present]",
      req  = true,
   }

   op:option{
      "--net-dir",
      dest = "net_dir",
      help = "a directory containing parts of the whole nn",
      req  = true,
   }

   op:option{
      "--train-dir",
      dest = "train_dir",
      help = "a directory containing train datasets",
   }

   op:option{
      "--test-dir",
      dest = "test_dir",
      help = "a directory containing test datasets",
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
      default = 10,
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
      "--soft-crit",
      default = "KLDiv",
      dest    = "soft_crit",
      help    = "criterion to use on soft target: KLDiv | Abs | MSE",
   }

   op:option{
      "--finetune",
      action  = "store_true",
      dest    = "finetune",
      help    = "perform finetuning of the new specific layer",
   }

   op:option{
      "--optim",
      default = "adam",
      dest    = "optim",
      help    = "optimization method: sgd | adam | lbfgs | any other method in optim"
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
      help    = "weight initialization mode: xavier | heuristic | xavier_caffe | kaiming",
   }

   op:option{
      "--learning-rate",
      default = 0.001,
      dest    = "learning_rate",
      help    = "learning rate to start with",
   }

   op:option{
      "--momentum",
      dest    = "momentum",
      help    = "momentum for sgd optim method",
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

   op:option{
      "--log",
      dest = "log",
      help = "name of the log file (inside --net-dir)",
   }

   local opts, args = op:parse()

   opts.split         = tonumber(opts.split)
   opts.try_epochs    = tonumber(opts.try_epochs)
   opts.batch_size    = tonumber(opts.batch_size)
   opts.n             = tonumber(opts.n)
   opts.lambda        = tonumber(opts.lambda)
   opts.weight_decay  = tonumber(opts.weight_decay)
   opts.learning_rate = tonumber(opts.learning_rate)
   opts.momentum      = tonumber(opts.momentum)

   if opts.soft_crit ~= 'KLDiv'
      and opts.soft_crit ~= 'Abs'
      and opts.soft_crit ~= 'MSE' then
      op:error("Unknown soft criterion!")
   end

   if opts.seed then
      torch.manualSeed(tonumber(opts.seed))
   end

   args = tnt.utils.table.foreach(
      args,
      function(a)
         return paths.concat(opts.net_dir, a) .. ".t7"
      end
   )

   print_settings(op)

   return opts, args
end

local lossfmt = '%8.6f'
local accfmt  = '%6.2f%%'
local create_log = argcheck{
   {name='path', type='string', default='/dev/null'},
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


-- not local so that it can be called resursively
visualize_layer = argcheck{
   {name='path',  type='string'},
   {name='net', type='nn.Container'},
   {name='per_row', type='number', default=10},

   call =
      function(path, net, per_row)
         local parameters
         for _, module in ipairs(net.modules) do
            if module.__typename == "nn.Linear"
               or module.__typename == "nn.LinearDropconnect" then
               local nunits = module.weight:size(1)
               parameters   = module.weight:view(nunits, 64, 64)
               break
            elseif module.__typename == 'nn.Sequential' then
               return visualize_layer(path, module, per_row) -- recursion
            end
         end

         image.save(path,
            image.toDisplayTensor{
               input   = parameters,
               nrow    = per_row,
               padding = 1
            }
         )
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

local write = argcheck{
   {name='where', type='string'},
   {name='what', type='nn.Container'},
   call =
      function(where, what)
         torch.save(where, what)
         print("File '" .. where .. "' saved.")
      end
}


-- [[MAIN BEGINS HERE ]]----

sig.signal(sig.SIGINT, sig.signal_handler)

local opts, args = parse_args(_G.arg)

local logpath   = paths.concat(opts.net_dir,
                               opts.log and opts.log or opts.task .. ".txt")
local log       = create_log(logpath)
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

engine.hooks.onStart = function(state)
   if state.training then
      stopper:reset()
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
         local path = paths.concat(opts.net_dir, opts.task
                                                .. "_" .. state.epoch .. ".png")
         visualize_layer(path, state.network)
      end
   end
end

engine.hooks.onForwardCriterion = function(state)
   if type(state.network.output) == 'table' then
      local hl = state.criterion.criterions[1].output
      hard_loss:add(hl)
      soft_loss:add(state.criterion.output - hl)
      emmeter:add(state.network.output[1], state.sample.target[1])
   else
      hard_loss:add(state.criterion.output)
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
      local path = paths.concat(opts.net_dir, opts.task
                                              .. "_" .. state.epoch .. ".png")
      visualize_layer(path, state.network)
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
      state.maxepoch = 0 -- end training
   end
end

local criterion
local net
local shared_path   = paths.concat(opts.net_dir, "shared.t7")
local specific_path = paths.concat(opts.net_dir, opts.task) .. ".t7"
local shared        = torch.load(shared_path)
local specific      = torch.load(specific_path)

local test_dataset = torch.load(paths.concat(opts.test_dir, opts.task) .. ".t7")

if #args == 0 then -- only the first specific + shared parameters to train
   criterion = nn.BCECriterion()

   net = nn.Sequential():add(shared):add(specific)

   print(net)

   if opts.train_dir then
      local train_path = paths.concat(opts.train_dir, opts.task) .. ".t7"
      local train_dataset = torch.load(train_path):shuffle()
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
            momentum     = opts.momentum,
         }
      }

      net = stopper:getBestNet()
   end

   engine:test{
      network   = net,
      iterator  = test_dataset:batch(test_dataset:size()):iterator(),
      criterion = criterion,
   }

   log:status("Stats on the test set:")
   log:set{
      train_hardloss = hard_loss:value(),
      train_softloss = soft_loss:value(),
      train_acc      = emmeter:value() * 100,
   }
   log:flush()

   if opts.visual_check then
      visual_check(net, test_dataset)
   end

   if opts.train_dir then
      if cmdio.check_useragrees("Write trained nets") then
         write(specific_path, net.modules[2])
         write(shared_path, net.modules[1])
      end
   end
else
   local train_path = paths.concat(opts.train_dir, opts.task) .. ".t7"
   local train_dataset = torch.load(train_path):shuffle()
   specific  = require('weight-init')(specific, opts.weight_init)
   print(specific)

   if opts.finetune then
      criterion = nn.BCECriterion()
      log:status("Pre-processing dataset for fine-tuning...")
      local preprocessed_dataset = preprocess_dataset(train_dataset, shared, 'input')
      log:status("Fine-tuning new specific net...")

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
            momentum     = opts.momentum,
         }
      }

      specific = stopper:getBestNet()

      engine:test{
         network   = nn.Sequential():add(shared):add(specific),
         iterator  = test_dataset:batch(test_dataset:size()):iterator(),
         criterion = criterion,
      }

      log:status("Stats on the test set:")
      log:set{
         train_hardloss = hard_loss:value(),
         train_softloss = soft_loss:value(),
         train_acc      = emmeter:value() * 100,
      }
      log:flush()
   end

   criterion = nn.ParallelCriterion():add(nn.BCECriterion()) -- for the new spec. net

   local specific_old = tnt.utils.table.foreach(
      args,
      function(item)
         -- for each old spec. net add criterion
         local crit
         if opts.soft_crit == "KLDiv" then
            criterion:add(nn.DistKLDivCriterion(), opts.lambda/#args)
         elseif opts.soft_crit == "Abs" then
            criterion:add(nn.AbsCriterion(), opts.lambda/#args)
         elseif opts.soft_crit == "MSE" then
            criterion:add(nn.MSECriterion(), opts.lambda/#args)
         end

         -- and return loaded spec net
         return torch.load(item)
      end
   )

   if opts.soft_crit == "KLDiv" then
      -- modify old specific nets to output temperatured SoftMax
      tnt.utils.table.foreach(
         specific_old,
         function(item)
            item:remove() -- remove last module
            if opts.n ~= 1 then item:add(nn.MulConstant(1/opts.n)) end
            item:add(nn.SoftMax())
         end
      )
   end

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

   log:status("Saving old specific nets' outputs...")

   train_dataset = preprocess_dataset(train_dataset, preprocess_net, 'target')
   test_dataset  = preprocess_dataset(test_dataset,  preprocess_net, 'target')

   log:status("INCREMENTAL TRAINING...")

   if opts.soft_crit == "KLDiv" then
      -- modify old specific nets to output temperatured LogSoftMax
      tnt.utils.table.foreach(
         specific_old,
         function(item)
            item:remove() -- remove last module
            item:add(nn.LogSoftMax())
         end
      )
   end

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

   local function mergefunc(table)
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
   end

   engine:train{
      network  = net,
      iterator =
         train_dataset
         :split({train = opts.split, valid = 1-opts.split}, 'train')
         :shuffle()
         :batch(opts.batch_size,
               function(idx, size) return idx end,
               mergefunc)
         :iterator(),
      criterion   = criterion,
      optimMethod = optim[opts.optim],
      maxepoch    = math.huge,
      config      = {
         learningRate = opts.learning_rate,
         weightDecay  = opts.weight_decay,
         momentum     = opts.momentum,
      }
   }

   net = stopper:getBestNet()

   engine:test{
      network   = net,
      iterator  = test_dataset
         :batch(test_dataset:size(),
               function(idx, size) return idx end,
               mergefunc)
         :iterator(),
      criterion = criterion,
   }

   log:status("Stats on the test set:")
   log:set{
      train_hardloss = hard_loss:value(),
      train_softloss = soft_loss:value(),
      train_acc      = emmeter:value() * 100,
   }
   log:flush()

   if cmdio.check_useragrees("Write trained nets") then
      for i = 1, #args do
         local module = net.modules[i+2]
         if opts.soft_crit == "KLDiv" then
            module:remove()
            if opts.n ~= 1 then module:remove() end
            module:add(nn.Sigmoid())
         end

         write(args[i], module)
      end

      write(specific_path, net.modules[2])
      write(shared_path, net.modules[1])
   end
end
