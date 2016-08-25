local sig = require 'signal'

local argcheck = require 'argcheck'
local image    = require 'image'
local nn       = require 'nn'
local optim    = require 'optim'
local tnt      = require 'torchnet'

require 'dropconnect'

local function parse_args(args)
   local op = xlua.OptionParser("train.lua --train TRAIN_DATASET"
      .. " --test TEST_DATASET [OPTIONS...] SAVE_NET.t7")

   op:option{
      "--train",
      dest = "train_path",
      help = "a file containing training data",
   }

   op:option{
      "--test",
      dest = "test_path",
      help = "a file containing testing data",
   }

   op:option{
      "--visual-check",
      action = "store_true",
      dest   = "visual_check",
      help   = "after training show input images one by one + the net's responses",
   }

   op:option{
      "--use-net",
      dest = "use_net",
      help = "a file containing a previously saved nn",
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
      "--act",
      default = "ReLU",
      dest    = "act",
      help    = "activation function: ReLU | ELU | Tanh | Sigmoid",
   }

   op:option{
      "--layers",
      dest    = "layers",
      default = "100,100",
      help    = "comma separated hidden layer sizes",
   }

   op:option{
      "--batchnorm",
      action  = "store_true",
      default = false,
      dest    = "batchnorm",
      help    = "use batch normalization before each activation",
   }

   op:option{
      "--dropout",
      default = 0,
      dest    = "dropout",
      help    = "use dropout with specified probability",
   }

   op:option{
      "--dropconnect",
      dest    = "dropconnect",
      default = 0,
      help    = "use dropconnect with specified probability",
   }

   op:option{
      "--weight-init",
      default = "xavier",
      dest    = "weight_init",
      help    = "weight initialization mode: heuristic | xavier | xavier_caffe | kaiming",
   }

   op:option{
      "--visualize",
      action  = "store_true",
      dest    = "visualize",
      help    = "visualize the first hidden layer's weights",
   }

   op:option{
      "--split",
      default = 0.8,
      dest    = "split",
      help    = "keep this percentage for training and the rest for validation"
   }


   local opts, args = op:parse()

   if opts.train_path and not paths.filep(opts.train_path) then
      op:fail("The training dataset file must exist!")
   elseif opts.test_path and not paths.filep(opts.test_path) then
      op:fail("The testing dataset file must exit!")
   elseif opts.use_net and not paths.filep(opts.use_net) then
      op:fail("The file containing previously saved nn must exit!")
   end

   opts.batch_size  = tonumber(opts.batch_size)
   opts.dropout     = tonumber(opts.dropout)
   opts.dropconnect = tonumber(opts.dropconnect)
   opts.split       = tonumber(opts.split)
   opts.layers      = loadstring("return {" .. opts.layers .. "}")()

   return opts, args
end

local function create_net(opts)
   local net = nn.Sequential()
   local bias = not opts.batchnorm

   for i = 2, #opts.layers do
      local nprev = opts.layers[i-1]
      local ncurr = opts.layers[i]

      if opts.dropout > 0 then
         net:add(nn.Dropout(opts.dropout))
      end

      if opts.dropconnect == 0 then
         net:add(nn.Linear(nprev, ncurr, bias))
      else
         net:add(nn.LinearDropconnect(nprev, ncurr, opts.dropconnect, bias))
      end

      if opts.batchnorm then
         net:add(nn.BatchNormalization(ncurr))
      end

      if i < #opts.layers then
         net:add(nn[opts.act]())
      else
         net:add(nn.Sigmoid())
      end
   end

   net = require('weight-init')(net, opts.weight_init)

   return net
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
            'epoch_time',
         }

         local logtext   = require 'torchnet.log.view.text'
         local logstatus = require 'torchnet.log.view.status'

         local format  = {'%4d', lossfmt, accfmt, lossfmt, accfmt, '%7.3fs'}

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


sig.signal(sig.SIGINT, sig.signal_handler)

local opts, args = parse_args(_G.arg)

local train_dataset =
   opts.train_path
   and torch.load(opts.train_path):shuffle():split{
      train = opts.split,
      valid = 1 - opts.split
   }
   or nil

local test_dataset = opts.test_path and torch.load(opts.test_path) or nil

local net
if opts.use_net then
   net = torch.load(opts.use_net)
   print("Loaded a nn from the file '" .. opts.use_net .. "'")
else
   table.insert(opts.layers, 1, train_dataset:get(1).input:nElement())
   table.insert(opts.layers, train_dataset:get(1).target:nElement())
   net = create_net(opts)
end

print(net)


local criterion = nn.BCECriterion()

local log      = create_log()
local engine   = tnt.OptimEngine()
local avgloss  = tnt.AverageValueMeter()
local mapmeter = tnt.mAPMeter()
local timer    = tnt.TimeMeter()

engine.hooks.onStartEpoch = function(state)
   avgloss:reset()
   mapmeter:reset()
   timer:reset()
end

local visualize_window

engine.hooks.onStart = function(state)
   if opts.visualize then
      visualize_window = visualize_layer(net.modules, visualize_window)
   end
end

engine.hooks.onForwardCriterion = function(state)
   avgloss:add(state.criterion.output)
   mapmeter:add(state.network.output, state.sample.target)
end

engine.hooks.onEndEpoch = function(state)
   log:set{
      epoch      = state.epoch,
      train_loss = avgloss:value(),
      train_acc  = mapmeter:value() * 100,
   }
   avgloss:reset()
   mapmeter:reset()

   train_dataset:select('valid')
   engine:test{
      network   = net,
      iterator  = train_dataset:batch(train_dataset:size()):iterator(),
      criterion = criterion,
   }
   train_dataset:select('train')

   log:set{
      valid_loss = avgloss:value(),
      valid_acc  = mapmeter:value() * 100,
      epoch_time = timer:value(),
   }
   log:flush()

   if opts.visualize then
      visualize_window = visualize_layer(net.modules, visualize_window)
   end

   if _G.interrupted then
      if visualize_window then visualize_window.window:close() end
      state.maxepoch = 0 -- end training
   end

   state.iterator:exec('resample') -- call :resample() on the underlying dataset
end


if opts.train_path then
   -- train the model:
   engine:train{
      network     = net,
      iterator    = train_dataset:shuffle():batch(opts.batch_size):iterator(),
      criterion   = criterion,
      optimMethod = optim[opts.optim],
      maxepoch    = 100,
   }
end

if opts.test_path then
   -- measure test loss and error:
   engine:test{
      network   = net,
      iterator  = test_dataset:batch(test_dataset:size()):iterator(),
      criterion = criterion,
   }

   print("Stats on the test set:")
   print(string.format("Loss: " .. lossfmt, avgloss:value()))
   print(string.format("Acc: " .. accfmt, mapmeter:value() * 100))

   if opts.visual_check then
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
end

if #args > 0 then
   torch.save(args[1], net:clearState())
   print("Saved the trained network as '" .. args[1] .. "'")
end
