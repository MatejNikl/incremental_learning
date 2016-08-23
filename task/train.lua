local sig   = require 'signal'

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
        dest = "visual_check",
        help = "after training show input images one by one + the net's responses",
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
    opts.layers = loadstring("return {" .. opts.layers .. "}")()

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


local logkeys = {'loss', 'accuracy', 'per_class'}

local logtext   = require 'torchnet.log.view.text'
local logstatus = require 'torchnet.log.view.status'

local log = tnt.Log{
    keys = logkeys,
    onSet = {
        logstatus{}
    },
    onFlush = {
        logtext{
            keys   = logkeys,
            format = {'%10.8f', '%7.3f%%', '%s'},
        },
        logtext{
            filename = 'log.txt',
            keys     = logkeys,
            format   = {'%10.8f', '%7.3f', '%s'},
        },
    },
}

sig.signal(sig.SIGINT, sig.signal_handler)

local opts, args = parse_args(_G.arg)

local train_dataset = opts.train_path and torch.load(opts.train_path) or nil
local test_dataset  = opts.test_path and torch.load(opts.test_path) or nil

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

local engine  = tnt.OptimEngine()
local apmeter = tnt.APMeter()
local avgloss = tnt.AverageValueMeter()

apmeter.strvalue = argcheck{
    {name='self', type='tnt.APMeter'},
    call =
        function(self)
            local str = ''
            local val = self:value()
            for i = 1, val:nElement() do
                str = str .. string.format('%7.3f%% ', val[i] * 100)
            end

            return str
        end
}


engine.hooks.onStartEpoch = function(state)
    avgloss:reset()
    apmeter:reset()
    log:status("Epoch: " .. state.epoch)
end

local visualize_window
engine.hooks.onForwardCriterion = function(state)
    avgloss:add(state.criterion.output)
    apmeter:add(state.network.output, state.sample.target)

    if opts.visualize then
        local parameters
        local nunits
        for _, module in ipairs(net.modules) do
            if module.__typename == "nn.Linear"
            or module.__typename == "nn.LinearDropconnect" then
                nunits     = module.weight:size(1)
                parameters = module.weight:view(nunits, 64, 64)
                break
            end
        end

        visualize_window = image.display{
            image = image.toDisplayTensor{
                input   = parameters,
                nrow    = 10,
                padding = 1},
            zoom  = 2,
            win   = visualize_window
        }
    end

    if state.training then
        log:set{
            loss      = avgloss:value(),
            accuracy  = apmeter:value():mean() * 100,
            per_class = apmeter:strvalue(),
        }
        log:flush()

    end
end

engine.hooks.onEndEpoch = function(state)
    state.iterator:exec('resample') -- call :resample() on the underlying dataset

    if _G.interrupted then
        state.maxepoch = 0 -- end training
    end
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
    avgloss:reset()
    apmeter:reset()

    engine:test{
        network   = net,
        iterator  = test_dataset:batch(test_dataset:size()):iterator(),
        criterion = criterion,
    }

    log:status("Stats on the test set:")
    log:set{
        loss      = avgloss:value(),
        accuracy  = apmeter:value():mean() * 100,
        per_class = apmeter:strvalue(),
    }
    log:flush()

    if opts.visual_check then
        local w
        for data in test_dataset:iterator()() do
            w = image.display{image=data.input:view(1, 64, 64), win = w}
            local a = net:forward(data.input):squeeze()
            local b = data.target
            a = torch.cat(a, a:ge(0.5):double(), 2)
            print(torch.cat(a, b, 2):t())
            print("Press enter to load next example...")
            io.read()

            if _G.interrupted then break end
        end
    end
end

if #args > 0 then
    torch.save(args[1], net:clearState())
    print("Saved the trained network as '" .. args[1] .. "'")
end

if visualize_window then visualize_window.window:close() end
