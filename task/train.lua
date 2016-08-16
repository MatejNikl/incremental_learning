local sig   = require 'signal'

local argcheck = require 'argcheck'
local image    = require 'image'
local nn       = require 'nn'
local optim    = require 'optim'
local tnt      = require 'torchnet'


local function parse_args(args)
    local op = xlua.OptionParser("dataset_creator.lua -c|--csv=CSV -d|--dir=DIR"
        .. "[-p|--pattern ptrn] OUTPUT_FILE")

    op:option{
        "--train",
        dest   = "train_path",
        help   = "a file containing training data",
        req    = true,
    }

    op:option{
        "--test",
        dest   = "test_path",
        help   = "a file containing testing data",
        req    = true,
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
        help    = "batch normalization",
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

    if not paths.filep(opts.train_path) then
        op:fail("The training dataset file must exist!")
    elseif not paths.filep(opts.test_path) then
        op:fail("The testing dataset file must exit!")
    end

    opts.dropout = tonumber(opts.dropout)
    opts.dropconnect = tonumber(opts.dropconnect)
    opts.layers = loadstring("return {" .. opts.layers .. "}")()

    return opts, args
end

local function create_net(opts)
    local net = nn.Sequential()

    if opts.dropconnect > 0 then
        require 'dropconnect'
    end

    for i = 2, #opts.layers do
        local nprev = opts.layers[i-1]
        local ncurr = opts.layers[i]

        if opts.dropout > 0 then
            net:add(nn.Dropout(opts.dropout))
        end

        if opts.dropconnect == 0 then
            net:add(nn.Linear(nprev, ncurr))
        else
            net:add(nn.LinearDropconnect(nprev, ncurr, opts.dropconnect))
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

local print_acc = argcheck{
    {name='avgloss', type='tnt.AverageValueMeter', help='average loss'},
    {name='apmeter', type='tnt.APMeter', help='precision per class'},
    call =
        function(avgloss, apmeter)
            local str = string.format('avg. loss: %2.10f, precision per class:',
                avgloss:value())
            local val = apmeter:value()

            for i = 1, val:nElement() do
                str = str .. string.format(' %7.3f%%', val[i] * 100)
            end

            print(str)
        end
}

sig.signal(sig.SIGINT, sig.signal_handler)

local opts, args = parse_args(_G.arg)

local train_dataset = torch.load(opts.train_path)
local test_dataset  = torch.load(opts.test_path)

table.insert(opts.layers, 1, train_dataset:get(1).input:nElement())
table.insert(opts.layers, train_dataset:get(1).target:nElement())

local net = create_net(opts)
print(net)


local criterion = nn.BCECriterion()

local engine  = tnt.OptimEngine()
local apmeter = tnt.APMeter()
local avgloss = tnt.AverageValueMeter()


engine.hooks.onStartEpoch = function(state)
    avgloss:reset()
    apmeter:reset()
    print("Epoch: " .. state.epoch)
end

engine.hooks.onForwardCriterion = function(state)
    avgloss:add(state.criterion.output)
    apmeter:add(state.network.output, state.sample.target)

    if state.training then
        print_acc(avgloss, apmeter)
    end
end

local visualize_window
engine.hooks.onEndEpoch = function(state)
    state.iterator:exec('resample') -- call :resample() on the underlying dataset

    if opts.visualize then
        local parameters
        for _, module in ipairs(net.modules) do
            if module.__typename == "nn.Linear"
            or module.__typename == "nn.LinearDropconnect" then
                parameters = module.weight:view(opts.layers[2], 64, 64)
                break
            end
        end

        visualize_window = image.display{
            image = parameters,
            zoom  = 2,
            win   = visualize_window
        }
    end

    if _G.interrupted then
        state.maxepoch = 0 -- end training
    end
end


-- train the model:
engine:train{
    network     = net,
    iterator    = train_dataset:shuffle():batch(opts.batch_size):iterator(),
    criterion   = criterion,
    optimMethod = optim[opts.optim],
    maxepoch    = 100,
}

-- measure test loss and error:
avgloss:reset()
apmeter:reset()

engine:test{
    network   = net,
    iterator  = test_dataset:batch(100):iterator(),
    criterion = criterion,
}


print("Stats on the test set:")
print_acc(avgloss, apmeter)


-- for data in test_dataset:iterator()() do
--     w = image.display{image=data.input:view(1, 64, 64), win = w}
--     local a = net:forward(data.input):squeeze()
--     local b = data.target
--     a = torch.cat(a, a:ge(0.5):double(), 2)
--     print(torch.cat(a, b, 2):t())
--     io.read()
-- end

