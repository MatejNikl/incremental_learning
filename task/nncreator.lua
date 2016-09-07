local cmdio = require 'cmdio'

local argcheck = require 'argcheck'
local nn       = require 'nn'
local tnt      = require 'torchnet'

require 'dropconnect'

local function parse_args(args)
   local op = xlua.OptionParser("nn_creator.lua --shared|--specific"
      .. " --layers #hl1,#hl2,... [OPTIONS...] DATASETS...\n\n"
      .. [[
For each specified dataset creates a part of the nn with specified
parameters, optionally in --dir.

For each dataset it creates one nn.

By default it creates a shared part of the nn, which can be used with
the specified dataset. If --shared=SHARED option is present, it uses
the previously created shared part to create a specific part of the nn,
that can be used with the shared part and the specified dataset.]])

   op:option{
      "--use-shared",
      dest   = "use_shared",
      help   = "a previously created shared part",
   }

   op:option{
      "--dir",
      default = "./",
      dest    = "dir",
      help    = "a directory to save the created nn in",
   }

   op:option{
      "--layers",
      default = "",
      dest    = "layers",
      help    = "comma separated hidden layer sizes",
   }

   op:option{
      "--act",
      default = "ReLU",
      dest    = "act",
      help    = "activation function: ReLU | ELU | Tanh | Sigmoid",
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
      default = 0,
      dest    = "dropconnect",
      help    = "use dropconnect with specified probability",
   }

   local opts, args = op:parse()

   if #args == 0 then
      op:fail("At least one dataset must be specified to create nn(s)!")
   elseif opts.use_shared and not paths.filep(opts.use_shared) then
      op:fail("The previously created shared part '" .. opts.use_shared
         .. "' does not exist")
   end

   opts.dropout     = tonumber(opts.dropout)
   opts.dropconnect = tonumber(opts.dropconnect)
   opts.layers      = loadstring("return {" .. opts.layers .. "}")()

   return opts, args
end

local function create_net(opts, shared)
   local net = nn.Sequential()
   local bias = not opts.batchnorm

   if shared then
      net:add(nn.View(-1):setNumInputDims(2))
   end

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

      if i < #opts.layers or shared then
         net:add(nn[opts.act]())
      else
         net:add(nn.Sigmoid())
      end
   end

   return net
end

local function get_output_size(nn_path)
   local nn = torch.load(nn_path)

   for i = nn:size(), 1, -1 do
      local module = nn:get(i)
      if module.__typename == "nn.Linear"
         or module.__typename == "nn.LinearDropconnect" then
         return module.weight:size(1)
      end
   end
end

local opts, args = parse_args(_G.arg)

if opts.use_shared then
   local sh_output_size = get_output_size(opts.use_shared)
   assert(sh_output_size, "'" .. opts.use_shared .. "' does not contain layers?")
   table.insert(opts.layers, 1, sh_output_size)
   table.insert(opts.layers, "to_be_replaced_by_an_actual_size")
else
   table.insert(opts.layers, 1, "to_be_replaced_by_an_actual_size")
end

for _, path in ipairs(args) do
   if not paths.filep(path) then
      print("Dataset '" .. path .. "' does not exist!")
   else
      local ds = torch.load(path)

      if opts.use_shared then
         opts.layers[#opts.layers] = ds:get(1).target:nElement()
      else
         opts.layers[1] = ds:get(1).input:nElement()
      end

      local nn = create_net(opts, opts.use_shared == nil)
      print(nn)

      local filename = opts.use_shared and paths.basename(path) or "shared.t7"
      local fullpath = paths.concat(opts.dir, filename)

      local write = true
      if paths.filep(fullpath) then
         print("File '" .. fullpath .. "' already exists.")
         write = cmdio.check_useragrees("Overwrite")
      end

      if write then
         torch.save(fullpath, nn)
         print("Successfully created the printed nn and saved to '" .. fullpath .. "'")
      else
         print("Skipping...")
      end
   end
end
