local sig  = require 'signal'

local csv  = require 'csvigo'
local glb  = require 'globtopattern'
local img  = require 'image'
local lfs  = require 'lfs'
local tnt  = require 'torchnet'
local xlua = require 'xlua' -- opts parser, progress bar

require 'common'

local function parse_args(args)
   local op = xlua.OptionParser("dataset_creator.lua --csv=CSV --dir=DIR"
      .. "[--mean=MEAN --std=STD --dont-care=VAL --pattern=PTRN] OUTPUT_FILE")

   op:option{
      "--csv",
      dest   = "csv",
      help   = "a csv file",
   }

   op:option{
      "--dir",
      dest   = "dir",
      help   = "a directory containing images",
   }

   op:option{
      "--mean",
      dest    = "mean",
      help    = "a mean to subtract from inputs",
   }

   op:option{
      "--std",
      dest    = "std",
      help    = "a std to divide the inputs with",
   }

   op:option{
      "--dont-care",
      default = 2,
      dest    = "dont_care",
      help    = "a target value not cared about",
   }

   op:option{
      "--pattern",
      dest    = "pattern",
      help    = "a (simple) glob pattern to match",
      default = "*.png",
   }


   local opts, args = op:parse()

   if not opts.csv or not opts.dir then
      op:fail("A csv file and a input directory must both be specified!")
   elseif not paths.dirp(opts.dir) then
      op:fail("The input directory containing cutouts must exist!")
   elseif not paths.filep(opts.csv) then
      op:fail("The csv file must exist!")
   elseif #args ~= 1 then
      op:fail("Exactly 1 output file must be specified!")
   end

   opts.pattern = glb.globtopattern(opts.pattern)

   return opts, args
end

local function check_sizes(s1, s2)

   if s2 == nil then
      return s1
   end

   if #s1 ~= #s2 then
      return false
   end

   for i = 1, #s1 do
      if s1[i] ~= s2[i] then
         return false
      end
   end

   return s1
end

sig.signal(sig.SIGINT, sig.signal_handler)

local opts, args = parse_args(_G.arg)

local res_path = args[1]
local csv_data = csv.load{path = opts.csv, mode = "raw"}
local total = 0
local input_table = {}
local target_table = {}

local input_dims
local target_dims

xlua.progress(total, #csv_data)
for fn in lfs.dir(opts.dir) do

   if string.match(fn, opts.pattern) then
      total = total + 1

      local path  = paths.concat(opts.dir, fn)
      local input = img.load(path, 1)
      local target = torch.Tensor(csv_data[total])
      target = target[target:ne(opts.dont_care)]

      input_dims = check_sizes(input:size(), input_dims)
      target_dims = check_sizes(target:size(), target_dims)

      if input_dims == false then
         error("All inputs must have the same size!")
      end
      if target_dims == false then
         error("All targets must have the same size!")
      end

      table.insert(input_table, input)
      table.insert(target_table, target)
   end

   xlua.progress(total, #csv_data)

   if _G.interrupted then break end
end

if not _G.interrupted then

   if total == 0 then
      print("No files in '" .. opts.dir .. "' match the '" .. opts.pattern .."' pattern.")
      return
   elseif total ~= #csv_data then
      print("Only " .. total .. " file(s) matched the '" .. opts.pattern
         .. "' (out of " .. #csv_data .. " target(s) in the csv file.")
      return
   end

   local input  = tnt.utils.table.mergetensor(input_table)
   local target = tnt.utils.table.mergetensor(target_table)

   local mean = opts.mean and opts.mean or input:mean()
   local std  = opts.std and opts.std or input:std()

   local function tostring(x)
      return string.format("%.20f",x)
   end

   print("Normalizing with:")
   print("mean: " .. tostring(mean))
   print("std:  " .. tostring(std))

   input:csub(mean):div(std)

   print("New dataset's mean: " .. tostring(input:mean()))
   print("New dataset's std:  " .. tostring(input:std()))

   local dataset = tnt.ListDataset{
      list = torch.range(1, total):long(),
      load =
      function(idx)
         return {
            input = input[idx],
            target = target[idx],
         }
      end,
   }

   torch.save(res_path, dataset)

   print("Wrote a dataset '" .. res_path .. "' containing " .. total .. " entrie(s)!")
end
