local tnt      = require 'torchnet.env'
local argcheck = require 'argcheck'

local EMMeter = torch.class('tnt.EMMeter', 'tnt.Meter', tnt)


EMMeter.__init = argcheck{
   doc = [[
<a name="EMMeter">
#### tnt.EMMeter(@ARGP)
@ARGT

The `tnt.EMMeter` measures the exact match ratio.

The `tnt.EMMeter` is designed to operate on `NxK` Tensors `output` and `target`,
where (1) the `output` contains model output scores for `N` examples and `K`
classes that ought to be higher when the model is more convinced that the
example should be positively labeled, and smaller when the model believes the
example should be negatively labeled (for instance, the output of a sigmoid
function); and (2) the `target` contains only values 0 (for negative examples)
and 1 (for positive examples).

The optional variable `threshold` defines how the `output` is interpreted:
   values >= `threshold` -> 1
   values <  `threshold` -> 0
]],
   {name="self", type="tnt.EMMeter"},
   {name="threshold", type="number", default=0.5},
   call = function(self, threshold)
      self.threshold = threshold
      self:reset()
   end
}

EMMeter.reset = argcheck{
   {name="self", type="tnt.EMMeter"},
   call = function(self)
      self.matched = 0
      self.n = 0
   end
}

EMMeter.add = argcheck{
   {name="self", type="tnt.EMMeter"},
   {name="output", type="torch.*Tensor"},
   {name="target", type="torch.*Tensor"},
   call = function(self, output, target)

      -- assertions on the input:
      target = target:squeeze()
      output = output:squeeze()
      if output:nDimension() == 1 then
         output = output:view(1, output:size(1))
      else
         assert(output:nDimension() == 2,
            'wrong output size (should be 1D or 2D with one column per class)'
         )
      end
      if target:nDimension() == 1 then
         target = target:view(1, target:size(1))
      else
         assert(target:nDimension() == 2,
            'wrong target size (should be 1D or 2D with one column per class)'
         )
      end
      assert(output:size(1) == target:size(1) and
             output:size(2) == target:size(2),
         'dimensions for output and target does not match'
      )
      assert(torch.eq(torch.eq(target, 0):add(torch.eq(target, 1)), 1):all(),
         'targets should be binary (0 or 1)'
      )

      self.matched = self.matched + output
         :ge(self.threshold)
         :eq(target:byte())
         :min(2)
         :sum()
      self.n = self.n + target:size(1)
   end
}

EMMeter.value = argcheck{
   {name="self", type="tnt.EMMeter"},
   call = function(self)
      assert(self.n > 0, 'add some outputs first!')
      return self.matched / self.n
   end
}
