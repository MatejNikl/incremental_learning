--[[

   Regularization of Neural Networks using DropConnect
   Li Wan, Matthew Zeiler, Sixin Zhang, Yann LeCun, Rob Fergus

   Dept. of Computer Science, Courant Institute of Mathematical Science, New York University

   Implemented by John-Alexander M. Assael (www.johnassael.com), 2015
   Further modified by Matej Nikl, 2016

]]--

local LinearDropconnect, parent = torch.class('nn.LinearDropconnect', 'nn.Linear')

function LinearDropconnect:__init(inputSize, outputSize, p, bias)

   local bias = ((bias == nil) and true) or bias
   self.train = true

   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<LinearDropconnect> illegal percentage, must be 0 <= p < 1')
   end

   self.noiseWeight = torch.Tensor(outputSize, inputSize)
   if bias then self.noiseBias = torch.Tensor(outputSize) end

   parent.__init(self, inputSize, outputSize, bias)
end

function LinearDropconnect:noBias()
   self.noiseBias = nil
   return parent.noBias(self)
end

function LinearDropconnect:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         if self.bias then self.bias[i] = torch.uniform(-stdv, stdv) end
      end
   else
      self.weight:uniform(-stdv, stdv)
      if self.bias then self.bias:uniform(-stdv, stdv) end
   end

   self.noiseWeight:fill(1)
   if self.noiseBias then self.noiseBias:fill(1) end

   return self
end

function LinearDropconnect:updateOutput(input)

   -- Dropconnect
   if self.train then
      self.noiseWeight:bernoulli(1-self.p):cmul(self.weight)
      if self.noiseBias then
         self.noiseBias:bernoulli(1-self.p):cmul(self.bias)
      end
   end

   if input:dim() == 1 then
      self.output:resize(self.weight:size(1))
      if self.train then
         if self.noiseBias then
            self.output:copy(self.noiseBias)
         else
            self.output:zero()
         end
         self.output:addmv(1, self.noiseWeight, input)
      else
         if self.bias then
            self.output:copy(self.bias)
         else
            self.output:zero()
         end
         self.output:addmv(1, self.weight, input)
      end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.weight:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      if self.train then
         self.output:addmm(0, self.output, 1, input, self.noiseWeight:t())
         if self.noiseBias then
            self.output:addr(1, self.addBuffer, self.noiseBias)
         end
      else
         self.output:addmm(0, self.output, 1, input, self.weight:t())
         if self.bias then
            self.output:addr(1, self.addBuffer, self.bias)
         end
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearDropconnect:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         if self.train then
            self.gradInput:addmv(0, 1, self.noiseWeight:t(), gradOutput)
         else
            self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
         end
      elseif input:dim() == 2 then
         if self.train then
            self.gradInput:addmm(0, 1, gradOutput, self.noiseWeight)
         else
            self.gradInput:addmm(0, 1, gradOutput, self.weight)
         end
      end

      return self.gradInput
   end
end
