local MultiLabelClassNLLCriterion, parent = torch.class('nn.MultiLabelClassNLLCriterion', 'nn.Criterion')

function MultiLabelClassNLLCriterion:__init()
   parent.__init(self)
   self.outputTensor = torch.Tensor(1)
end

function MultiLabelClassNLLCriterion:updateOutput(input, target)
   self.output = -input:dot(target) / target:size(1)
   return self.output
end

function MultiLabelClassNLLCriterion:updateGradInput(input, target)
   return self.gradInput
      :resizeAs(target)
      :copy(target)
      :mul(-1/target:size(1))
end
