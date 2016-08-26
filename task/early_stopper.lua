local argcheck = require 'argcheck'

local EarlyStopper = torch.class('EarlyStopper')

EarlyStopper.__init = argcheck{
   {name="self", type="EarlyStopper"},
   {name="try_epochs", type="number"},
   call =
      function(self, try_epochs)
         self.try_epochs = try_epochs
         EarlyStopper.reset(self)
      end
}

EarlyStopper.epoch = argcheck{
   {name="self", type="EarlyStopper"},
   {name="current_acc", type="number"},
   {name="net", type="nn.Sequential"},
   call =
      function(self, current_acc, net)
         self.is_reset = false

         if current_acc > self.best_acc then
            self.best_acc = current_acc
            self.best_net = net:clone():clearState()
            self.epochs_waited = 0
         else
            self.epochs_waited = self.epochs_waited + 1
         end

         return EarlyStopper.shouldStop(self)
      end
}

EarlyStopper.improved = argcheck{
   {name="self", type="EarlyStopper"},
   call =
      function(self)
         return not self.is_reset and self.epochs_waited == 0
      end
}

EarlyStopper.shouldStop = argcheck{
   {name="self", type="EarlyStopper"},
   call =
      function(self)
         return self.epochs_waited >= self.try_epochs
      end
}

EarlyStopper.resetEpochs = argcheck{
   {name="self", type="EarlyStopper"},
   call =
      function(self)
         self.epochs_waited = 0
         self.is_reset      = true
      end
}

EarlyStopper.reset = argcheck{
   {name="self", type="EarlyStopper"},
   call =
      function(self)
         EarlyStopper.resetEpochs(self)
         self.best_net      = nil
         self.best_acc      = -math.huge
      end
}

EarlyStopper.getBestNet = argcheck{
   {name="self", type="EarlyStopper"},
   call =
      function(self)
         return self.best_net
      end
}
