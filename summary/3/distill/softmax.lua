require 'nn'
require 'os'
require 'gnuplot'

os.setlocale('en_US.UTF-8')

T = {1, 2, 5, 10, 20}

v = torch.Tensor{2, 4, 0.1, 1, 10, -2, -5}
m = nn.SoftMax()

plot = {}
for _, T in ipairs(T) do
    plot[#plot+1] = {'T = ' .. T, v, m:forward(v:clone():div(T)):clone()}
end
gnuplot.plot(table.unpack(plot))

-- in .../torch/install/share/lua/5.1/gnuplot/gnuplot.lua in function gnulplot print hdr and data variables
