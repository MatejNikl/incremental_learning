##### Modified version of the sample given in
##### http://www.guidolin.net/blog/files/2010/03/gnuplot


set macro

#####  Color Palette by Color Scheme Designer
#####  Palette URL: http://colorschemedesigner.com/#3K40zsOsOK-K-


   blue_000 = "#A9BDE6" # = rgb(169,189,230)
   blue_025 = "#7297E6" # = rgb(114,151,230)
   blue_050 = "#1D4599" # = rgb(29,69,153)
   blue_075 = "#2F3F60" # = rgb(47,63,96)
   blue_100 = "#031A49" # = rgb(3,26,73)

   green_000 = "#A6EBB5" # = rgb(166,235,181)
   green_025 = "#67EB84" # = rgb(103,235,132)
   green_050 = "#11AD34" # = rgb(17,173,52)
   green_075 = "#2F6C3D" # = rgb(47,108,61)
   green_100 = "#025214" # = rgb(2,82,20)

   red_000 = "#F9B7B0" # = rgb(249,183,176)
   red_025 = "#F97A6D" # = rgb(249,122,109)
   red_050 = "#E62B17" # = rgb(230,43,23)
   red_075 = "#8F463F" # = rgb(143,70,63)
   red_100 = "#6D0D03" # = rgb(109,13,3)

   brown_000 = "#F9E0B0" # = rgb(249,224,176)
   brown_025 = "#F9C96D" # = rgb(249,201,109)
   brown_050 = "#E69F17" # = rgb(230,159,23)
   brown_075 = "#8F743F" # = rgb(143,116,63)
   brown_100 = "#6D4903" # = rgb(109,73,3)

   grid_color = "#d5e0c9"
   text_color = "#222222"

   my_font = "SVBasic Manual, 12"
   my_export_sz = "1024,768"

   my_line_width = "2"
   my_axis_width = "1"
   my_ps = "1"
   my_font_size = "14"

# must convert font fo svg and ps
# set term svg  size @my_export_sz fname my_font fsize my_font_size enhanced dynamic rounded
# set term png  size @my_export_sz large font my_font
# set term jpeg size @my_export_sz large font my_font
# set term wxt enhanced font my_font

set style data linespoints
set style function lines
set pointsize my_ps

set style line 1  linecolor rgbcolor blue_050  linewidth @my_line_width pt 7
set style line 2  linecolor rgbcolor green_050 linewidth @my_line_width pt 7
set style line 3  linecolor rgbcolor red_050   linewidth @my_line_width pt 7
set style line 4  linecolor rgbcolor brown_050 linewidth @my_line_width pt 7
set style line 5  linecolor rgbcolor blue_025  linewidth @my_line_width pt 5
set style line 6  linecolor rgbcolor green_025 linewidth @my_line_width pt 5
set style line 7  linecolor rgbcolor red_025   linewidth @my_line_width pt 5
set style line 8  linecolor rgbcolor brown_025 linewidth @my_line_width pt 5
set style line 9  linecolor rgbcolor blue_075  linewidth @my_line_width pt 9
set style line 10 linecolor rgbcolor green_075 linewidth @my_line_width pt 9
set style line 11 linecolor rgbcolor red_075   linewidth @my_line_width pt 9
set style line 12 linecolor rgbcolor brown_075 linewidth @my_line_width pt 9
set style line 13 linecolor rgbcolor blue_100  linewidth @my_line_width pt 13
set style line 14 linecolor rgbcolor green_100 linewidth @my_line_width pt 13
set style line 15 linecolor rgbcolor red_100   linewidth @my_line_width pt 13
set style line 16 linecolor rgbcolor brown_100 linewidth @my_line_width pt 13
set style line 17 linecolor rgbcolor "#224499" linewidth @my_line_width pt 11

## plot 1,2,3,4,5,6,7,8,9
set style increment user
set style arrow 1 filled

## used for bar chart borders
## set style fill solid 0.5

set size noratio
set samples 300

set border 31 lw @my_axis_width lc rgb text_color

# set title  "Comparison of different temperatures T"
set xlabel "z"
set ylabel "q"

set terminal postscript eps enhanced color
set output "softmax_example.eps"

plot  '-' using 2:xtic(1) title 'T = 1'  ,  '-' using 2:xtic(1) title 'T = 2'  ,  '-' using 2:xtic(1) title 'T = 5'  ,  '-' using 2:xtic(1) title 'T = 10'  ,  '-' using 2:xtic(1) title 'T = 20' 

2 0.000344403
4 0.0024924
0.1 5.36729e-05
1 0.000129063
10 0.996973
-2 7.11699e-06
-5 0
e
2 0.0168349
4 0.0457156
0.1 0.0065254
1 0.0102209
10 0.917889
-2 0.00229469
-5 0.000519139
e
2 0.103698
4 0.154696
0.1 0.070917
1 0.0849015
10 0.513608
-2 0.0465993
-5 0.0255803
e
2 0.136135
4 0.166275
0.1 0.112578
1 0.12318
10 0.302974
-2 0.0912543
-5 0.0676035
e
2 0.143246
4 0.158311
0.1 0.130264
1 0.136259
10 0.213697
-2 0.11728
-5 0.100944
e

