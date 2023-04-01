set terminal pngcairo enhanced font "arial,10" fontscale 1.0 size 1000, 800
set output 'contours.20.png'
set view map
set title "Isolikelihood contours (Topological trap)"
unset surface
set contour
#set cntrparam levels incr -120,50,0
#set cntrparam levels disc -40,-30,-20,-14,-10,-2,4,10,14
set cntrparam levels disc -30,-20,-10,-2,5,10,15

$data << EOD
0 0 0 2 2 0
EOD

splot 'cmake-build-debug/isocontours.dat' with lines, \
    'cmake-build-debug/gradient.dat' using 1:2:(0):3:4:(0) notitle w vector lc -1

#    $data using 1:2:3:4:5:6 w vector lc -1


