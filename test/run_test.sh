cd ..
rm tmp/*.error
make
./cnn
cd test
gnuplot plot_error.gpi
gvfs-open ../tmp/plot_error.pdf

