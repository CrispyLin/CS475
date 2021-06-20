#!/bin/csh
#number of threads:
foreach t ( 1 4 )
	g++ -DNUMT=$t project_0.cpp -o prog -lm -fopenmp
	./prog
end

#clear files
rm -f project_0.o prog
