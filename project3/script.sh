#!/bin/csh
#number of threads:
foreach t ( 4 )
	g++ -DNUMT=$t project_3.cpp -o prog -lm -fopenmp
	./prog
end

#clear files
rm -f project_3.o prog
