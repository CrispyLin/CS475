#!/bin/csh
#number of threads:
foreach t ( 1 2 4 8 )
	foreach k ( 1 10 100 1000 10000 100000 500000)
		g++ -DNUMT=$t -DNUMTRIALS=$k project_1.cpp -o prog -lm -fopenmp
		./prog
	end
end

#clear files
rm -f project_1.o prog
