#!/bin/csh
#number of threads:
foreach t ( 1 2 4 8 )
	foreach k ( 200 300 400 500 600 1000 2000 3000 4000 )
		g++ -DNUMT=$t -DNUMNODES=$k project_2.cpp -o prog -lm -fopenmp
		./prog
	end
end

#clear files
rm -f project_2.o prog
