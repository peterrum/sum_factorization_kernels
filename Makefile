CC=mpiCC
BFOLDER = build2
SFOLDER = src
IFOLDER = include/
FLAGS   = -std=c++14 -march=native -fopenmp -O3


all:
	@make test_dg_precomputed test_dg_precomputed_gprof test_dg_precomputed_scorep


test_dg_precomputed:
	$(CC) $(FLAGS) -o $(BFOLDER)/test_dg_precomputed $(SFOLDER)/test_dg_precomputed.cc -I $(IFOLDER)

test_dg_precomputed_gprof:
	$(CC) $(FLAGS) -pg -o $(BFOLDER)/test_dg_precomputed_gprof $(SFOLDER)/test_dg_precomputed.cc -I $(IFOLDER)

test_dg_precomputed_scorep:
	scorep-mpicc $(FLAGS) -o $(BFOLDER)/test_dg_precomputed_scorep $(SFOLDER)/test_dg_precomputed.cc -I $(IFOLDER)

clean:
	/bin/rm -f $(BFOLDER)/*
