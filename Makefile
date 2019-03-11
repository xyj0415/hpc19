CC=gcc-8
CXX=g++-8
CFLAGS= -fopenmp
CXXFLAGS= -std=c++11 -fopenmp -O3 -march=native
executable = $(basename $(wildcard *.c *.cpp))

all: $(executable)

.PHONY: all clean
clean:
	rm -f $(executable)