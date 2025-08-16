# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = hipcc
CFLAGS = --offload-arch=gfx90a -lm

CPP_FILES = run.cpp tokenizer.cpp

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: $(CPP_FILES)
	$(CC) -g -O0 -o run $(CPP_FILES)

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: $(CPP_FILES)
	$(CC) $(CFLAGS) --std=c++17 -g -o run $(CPP_FILES)

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
.PHONY: runfast
runfast: $(CPP_FILES)
	$(CC) $(CFLAGS) --std=c++17 -O3 -o run $(CPP_FILES)

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: $(CPP_FILES)
	$(CC) $(CFLAGS) --std=c++17 -O3 -fopenmp -march=native $(CPP_FILES) -o run

.PHONY: win64
win64: $(CPP_FILES)
	x86_64-w64-mingw32-gcc $(CFLAGS) -O3 -D_WIN32 -o run.exe -I. $(CPP_FILES)

# compiles with gnu99 standard flags for amazon linux, coreos, etc. compatibility
.PHONY: rungnu
rungnu: $(CPP_FILES)
	$(CC) $(CFLAGS) -O3 -std=gnu11 -o run $(CPP_FILES)

.PHONY: runompgnu
runompgnu: $(CPP_FILES)
	$(CC) $(CFLAGS) -O3 -fopenmp -std=gnu11 $(CPP_FILES) -o run

# run all tests
.PHONY: test
test:
	pytest

# run only tests for run.cpp C implementation (is a bit faster if only C code changed)
.PHONY: testc
testc:
	pytest -k runc

# run the C tests, without touching pytest / python
# to increase verbosity level run e.g. as `make testcc VERBOSITY=1`
VERBOSITY ?= 0
.PHONY: testcc
testcc:
	$(CC) $(CFLAGS) -DVERBOSITY=$(VERBOSITY) -O3 -o testc test.cpp
	./testc

.PHONY: clean
clean:
	rm -f run
