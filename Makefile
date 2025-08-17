# use hipcc as default
# if hipcc isn't available, use g++
CC := $(shell command -v hipcc 2>/dev/null || echo g++)
CFLAGS = --std=c++17 -lm
ifneq ($(CC),g++)
CFLAGS += --offload-arch=gfx90a
endif

CPP_FILES = run.cpp tokenizer.cpp

# the most basic way of building that is most likely to work on most systems
.PHONY: run
run: $(CPP_FILES) tokenizer-bin
	$(CC) -g -O0 -o run $(CPP_FILES)

# useful for a debug build, can then e.g. analyze with valgrind, example:
# $ valgrind --leak-check=full ./run out/model.bin -n 3
rundebug: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -g -o run $(CPP_FILES)

# https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html
.PHONY: runfast
runfast: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -o run $(CPP_FILES)

# additionally compiles with OpenMP, allowing multithreaded runs
# make sure to also enable multiple threads when running, e.g.:
# OMP_NUM_THREADS=4 ./run out/model.bin
.PHONY: runomp
runomp: $(CPP_FILES) tokenizer-bin
	$(CC) $(CFLAGS) -O3 -fopenmp -march=native $(CPP_FILES) -o run

.PHONY: decode
decode: decode.cpp tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -O3 decode.cpp tokenizer.cpp -o decode

.PHONY: tokenizer-bin
tokenizer-bin: export_tokenizer_bin.py
	python3 export_tokenizer_bin.py -o tokenizer.bin

.PHONY: tokenizer-test
tokenizer-test: test_tokenizer.cpp tokenizer.cpp tokenizer-bin
	$(CC) $(CFLAGS) -DTESTING -O3 test_tokenizer.cpp tokenizer.cpp -o tokenizer-test

.PHONY: clean
clean:
	rm -f run decode tokenizer.bin tokenizer-test
