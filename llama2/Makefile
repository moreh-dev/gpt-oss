# Compiler
CC = gcc

# Build targets for current Linux machine
.PHONY: run rundebug runfast test clean
test:
	$(CC) test.c -o test -lm
	./test

run: run.c
	$(CC) -O3 -o run run.c -lm

rundebug: run.c
	$(CC) -g -o run run.c -lm

runfast: run.c
	$(CC) -Ofast -o run run.c -lm


clean:
	rm -f run test
