# Compiler
CC = gcc

# Build targets for current Linux machine
.PHONY: gpt-oss gpt-oss-debug gpt-oss-fast clean

gpt-oss: gpt-oss.c
	$(CC) -O3 -o gpt-oss gpt-oss.c -lm

gpt-oss-debug: gpt-oss.c
	$(CC) -g -o gpt-oss gpt-oss.c -lm

gpt-oss-fast: gpt-oss.c
	$(CC) -Ofast -o gpt-oss gpt-oss.c -lm


clean:
	rm -f gpt-oss
