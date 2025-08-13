# Compiler
CC = gcc

# Build targets for current Linux machine
.PHONY: gpt-oss gpt-oss-debug gpt-oss-fast clean


gpt-oss: gpt_oss_main.c tokenizer.c tokenizer.h
	$(CC) -O3 -o gpt-oss gpt_oss_main.c tokenizer.c -lm


gpt-oss-debug: gpt_oss_main.c tokenizer.c tokenizer.h
	$(CC) -g -o gpt-oss gpt_oss_main.c tokenizer.c -lm


gpt-oss-fast: gpt_oss_main.c tokenizer.c tokenizer.h
	$(CC) -Ofast -o gpt-oss gpt_oss_main.c tokenizer.c -lm

# Build test_tokenizer
test_tokenizer: test_tokenizer.c tokenizer.c tokenizer.h
	$(CC) -O3 -o test_tokenizer test_tokenizer.c tokenizer.c -lm


clean:
	rm -f gpt-oss test_tokenizer
