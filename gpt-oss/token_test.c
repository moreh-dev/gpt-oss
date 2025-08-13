// token_test.c - encode-only test harness for tokenizer.bin
// Builds on top of gpt-oss.c tokenizer implementation.
// Usage:
//   gcc -O3 -DTESTING -o token_test token_test.c -lm
//   ./token_test -t tokenizer.bin -i "Hello world" -r
//
// Prints a single line of space-separated token IDs that the C encoder
// produces for the prompt. If -r is given, also prints the decoded text.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TESTING 1
#include "gpt-oss.c" // pulls in tokenizer structs & encode/decode but NO main()

static int count_vocab_in_tokenizer_bin(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "open %s failed: %s\n", path, strerror(errno));
    exit(1);
  }
  int maxlen = 0;
  if (fread(&maxlen, sizeof(int), 1, f) != 1) {
    fprintf(stderr, "bad tokenizer header in %s\n", path);
    exit(1);
  }
  int count = 0;
  while (1) {
    float score;
    int len;
    if (fread(&score, sizeof(float), 1, f) != 1)
      break;
    if (fread(&len, sizeof(int), 1, f) != 1)
      break;
    if (fseek(f, len, SEEK_CUR) != 0)
      break;
    count++;
  }
  fclose(f);
  return count;
}

int main(int argc, char **argv) {
  const char *tokpath = "tokenizer.bin";
  const char *prompt = "Hello";
  int print_redecode = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
      tokpath = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      prompt = argv[++i];
      continue;
    }
    if (strcmp(argv[i], "-r") == 0) {
      print_redecode = 1;
      continue;
    }
    fprintf(stderr, "Unknown arg: %s\n", argv[i]);
    fprintf(stderr, "Usage: %s -t tokenizer.bin -i \"prompt\" [-r]\n", argv[0]);
    return 2;
  }

  int vocab_size = count_vocab_in_tokenizer_bin(tokpath);
  if (vocab_size <= 0) {
    fprintf(stderr, "Failed to count vocab in %s\n", tokpath);
    return 1;
  }

  Tokenizer t;
  read_tokenizer(&t, tokpath, vocab_size);

  int max_tokens = 8192;
  int *tokens = (int *)malloc(sizeof(int) * max_tokens);
  int ntok = 0;
  // NOTE: bos_id = -1, eos_id = -1 so we test *just* the BPE behavior.
  encode(&t, prompt, -1, -1, tokens, &ntok, max_tokens);

  // Print token ids
  for (int i = 0; i < ntok; i++) {
    if (i)
      printf(" ");
    printf("%d", tokens[i]);
  }
  printf("\n");

  if (print_redecode) {
    // Re-decode via transitions; use a BOS-like prev=-1 then walk
    // Using adjacent pairs (prev->curr) to get byte tokens right.
    // First print the textual join of decoded pieces.
    for (int i = 0; i < ntok; i++) {
      int prev = (i == 0) ? -1 : tokens[i - 1];
      char *piece = decode_piece(&t, prev, tokens[i]);
      safe_printf(piece);
    }
    printf("\n");
  }

  free(tokens);
  free_tokenizer(&t);
  return 0;
}
