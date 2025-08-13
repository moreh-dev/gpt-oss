// gpt_oss_main.c
// CLI for verification and greedy inference using gpt_oss_core.c
//  ./gpt-oss gpt-oss-20b.bin -i "Hello" -n 64 -t tokenizer.bin

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"

// Pull the core in as a second translation unit via include.
// (Keeps deliverable to exactly two files as you requested.)
#include "gpt_oss_core.c"

// ------------------------------ sampling

typedef struct {
  int vocab_size;
} Sampler;

static inline int sample_argmax(const float *p, int n) {
  int mi = 0;
  float mv = p[0];
  for (int i = 1; i < n; i++)
    if (p[i] > mv) {
      mv = p[i];
      mi = i;
    }
  return mi;
}

static inline int sample_next(Sampler *s, float *logits) {
  (void)s;
  return sample_argmax(logits, s->vocab_size);
}

// ------------------------------ usage

static void usage(void) {
  fprintf(
      stderr,
      "Usage:\n"
      "  gpt-oss <checkpoint.bin> -i \"prompt\" -n <steps> [-t tokenizer.bin]\n"
      "  gpt-oss <checkpoint.bin> --verify [-v]\n");
  exit(1);
}

// ------------------------------ main

int main(int argc, char **argv) {
  if (argc < 2)
    usage();

  const char *ckpt = argv[1];
  const char *prompt = "Hello";
  const char *tokpath = "tokenizer.bin";
  int steps = 64;
  int verify_only = 0;
  int verify_verbose = 0;

  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "-i") && i + 1 < argc) {
      prompt = argv[++i];
    } else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
      steps = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
      tokpath = argv[++i];
    } else if (!strcmp(argv[i], "--verify")) {
      verify_only = 1;
    } else if (!strcmp(argv[i], "-v")) {
      verify_verbose = 1;
    } else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      usage();
    }
  }

  GPTOSSModel model;
  gptoss_build(&model, ckpt, verify_verbose);

  if (verify_only) {
    fprintf(stderr, "[OK] verification finished.\n");
    gptoss_free(&model);
    return 0;
  }

  // --- inference path ---
  if (steps <= 0 || steps > model.cfg.seq_len)
    steps = model.cfg.seq_len;

  Tokenizer tokenizer;
  read_tokenizer(&tokenizer, tokpath, model.cfg.vocab_size);

  const char *TOK_START = "<|start|>";
  const char *TOK_END = "<|end|>";
  const char *TOK_EOT = "<|endoftext|>";
  const char *TOK_RETURN = "<|return|>";

  int BOS = find_token_id(&tokenizer, TOK_START, (int)strlen(TOK_START));
  int EOS = -1;
  int tmp;

  if ((tmp = find_token_id(&tokenizer, TOK_END, (int)strlen(TOK_END))) >= 0)
    EOS = tmp;
  if (EOS < 0 &&
      (tmp = find_token_id(&tokenizer, TOK_EOT, (int)strlen(TOK_EOT))) >= 0)
    EOS = tmp;
  if (EOS < 0 && (tmp = find_token_id(&tokenizer, TOK_RETURN,
                                      (int)strlen(TOK_RETURN))) >= 0)
    EOS = tmp;

  int *tokens = (int *)malloc(sizeof(int) * model.cfg.seq_len);
  int ntok = 0;
  encode(&tokenizer, prompt, BOS, -1, tokens, &ntok, model.cfg.seq_len);
  if (ntok < 1) {
    fprintf(stderr, "empty prompt after encoding\n");
    return 1;
  }

  Sampler sampler = {.vocab_size = model.cfg.vocab_size};

  // Warm-up: run the prompt through (do not echo prompt)
  int pos = 0;
  int token = tokens[0];
  for (; pos < ntok - 1; pos++) {
    (void)gptoss_forward(&model, token, pos);
    token = tokens[pos + 1];
  }

  // Generate
  int to_generate = steps;
  if (to_generate > model.cfg.seq_len - ntok)
    to_generate = model.cfg.seq_len - ntok;

  int generated = 0;
  while (generated < to_generate) {
    float *logits = gptoss_forward(&model, token, pos);
    int next = sample_next(&sampler, logits);
    if (EOS >= 0 && next == EOS)
      break;

    char *piece = decode_piece(&tokenizer, token, next);
    safe_printf(piece);
    fflush(stdout);

    token = next;
    pos++;
    generated++;
  }
  printf("\n");

  free(tokens);
  free_tokenizer(&tokenizer);
  gptoss_free(&model);
  return 0;
}
