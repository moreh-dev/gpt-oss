// gpt_oss_main.c — CLI + tokenizer glue (builds/uses core)

#include "tokenizer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// bring in core
#include "gpt_oss_core.c"

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

static void usage(void) {
  fprintf(stderr, "Usage:\n");
  fprintf(stderr, "  gpt-oss <checkpoint.bin> --verify\n");
  fprintf(stderr, "  gpt-oss <checkpoint.bin> -i \"prompt\" -n <steps> [-t "
                  "tokenizer.bin]\n");
  exit(1);
}

int main(int argc, char **argv) {
  if (argc < 2)
    usage();
  const char *ckpt = argv[1], *prompt = "Hello", *tokpath = "tokenizer.bin";
  int steps = 64, verify_only = 0;

  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--verify"))
      verify_only = 1;
    else if (!strcmp(argv[i], "-i") && i + 1 < argc)
      prompt = argv[++i];
    else if (!strcmp(argv[i], "-n") && i + 1 < argc)
      steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-t") && i + 1 < argc)
      tokpath = argv[++i];
    else {
      fprintf(stderr, "Unknown arg: %s\n", argv[i]);
      usage();
    }
  }

  GPTOSSModel model;
  gptoss_build(&model, ckpt, /*verbose=*/1);
  if (verify_only) {
    fprintf(stderr, "[OK] verification finished.\n");
    gptoss_free(&model);
    return 0;
  }

  if (steps <= 0 || steps > model.cfg.seq_len)
    steps = model.cfg.seq_len;

  // tokenizer
  Tokenizer tokenizer;
  read_tokenizer(&tokenizer, tokpath, model.cfg.vocab_size);

  // preferential EOS picking
  const char *TOK_START = "<|start|>", *TOK_END = "<|end|>",
             *TOK_EOT = "<|endoftext|>", *TOK_RET = "<|return|>";
  int BOS = find_token_id(&tokenizer, TOK_START, (int)strlen(TOK_START)),
      EOS = -1, tmp;
  if ((tmp = find_token_id(&tokenizer, TOK_END, (int)strlen(TOK_END))) >= 0)
    EOS = tmp;
  if (EOS < 0 &&
      (tmp = find_token_id(&tokenizer, TOK_EOT, (int)strlen(TOK_EOT))) >= 0)
    EOS = tmp;
  if (EOS < 0 &&
      (tmp = find_token_id(&tokenizer, TOK_RET, (int)strlen(TOK_RET))) >= 0)
    EOS = tmp;

  // encode
  int *tokens = (int *)malloc(sizeof(int) * model.cfg.seq_len), ntok = 0;
  encode(&tokenizer, prompt, BOS, -1, tokens, &ntok, model.cfg.seq_len);
  if (ntok < 1) {
    fprintf(stderr, "empty prompt after encoding\n");
    return 1;
  }

  // warm-up through prompt context
  int pos = 0, tok = tokens[0];
  for (; pos < ntok - 1; pos++) {
    (void)gptoss_forward(&model, tok, pos);
    tok = tokens[pos + 1];
  }

  // generate
  Sampler samp = {.vocab_size = model.cfg.vocab_size};
  int to_gen = steps;
  if (to_gen > model.cfg.seq_len - ntok)
    to_gen = model.cfg.seq_len - ntok;
  for (int g = 0; g < to_gen; ++g) {
    float *logits = gptoss_forward(&model, tok, pos);
    int next = sample_next(&samp, logits);
    if (EOS >= 0 && next == EOS)
      break;
    char *piece = decode_piece(&tokenizer, tok, next);
    safe_printf(piece);
    fflush(stdout);
    tok = next;
    pos++;
  }
  printf("\n");

  free(tokens);
  free_tokenizer(&tokenizer);
  gptoss_free(&model);
  return 0;
}
