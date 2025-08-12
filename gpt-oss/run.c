/* Inference for gpt-oss model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// ----------------------------------------------------------------------------

typedef struct {
	// Model Config
	int vocab_size;				// vocabulary size
	int hidden_dim;				// model dim
	// MLP Config
	int n_experts;				// number of experts
	int experts_per_token;		// num top-k
	int intermediate_dim;		// for ffn layers
	int n_layers;				// num hidden layers
	// Attention Config
	int head_dim;				// head dimension
	int n_attn_heads;			// number of query heads
	int n_kv_heads;				// number of key/value heads (can be < query heads because of MQA)
	int seq_len;				// max sequence length
} Config;

typedef struct {
	// token_embedding_table - embedding.weight
	float *token_embedding_table;	// (vocab_size, hidden_dim)
	// weights for rmsnorms
	float *rms_attn_w;				// (n_layers, hidden_dim) [attn.norm.scale]
	float *rms_ffn_w;				// (n_layers, hidden_dim) [mlp.norm.scale]
	// weights for attention [attn.qkv.weight & attn.qkv.bias]
	float *w_qkv;					// (n_layers, hidden_dim, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads) where
									// w_q (hidden_dim, head_dim * n_attn_heads)
									// w_k (hidden_dim, head_dim * n_kv_heads)
									// w_v (hidden_dim, head_dim * n_kv_heads)
	float *w_o;						// (n_layers, head_dim * n_attn_heads, hidden_dim)
	float *b_qkv;					// (n_layers, head_dim * n_attn_heads + 2 * head_dim * n_kv_heads)
									// (head_dim * n_attn_heads)
									// (head_dim * n_kv_heads)
									// (head_dim * n_kv_heads)
	float *b_o;						// (n_layers, hidden_dim)
	float *attn_sinks;				// (n_layers, n_attn_heads)
	// weights for router [mlp.gate.weight & mlp.gate.bias]
	float *w_router;				// (n_layers, hidden_dim, n_experts)
	float *b_router;				// (n_layers, n_experts)
	// weights for MoE [mlp.mlp1_weight & mlp.mlp1_bias & mlp.mlp2_weight & mlp.mlp2_bias]
	// NOTE: gate_up projects from hidden_dim to intermediate_dim, the shape is kinda reverted because the original code use einsum to reduce over hidden_dim
	float *w_mlp1;					// gate_up_proj (n_layers, n_experts, 2 * intermediate_dim, hidden_dim)
	float *w_mlp2;					// down_proj (n_layers, n_experts, hidden_dim, intermediate_dim)
	float *b_mlp1;					// gate_up proj (n_layers, n_experts, 2 * intermediate_dim)
	float *b_mlp2;					// down_proj (n_layers, n_experts, hidden_dim)
	// final norm [norm.scale]
	float *rms_out_w;				// (hidden_dim, )
	// classifier weights for the logits [unembedding.weight]
	float *out;						// (vocab_size, hidden_dim)
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (hidden_dim,)
    float *xb; // same, but inside a residual branch (hidden_dim,)
    float *xb2; // an additional buffer just for convenience (hidden_dim,)
	float *router_score; // router score (num_experts, )
	float *topk_v;
	int *topk_i;
    float *hb; // buffer for hidden dimension in the ffn (intermediate_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (intermediate_dim,)
    float *q; // query (2*hidden_dim,)
    float *k; // key (hidden_dim,)
    float *v; // value (hidden_dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, kv_dim)
    float* value_cache; // (layer, seq_len, kv_dim)
} RunState;

typedef struct {
	Config config;
	TransformerWeights weights;
	RunState state;		// buffers for the "wave" of activations in the forward pass
	int fd;				// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	ssize_t file_size;	// size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->hidden_dim * p->n_kv_heads) / p->n_attn_heads;
    s->x = calloc(p->hidden_dim, sizeof(float));
    s->xb = calloc(p->hidden_dim, sizeof(float));
    s->xb2 = calloc(p->hidden_dim, sizeof(float));
    s->hb = calloc(p->intermediate_dim, sizeof(float));
    s->hb2 = calloc(p->intermediate_dim, sizeof(float));
    s->q = calloc(p->hidden_dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = calloc(p->n_attn_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *cfg, float *ptr) {
	int head_dim = cfg->head_dim;
	int n_layers = cfg->n_layers;
	int n_experts = cfg->n_experts;
	w->token_embedding_table = ptr;
	ptr += cfg->vocab_size * cfg->hidden_dim;
	w->out = ptr; // unembedding
	ptr += cfg->vocab_size * cfg->hidden_dim;
	w->rms_attn_w = ptr;
	ptr += n_layers * cfg->hidden_dim;
	w->rms_ffn_w = ptr;
	ptr += n_layers * cfg->hidden_dim;
	w->rms_out_w = ptr;
	ptr += cfg->hidden_dim;
	// hey it's qkvqkv, not qqkkvv
	w->w_qkv = ptr;
	ptr += n_layers * cfg->hidden_dim * (head_dim * cfg->n_attn_heads +
											  2 * head_dim * cfg->n_kv_heads);
	w->b_qkv = ptr;
	ptr += n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
	w->w_o = ptr;
	ptr += n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
	w->b_o = ptr;
	ptr += n_layers * cfg->hidden_dim;
	w->attn_sinks = ptr;
	ptr += n_layers * cfg->n_attn_heads;
	w->w_router = ptr;
	ptr += n_layers * cfg->hidden_dim * n_experts;
	w->b_router = ptr;
	ptr += n_layers * n_experts;
	// hey it's gate_upgate_up, not gategateupup
	w->w_mlp1 = ptr;
	ptr += n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
	w->b_mlp1 = ptr;
	ptr += n_layers * 2 * cfg->intermediate_dim * cfg->hidden_dim;
	w->w_mlp2 = ptr;
	ptr += n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
	w->b_mlp2 = ptr;
	ptr += n_layers * cfg->hidden_dim * cfg->intermediate_dim;
}

void load_checkpoint(char *ckpt, Config *config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
	FILE *file = fopen(ckpt, "rb");
	if (!file) { fprintf(stderr, "Couldn't open file %s\n", ckpt); exit(EXIT_FAILURE); }

	// read in the config header
	// load sizeof(Config) bytes into config
	if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
	// figure out the file size
	printf("max_seq_len: %d", config->seq_len);
	fseek(file, 0, SEEK_END); // move file pointer to end of file
	*file_size = ftell(file); // get the file size, in bytes
	fclose(file);
	// memory map the Transformer weights into the data pointer
	*fd = open(ckpt, O_RDONLY); // open in read only mode
	if (*fd == -1) { fprintf(stderr, "open failed\n"); exit(EXIT_FAILURE); }
	*data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
	if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
	float* weights_ptr = *data + sizeof(Config)/sizeof(float);
	memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer *t, char* ckpt_path) {
	// read in the Config and the Weights from the checkpoint
	load_checkpoint(ckpt_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void transpose(float* x_T, float *x, size_t n_row, size_t n_col) {
	for (size_t r = 0; r < n_row; r++) {
		for (size_t c = 0; c < n_col; c++) {
			x_T[c * n_row + r] = x[r * n_col + c];
		}
	}
}

// Pair struct to store score and original index
typedef struct {
	float value;
	int index;
} Pair;

// Comparator for descending sort (largest value first)
int compare_desc(const void* a, const void* b) {
	float diff = ((Pair*)b)->value - ((Pair*)a)->value;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

// topk function: returns top-k values and their indices
void topk(float* topk_values, int* topk_indices, float* router_score, int num_experts, int experts_per_token) {
    // Allocate temp array to store (value, index) pairs
    Pair* pairs = (Pair*)malloc(num_experts * sizeof(Pair));
    for (int i = 0; i < num_experts; ++i) {
        pairs[i].value = router_score[i];
        pairs[i].index = i;
    }

    // Sort in descending order of value
    qsort(pairs, num_experts, sizeof(Pair), compare_desc);

    // Fill output arrays
    for (int i = 0; i < experts_per_token; ++i) {
        topk_values[i] = pairs[i].value;
        topk_indices[i] = pairs[i].index;
    }
    free(pairs);
}

float* forward(Transformer *transformer, int token, int pos) {
	Config *p = &transformer->config;
	TransformerWeights *w = &transformer->weights;
	RunState *s = &transformer->state;

	float *x = s->x;
	int head_dim = p->head_dim;
	int hidden_dim = p->hidden_dim;
    int kv_dim =  p->head_dim * p->n_kv_heads;
    int kv_mul = p->n_attn_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
	int intermediate_dim = p->intermediate_dim;
	int n_experts = p->n_experts;

	// copy the token embedding into x
	float *content_row = w->token_embedding_table + token * hidden_dim;
	memcpy(x, content_row, hidden_dim*sizeof(*x));

	// forward all the layers
	for (unsigned long long l = 0; l < p->n_layers; l++) {
		rmsnorm(s->xb, x, w->rms_attn_w + l*hidden_dim, hidden_dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;
		// Separate q, k, v
		float *w_qkv = w->w_qkv + l * (head_dim * p->n_attn_heads + 2 * head_dim * p->n_kv_heads);
		float *w_q = w_qkv;
		float *w_k = w_qkv + hidden_dim * (head_dim * p->n_attn_heads);
		float *w_v = w_qkv + hidden_dim * (head_dim * p->n_attn_heads) + hidden_dim * (head_dim * p->n_kv_heads);

        // qkv matmuls for this position
        matmul(s->q, s->xb, w_q, hidden_dim, head_dim * p->n_attn_heads);
        matmul(s->k, s->xb, w_k, hidden_dim, head_dim * p->n_kv_heads);
        matmul(s->v, s->xb, w_v, hidden_dim, head_dim * p->n_kv_heads);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < head_dim * p->n_attn_heads; i+=2) {
            int pair_idx_in_head = i % head_dim;
            float freq = 1.0f / powf(10000.0f, pair_idx_in_head / (float)head_dim);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
		
		// TODO: integrate bias and sink
        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_attn_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_dim;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_dim);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_dim;
            memset(xb, 0, head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_dim;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_dim; i++) {
                    xb[i] += a * v[i];
                }
            }
        }
        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->w_o + l*(head_dim * p->n_attn_heads)*hidden_dim, head_dim * p->n_attn_heads, hidden_dim);

        // residual connection back into x
        for (int i = 0; i < hidden_dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_w + l * hidden_dim, hidden_dim);

		// TODO: MoE
		// Compute router_score
		matmul(s->router_score, s->xb, w->w_router + l*hidden_dim*n_experts, hidden_dim, p->n_experts); // s->router_score now stores router_score (num_experts, )
		// Select top-k experts
		topk(s->topk_v, s->topk_i, s->router_score, n_experts, p->experts_per_token);
		// Normalize selected experts using softmax or sigmoid
		softmax(s->topk_v, p->experts_per_token); // expert

		// Route the tokens to their corresponding top-k experts
		float *e_agg = (float*)calloc(hidden_dim, sizeof(float));
		if (!e_agg) { fprintf(stderr, "Failed init e_agg\n"); exit(EXIT_FAILURE); }
		for (int e = 0; e < n_experts; e++) {
			int in_topk = 0;
			// Check if expert i is in top-k experts
			for (int idx = 0; idx < p->experts_per_token; idx++) {
				if (s->topk_i[idx] == e) {
					in_topk = 1;
					float expert_w = s->topk_v[idx];
					break;
				}
			}

			if (in_topk) {
				float *block_ptr = w->w_mlp1 + ((l * n_experts + e) * 2 * p->intermediate_dim + hidden_dim);
				float *w_gate = block_ptr;
				float *w_up = block_ptr + p->intermediate_dim * hidden_dim;
				float *w_gate_T, *w_up_T, *gate, *up, *gate_up;
				transpose(w_gate_T, w_gate, p->intermediate_dim, hidden_dim);
				transpose(w_up_T, w_up, p->intermediate_dim, hidden_dim);
				matmul(gate, s->xb, w_gate_T, hidden_dim, p->intermediate_dim); // (intermediate_dim, )
				matmul(up, s->xb, w_up_T, hidden_dim, p->intermediate_dim); // (intermediate_dim, )

				// SwiGLU non-linearity
				for (int i = 0; i < p->intermediate_dim; i++) {
					float val = gate[i];
					// silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
					val *= (1.0f / (1.0f + expf(-val)));
					// elementwise multiply with w3(x)
					val *= up[i];
					gate_up[i] = val;
				}
				
				// final matmul to get the output of the ffn
				matmul(s->xb, gate_up, w->w_mlp2 + , p->intermediate_dim, hidden_dim); // (hidden_dim, )

				// aggregate topk experts using weighted sum 
				for (int i = 0; i < hidden_dim; i++) {
					e_agg[i] += s->xb[i] * expert_w;
				}
			}
		}

		//
	}
	// final rmsnorm
	rmsnorm(x, x, w->rms_out_w, hidden_dim);

	// classifier into logits
	matmul(s->logits, x, w->out, hidden_dim, p->vocab_size);
	return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read 1\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read 2\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read 3\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read 4\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
	generate(&transformer, &tokenizer, &sampler, prompt, steps);

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
	return 0;
}

#endif /* TESTING */
