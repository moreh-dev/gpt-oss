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
	int hidden_dim;			// model dim
	// MLP Config
	int n_experts;				// number of experts
	int experts_per_token;		// num top-k
	int intermediate_dim;		// for ffn layers
	int n_layers;				// num hidden layers
	// Attention Config
	int head_dim;				// head dimension
	int n_attn_heads;			// number of query heads
	int n_kv_heads;				// number of key/value heads (can be < query heads because of MQA)
	// Misc.
	bool is_mxfp4;				// check if weights stored in MXFP4 format
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
	float *rms_out;					// (dim, )
	// classifier weights for the logits [unembedding.weight]
	float *out;						// (vocab_size, hidden_dim)
} TransformerWeights;

typedef struct {
	Config config;
	TransformerWeights weights;
	int fd;				// file descriptor for memory mapping
	float *data;		// memory mapped data pointer
	ssize_t file_size;	// size of the checkpoint file in bytes
} Transformer;

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
	w->rms_out = ptr;
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

	printf("%f\n", w->b_mlp2[1023]);
	printf("%f\n", w->b_mlp2[2048]);
	//for (int l = 0; l < n_layers; l++) {
	//	for (int h = 0; h < n_experts; h++) {
	//		printf("%f, ", w->b_router[l * n_experts + h]);
	//	}
	//	printf("\n");
	//}
}

void load_checkpoint(char *ckpt, Config *config, TransformerWeights *weights, int *fd, float **data, ssize_t *file_size) {
	FILE *file = fopen(ckpt, "rb");
	if (!file) { fprintf(stderr, "Couldn't open file %s\n", ckpt); exit(EXIT_FAILURE); }

	// read in the config header
	// load sizeof(Config) bytes into config
	if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
	// figure out the file size
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

int main(int argc, char **argv) {
	Transformer model;
	char *ckpt_path = "gpt-oss-7M.bin";
	build_transformer(&model, ckpt_path);
	return 0;
}
