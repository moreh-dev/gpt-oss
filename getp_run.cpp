// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"

#ifndef GETP_RUN
#define GETP_RUN

void warm_up(Transformer* transformer, Tokenizer* tokenizer, Requests* requests) {

}

long long simple_getp_generate(Transformer* transformer, Tokenizer* tokenizer, Sampler *sampler, const char* input_seq, char* output_seq, int steps)
{
    const char *empty_prompt = "";
    if (input_seq == NULL) { input_seq = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(input_seq)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    encode(tokenizer, input_seq, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    int pos_str = 0; // position in the output string
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

        // data-dependent terminating condition: the BOS (=1 or =2) token delimits sequences
        if (next == 1 || next == 2) { break; }

        // save the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        // uncomment 2 rows below to print the decoded token to stdout
        // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        // fflush(stdout);
        if (piece != NULL) {
            // see safe_printf(char*) at run.cpp for detail
            for (int i = 0; piece[i] != '\0'; i++) {
                unsigned char byte_val = piece[i];
                if (!(isprint(byte_val) || isspace(byte_val)))
                    output_seq[pos_str++] = piece[i];
            }
        }
        token = next;
    }

    free(prompt_tokens);

    return pos - 1;
}

long long inference(Transformer* transformer, Tokenizer* tokenizer, Sampler *sampler, Requests* requests) {
    long long num_token_out = 0;
    for (int idx = 0; idx < requests->num_reqs; ++idx) {
        char* input_seq = get_str_req_ptr(requests, idx);
        char* output_seq = get_str_gen_ptr(requests, idx);
        num_token_out += simple_getp_generate(transformer, tokenizer, sampler, input_seq, output_seq, requests->max_seq_len);
    }
    return num_token_out;
}

#endif // GETP_RUN