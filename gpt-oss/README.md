# Commands

```bash
python export_o200k_bin.py -o tokenizer.bin
gcc -O3 -DTESTING -o test_tokenizer test_tokenizer.c -lm
python test_tokenizer.py --bin ./test_tokenizer --tok ./tokenizer.bin --verbose --prompt prompts.txt
```
