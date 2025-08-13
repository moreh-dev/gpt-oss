# Commands

```bash
python export_o200k_bin.py -o tokenizer.bin
gcc -O3 -DTESTING -o token_test token_test.c -lm
python test_tokenizer.py --bin ./token_test --tok ./tokenizer.bin --verbose
```
