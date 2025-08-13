# Commands

## Building

The tokenizer logic is now modularized in `tokenizer.c`/`tokenizer.h` and shared by both `gpt-oss` and `test_tokenizer`.

To build everything:

```bash
make clean
make gpt-oss
make test_tokenizer
```

Or, to build manually:

```bash
gcc -O3 -o gpt-oss gpt-oss.c tokenizer.c -lm
gcc -O3 -o test_tokenizer test_tokenizer.c tokenizer.c -lm
```

## Usage

Export the tokenizer:

```bash
python export_o200k_bin.py -o tokenizer.bin
```

Run the tokenizer test:

```bash
./test_tokenizer -t tokenizer.bin -i "Hello world"
```

Or run the Python test harness:

```bash
python test_tokenizer.py --bin ./test_tokenizer --tok ./tokenizer.bin --verbose --prompt prompts.txt
```

```bash
make gpt-oss
./gpt-oss gpt-oss-20b.bin -i "Hello" -n 64 -t tokenizer.bin
```

```bash
python export_model_bin.py ./gpt-oss-20b ./gpt-oss-20b.bin --verify
```
