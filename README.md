# Tokenizer

## Export Tokenizer

```bash
python export_tokenizer_bin.py -o tokenizer.bin
```

## Build & Run Tokenizer Test (C++)

```bash
g++ -O3 -DTESTING -o test_tokenizer test_tokenizer.cpp tokenizer.cpp -lm
./test_tokenizer -t tokenizer.bin -i "Hello world"
# Expected output: 13225 2375
```

## Verify Compatibility with Tiktoken

```bash
python test_tokenizer.py \
  --bin ./test_tokenizer \
  --tok ./tokenizer.bin \
  --verbose \
  --prompt prompts.txt
```

### Example Results

```
PROMPT: 'ฉันรักทะเล'
  C  encoded: [97797, 6560, 151737, 37899, 17758]
  PY encoded: [97797, 6560, 151737, 37899, 17758]
  C  decoded: 'ฉันรักทะเล'
  PY decoded: 'ฉันรักทะเล'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: 'naïve façade — déjà vu'
  C  encoded: [1503, 9954, 737, 114665, 2733, 21229, 12005]
  PY encoded: [1503, 9954, 737, 114665, 2733, 21229, 12005]
  C  decoded: 'naïve façade — déjà vu'
  PY decoded: 'naïve façade — déjà vu'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: '🍣 sushi and 🍜 ramen'
  C  encoded: [102415, 96, 85535, 326, 197348, 250, 90938]
  PY encoded: [102415, 96, 85535, 326, 197348, 250, 90938]
  C  decoded: '🍣 sushi and 🍜 ramen'
  PY decoded: '🍣 sushi and 🍜 ramen'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: 'email: test@example.com'
  C  encoded: [4261, 25, 1746, 81309, 1136]
  PY encoded: [4261, 25, 1746, 81309, 1136]
  C  decoded: 'email: test@example.com'
  PY decoded: 'email: test@example.com'
  [ENCODE MATCH] [DECODE MATCH]
------------------------------------------------------------
PROMPT: 'newlines:'
  C  encoded: [1389, 10105, 25]
  PY encoded: [1389, 10105, 25]
  C  decoded: 'newlines:'
  PY decoded: 'newlines:'
  [ENCODE MATCH] [DECODE MATCH]
```
