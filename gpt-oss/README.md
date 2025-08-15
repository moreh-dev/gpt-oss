# Commands
---
## Building
- Build 
---
```bash
make run
```

## Usage
- Export the tokenizer
```bash
python export_o200k_bin.py -o tokenizer.bin
```

- Export the binary weights.
```bash
# E.g., generate gpt-oss-7m
cd export_model_bin/gpt-oss-7m
python convert_state_dict.py -i /path/to/safetensors
python generate_bin.py -i /path/to/safetensors -o /path/to/bin -c /path/to/config.json
```

- Run the model
```bash
./run /path/to/model/bin -t 0.0 -z tokenizer.bin -i "Your prompt here"
```
