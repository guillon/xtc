# Model Explorer Export

XTC can export graphs to a JSON structure designed for graph visualization tools such as model-explorer.

## CLI

Use the `xtc-export-model-explorer` command:

```bash
xtc-export-model-explorer \
  --operator matmul \
  --dims 512 1024 128 \
  --dtype float32 \
  --name matmul \
  --out matmul.model_explorer.json
```

Supported operators:

- `matmul` with dims: `i j k`
- `relu` with dims: `n`
- `conv2d` with dims: `n h w f r s c`

## Output

The generated JSON contains:

- `nodes`: operation/tensor nodes with ids, names, types, attrs, and optional access maps
- `edges`: producer -> consumer dependencies
- graph metadata: `name`, `inputs`, `outputs`

Schema version is stored in `format` as `xtc.model_explorer.v1`.

