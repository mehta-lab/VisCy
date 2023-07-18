# Command-line interface

Access CLI help message by:

```sh
viscy --help
```

## Exporting models to ONNX

Current implementation will export a checkpoint to ONNX IR version 9
and OP set version 18 with:

```sh
viscy export -c config.yaml
```

Use argument `export_path` to configure where the output is stored.

### Notes

* For CPU sharing reasons, running an ONNX model
requires an exclusive node on HPC OR a non-distributed system (e.g. a PC).

* Models must be located in a lighting training logs directory
with a valid `config.yaml` in order to be initialized.
This can be "hacked" by locating the config in a directory
called `checkpoints` beneath a valid config's directory.
