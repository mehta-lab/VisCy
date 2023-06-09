# Command-line interface

Access CLI help message by:

```sh
viscy --help
```

## Exporting models to ONNX

Current implementation will export a checkpoint to ONNX IR version 9
and OP set version 19 with:

```sh
viscy export -c config.yaml
```

### Notes

* For cpu sharing reasons, running an ONNX model
requires a dedicated node on HPC OR a non-distributed system
(for example a personal laptop or other device).

* Models must be located in a lighting training logs directory
with a valid `config.yaml` in order to be initialized.
This can be "hacked" by locating the config in a directory
called `checkpoints` beneath a valid config's directory.
