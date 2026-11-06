# Contributing to GeneZip

Questions, bug reports and pull requests are welcome.

## Reporting a bug

Include the command you ran, the expected and observed behavior, and the
traceback. For model execution, include your GPU, Python, PyTorch and CUDA
versions, the model name, and the task configuration. Remove credentials
and private data from logs before sharing them.

## Proposing a change

For a substantial change, open an issue first to discuss the approach.
Keep each pull request focused, describe the behavior it changes, and include
the commands and results used to check it. Update the usage documentation
when a command or configuration changes.

Follow the existing Python style and keep shared defaults in `configs/`.
For training or evaluation changes, run a small check on the affected task;
the [TISP pipeline example](results/reproduce_release.md#quick-pipeline-check)
shows how to reduce a run's sample and step counts.
