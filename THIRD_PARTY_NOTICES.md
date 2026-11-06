# Third-Party Notices

This repository includes code adapted from, or designed to interoperate with,
the following open-source projects. GeneZip-authored code is licensed under MIT.
The components below retain their original copyright and license terms.

## H-Net

- Source: https://github.com/goombalab/hnet
- License: [MIT](third_party/hnet/LICENSE)
- Use in this repository: `src/hnet` includes HNet-derived modeling code with
  GeneZip-specific modifications.

## Mamba

- Source: https://github.com/state-spaces/mamba
- License: [Apache-2.0](third_party/mamba/LICENSE)
- Use in this repository: selected HNet module implementations are based on
  Mamba building blocks and interfaces.

## FlashAttention

- Source: https://github.com/Dao-AILab/flash-attention
- License: [BSD-3-Clause](third_party/flash-attention/LICENSE)
- Use in this repository: optional CUDA attention kernels used through the
  Python package dependency.

## causal-conv1d

- Source: https://github.com/Dao-AILab/causal-conv1d
- License: [BSD-3-Clause](third_party/causal-conv1d/LICENSE)
- Use in this repository: optional CUDA convolution kernels used through the
  Python package dependency.

## Other Runtime Dependencies

Additional third-party packages are listed in `pyproject.toml` and resolved in
`uv.lock`. Users should consult each package's upstream license for the full
license text.
