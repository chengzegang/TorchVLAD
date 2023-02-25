# TorchVLAD

This is a simple implementation of VLAD in pure PyTorch (with kornia).

-----

## **Table of Contents**

- [Usage](#usage)
- [License](#license)

## Usage

In bash

```bash
python -m pytorch_vlad --train-dir /PATH/TO/ROOT/IMAGE/DIR 

```

or in python

```python
import torch
import pytorch_vlad as vlad

model = vlad.train(**kwargs)
index_df, index_df_path = vlad.index(**kwargs)
retr_indices, retr_df = vlad.retrieve(**kwargs)
```

## License

`TorchVLAD` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
