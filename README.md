# Maximal Mutual Information (MMI) Tagger

This is a minimalist PyTorch implementation of the label inducer in [1]. For the full codebase used in experiments, refer to the repository at [2].

### Requirement

The code is in Python 3.6 and uses PyTorch version `1.0.1.post2`. Tested with Geforce RTX 2080 Ti and CUDA version `10.1.105`.

### Data

You can get the universal treebank v2.0 at [3] (McDonald et al., 2013), which provides both fine-grained and coarse-grained labels.

### Running the code

```bash
python main.py example-model example.words --train --epochs 10 --num_labels 3
python main.py en45-model ${EN45PATH}/en.words --train --num_labels 45 --epochs 5 --cuda --clusters clusters.txt --pred pred.txt
```

Output logged in file `en45-model.log`

```bash
| epoch   1 | loss  -1.08 |   1.55 bits | acc  77.95 | vm  72.51 | time 0:02:38
| epoch   2 | loss  -1.55 |   2.23 bits | acc  79.17 | vm  73.25 | time 0:02:41
| epoch   3 | loss  -1.73 |   2.50 bits | acc  79.47 | vm  73.45 | time 0:02:42
| epoch   4 | loss  -1.86 |   2.69 bits | acc  79.37 | vm  73.44 | time 0:02:38
| epoch   5 | loss  -1.96 |   2.83 bits | acc  79.28 | vm  73.35 | time 0:02:41

Training time 0:13:23
=========================================================================================
| Best | acc 79.47 | vm 73.45
=========================================================================================
```

### References

[1] [Mutual Information Maximization for Simple and Accurate Part-Of-Speech Induction (Stratos, 2018)](https://www.aclweb.org/anthology/N19-1113)

[2] https://github.com/karlstratos/iaan

[3] https://github.com/ryanmcd/uni-dep-tb
