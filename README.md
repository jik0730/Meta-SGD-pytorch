# Meta-SGD in pytorch

The only difference compared to MAML is to parametrize task learning rate in vector form when meta-training.

## Performance comparisions to MAML

The reported performance are refered to the ones in Meta-SGD paper.

|    Omniglot   | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|:-------------:|:------------:|:------------:|:-------------:|:-------------:|
|      MAML     |     98.7%    |     99.9%    |     95.8%     |     98.9%     |
|   Ours MAML   |     99.4%    |     99.9%    |     92.8%     |       -       |
|    Meta-SGD   |     99.5%    |     99.9%    |     95.9%     |     99.0%     |
| Ours Meta-SGD |              |              |               |               |

|  miniImageNet | 5-way 1-shot | 5-way 5-shot | 20-way 1-shot | 20-way 5-shot |
|:-------------:|:------------:|:------------:|:-------------:|:-------------:|
|      MAML     |     48.7%    |     63.1%    |     16.5%     |     19.3%     |
|   Ours MAML   |              |              |               |       -       |
|    Meta-SGD   |     50.5%    |     64.0%    |     17.6%     |     28.9%     |
| Ours Meta-SGD |              |              |               |               |

## TODO
- TBD