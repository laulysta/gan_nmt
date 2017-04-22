import numpy as np
import os
import sys

model = sys.argv[1]
history_errs = list(np.load(model)['history_errs'])
print('Validation NLL: {}'.format(history_errs[-1][0]))


