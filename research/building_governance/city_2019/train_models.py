import time,traceback
from ml import *
try:
 status('TRAINING_START');fit('alignment');fit('facade');save('MODEL_FREEZE.json',dict(time=datetime.now().astimezone().isoformat(),sequence='alignment training and validation, facade training and validation, then whole-city inference',checkpoints=['alignment_best.pt','facade_best.pt'],protected_reads=0));status('MODELS_FROZEN')
except BaseException:save('TRAINING_FAILURE.json',dict(traceback=traceback.format_exc()));raise
