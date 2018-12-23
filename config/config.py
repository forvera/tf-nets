from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

__C.NUM_EPOCHS = 1000
__C.LEARNING_RATE = 1e-4
__C.KEEP_PROB = 0.5
__C.TRAIN_LAYERS = 'fc8,fc7,fc6'
__C.NUM_CLASSES = 10
__C.TRAINING_FILE = ''
__C.CAL_FILE = ''
__C.CKPT_PATH = ''
__C.BATCH_SIZE = 64