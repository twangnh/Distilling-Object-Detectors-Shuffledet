import numpy as np

from config import base_model_config

def kitti_shuffledet_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('KITTI')

  mc.IMAGE_WIDTH           = 1248
  mc.IMAGE_HEIGHT          = 384
  # mc.IMAGE_WIDTH           = 560
  # mc.IMAGE_HEIGHT          = 180

  mc.BATCH_SIZE            = 8

  mc.WEIGHT_DECAY          = 0.0005
  mc.LEARNING_RATE         = 0.01
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 5.0
  mc.LOSS_COEF_CONF_POS    = 75.0
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0

  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 64

  mc.DATA_AUGMENTATION     = True
  # mc.DRIFT_X               = 150/(1248./560)
  # mc.DRIFT_Y               = 100/(384./180)
  mc.DRIFT_X               = 150
  mc.DRIFT_Y               = 100
  mc.EXCLUDE_HARD_EXAMPLES = False

  mc.ANCHOR_BOX            = set_anchors(mc)
  mc.ANCHORS               = len(mc.ANCHOR_BOX)
  mc.ANCHOR_PER_GRID       = 9

  return mc

def set_anchors(mc):
  H, W, B = mc.IMAGE_HEIGHT // 16, mc.IMAGE_WIDTH // 16, 9
  #H, W, B = 12, 35, 9
  #original anchors
  anchor_shape_base = np.array(
          [[  36.,  37.], [ 366., 174.], [ 115.,  59.],
           [ 162.,  87.], [  38.,  90.], [ 258., 173.],
           [ 224., 108.], [  78., 170.], [  72.,  43.]])

  # randomly modified anchors
  # anchor_shape_base = np.array(
  #         [[  50.,  50.], [ 320., 180.], [ 90.,  48.],
  #          [ 180.,  100.], [  50.,  120.], [ 200., 130.],
  #          [ 180., 80.], [  90., 190.], [  100.,  60.]])

  # anchor_shape_base = np.array(
  # [[20.63007745, 45.40804647],
  #  [69.9036478, 153.81476415],
  #  [135.64310606, 213.72166667],
  #  [39.594868, 86.59731785],
  #  [209.20414977, 127.49268851],
  #  [75.59330804, 47.45570814],
  #  [337.28631668, 174.02375953],
  #  [130.50749455, 72.92875091],
  #  [38.78412702, 28.50398895]])


  anchor_shapes = np.reshape(
      [anchor_shape_base] * H * W,
      (H, W, B, 2)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_shapes), axis=3),
      (-1, 4)
  )

  return anchors
