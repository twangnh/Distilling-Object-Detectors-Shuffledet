import os
import random
import shutil

from PIL import Image, ImageFont, ImageDraw
import cv2
import numpy as np
from lib.utils.util import iou, batch_iou
from augmentation import BrightnessTransform, ContrastTransform, SaturationTransform


class imdb(object):
    """Image database."""

    def __init__(self, name, mc):
        self._name = name
        self._classes = []
        self._image_set = []
        self._image_idx = []
        self._data_root_path = []
        self._rois = {}
        self.mc = mc

        # batch reader
        self._perm_idx = None
        self._cur_idx = 0

        #mimic thresh
        self._mimic_trhesh1= 0.5
        self._mimic_trhesh2= 0.5
    @property
    def name(self):
        return self._name

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def image_idx(self):
        return self._image_idx

    @property
    def image_set(self):
        return self._image_set

    @property
    def data_root_path(self):
        return self._data_root_path

    @property
    def year(self):
        return self._year

    def _shuffle_image_idx(self):
        self._perm_idx = [self._image_idx[i] for i in
                          np.random.permutation(np.arange(len(self._image_idx)))]
        self._cur_idx = 0

    def read_image_batch(self, shuffle=True):
        """Only Read a batch of images
        Args:
          shuffle: whether or not to shuffle the dataset
        Returns:
          images: length batch_size list of arrays [height, width, 3]
        """
        mc = self.mc
        if shuffle:
            if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
                self._shuffle_image_idx()
            batch_idx = self._perm_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
            self._cur_idx += mc.BATCH_SIZE
        else:
            if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
                batch_idx = self._image_idx[self._cur_idx:] \
                            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE - len(self._image_idx)]
                self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
            else:
                batch_idx = self._image_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
                self._cur_idx += mc.BATCH_SIZE

        images, scales = [], []
        for i in batch_idx:
            im = cv2.imread(self._image_path_at(i))
            im = im.astype(np.float32, copy=False)
            im -= mc.BGR_MEANS
            orig_h, orig_w, _ = [float(v) for v in im.shape]
            im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
            x_scale = mc.IMAGE_WIDTH / orig_w
            y_scale = mc.IMAGE_HEIGHT / orig_h
            images.append(im)
            scales.append((x_scale, y_scale))

        return images, scales

    # only used for kitti test set output
    # def read_image_batch_testset(self, shuffle=True):
    #
    #     def image_path_at(idx):
    #         image_path = os.path.join('./data/KITTI/testing/image_2', idx)
    #         assert os.path.exists(image_path), \
    #             'Image does not exist: {}'.format(image_path)
    #         return image_path
    #
    #     mc = self.mc
    #     if shuffle:
    #         if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
    #             self._shuffle_image_idx()
    #         batch_idx = self._perm_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
    #         self._cur_idx += mc.BATCH_SIZE
    #     else:
    #         if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
    #             batch_idx = self._image_idx[self._cur_idx:] \
    #                         + self._image_idx[:self._cur_idx + mc.BATCH_SIZE - len(self._image_idx)]
    #             self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
    #         else:
    #             batch_idx = self._image_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
    #             self._cur_idx += mc.BATCH_SIZE
    #
    #     images, scales = [], []
    #     for i in batch_idx:
    #         im = cv2.imread(image_path_at(i))
    #         # modify for eval result output
    #         im_orig = im
    #         im = im.astype(np.float32, copy=False)
    #         im -= mc.BGR_MEANS
    #         orig_h, orig_w, _ = [float(v) for v in im.shape]
    #         im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
    #         x_scale = mc.IMAGE_WIDTH / orig_w
    #         y_scale = mc.IMAGE_HEIGHT / orig_h
    #         images.append(im)
    #         scales.append((x_scale, y_scale))
    #
    #     return images, im_orig, scales

    def read_batch(self, shuffle=True, batch_size=1):
        """Read a batch of image and bounding box annotations.
        Args:
          shuffle: whether or not to shuffle the dataset
        Returns:
          image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
          label_per_batch: labels. Shape: batch_size x object_num
          delta_per_batch: bounding box deltas. Shape: batch_size x object_num x
              [dx ,dy, dw, dh]
          aidx_per_batch: index of anchors that are responsible for prediction.
              Shape: batch_size x object_num
          bbox_per_batch: scaled bounding boxes. Shape: batch_size x object_num x
              [cx, cy, w, h]
        """
        mc = self.mc

        if shuffle:
            if self._cur_idx + batch_size >= len(self._image_idx):
                self._shuffle_image_idx()
            batch_idx = self._perm_idx[self._cur_idx:self._cur_idx + batch_size]
            self._cur_idx += batch_size
        else:
            if self._cur_idx + batch_size >= len(self._image_idx):
                batch_idx = self._image_idx[self._cur_idx:] \
                            + self._image_idx[:self._cur_idx + batch_size - len(self._image_idx)]
                self._cur_idx += batch_size - len(self._image_idx)
            else:
                batch_idx = self._image_idx[self._cur_idx:self._cur_idx + batch_size]
                self._cur_idx += batch_size

        image_per_batch = []
        label_per_batch = []
        bbox_per_batch = []
        delta_per_batch = []
        aidx_per_batch = []
        if mc.DEBUG_MODE:
            avg_ious = 0.
            num_objects = 0.
            max_iou = 0.0
            min_iou = 1.0
            num_zero_iou_obj = 0

        mask_per_batch = np.zeros([len(batch_idx), 24, 78]).astype(bool)
        mask_per_batch2 = np.zeros([len(batch_idx), 24, 78, 9]).astype(bool)
        for num_idx, idx in enumerate(batch_idx):
            # load the image
            # im = cv2.imread(self._image_path_at(idx)).astype(np.float32, copy=False)
            ##Mar 14 add augmentation
            im = cv2.imread(self._image_path_at(idx))
            im = BrightnessTransform(im)
            im = ContrastTransform(im)
            im = SaturationTransform(im)
            im = im.astype(np.float32, copy=False)

            im -= mc.BGR_MEANS
            orig_h, orig_w, _ = [float(v) for v in im.shape]

            # load annotations
            label_per_batch.append([b[4] for b in self._rois[idx][:]])
            gt_bbox = np.array([[b[0], b[1], b[2], b[3]] for b in self._rois[idx][:]])

            if mc.DATA_AUGMENTATION:
                assert mc.DRIFT_X >= 0 and mc.DRIFT_Y > 0, \
                    'mc.DRIFT_X and mc.DRIFT_Y must be >= 0'

                if mc.DRIFT_X > 0 or mc.DRIFT_Y > 0:
                    # Ensures that gt boundibg box is not cutted out of the image
                    max_drift_x = min(gt_bbox[:, 0] - gt_bbox[:, 2] / 2.0 + 1)
                    max_drift_y = min(gt_bbox[:, 1] - gt_bbox[:, 3] / 2.0 + 1)
                    assert max_drift_x >= 0 and max_drift_y >= 0, 'bbox out of image'

                    dy = np.random.randint(-mc.DRIFT_Y, min(mc.DRIFT_Y + 1, max_drift_y))
                    dx = np.random.randint(-mc.DRIFT_X, min(mc.DRIFT_X + 1, max_drift_x))

                    # shift bbox
                    gt_bbox[:, 0] = gt_bbox[:, 0] - dx
                    gt_bbox[:, 1] = gt_bbox[:, 1] - dy

                    # distort image
                    orig_h -= dy
                    orig_w -= dx
                    orig_x, dist_x = max(dx, 0), max(-dx, 0)
                    orig_y, dist_y = max(dy, 0), max(-dy, 0)

                    distorted_im = np.zeros(
                        (int(orig_h), int(orig_w), 3)).astype(np.float32)
                    distorted_im[dist_y:, dist_x:, :] = im[orig_y:, orig_x:, :]
                    im = distorted_im

                # Flip image with 50% probability
                if np.random.randint(2) > 0.5:
                    im = im[:, ::-1, :]
                    gt_bbox[:, 0] = orig_w - 1 - gt_bbox[:, 0]

            # scale image
            im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
            image_per_batch.append(im)

            # scale annotation
            x_scale = mc.IMAGE_WIDTH / orig_w
            y_scale = mc.IMAGE_HEIGHT / orig_h
            gt_bbox[:, 0::2] = gt_bbox[:, 0::2] * x_scale
            gt_bbox[:, 1::2] = gt_bbox[:, 1::2] * y_scale
            bbox_per_batch.append(gt_bbox)

            aidx_per_image, delta_per_image = [], []
            aidx_set = set()

            mask_per_img = np.zeros([24, 78])
            mask_per_img2 = np.zeros([24, 78, 9])
            for i in range(len(gt_bbox)):
                overlaps = batch_iou(mc.ANCHOR_BOX, gt_bbox[i])
                #############add for mask preparation
                ##modify for det head mimic
                overlaps_temp = np.transpose(np.reshape(overlaps, [24, 78, 9]), [2, 0, 1])

                max_overlap_per_gbox = np.max(overlaps_temp)
                positive_thresh1 = max_overlap_per_gbox * self._mimic_trhesh1
                mask_per_gbox_per_anchor1 = (overlaps_temp > positive_thresh1).astype(int)

                # merge all anchor mask, for last feature mimic
                mask_per_gbox = mask_per_gbox_per_anchor1[0]
                for anchor_inx in range(len(overlaps_temp)):
                    mask_per_gbox += mask_per_gbox_per_anchor1[anchor_inx]

                # modify for gt box supervison
                mask_per_img += mask_per_gbox
                # x = int(gt_bbox[i][0]//16)
                # y = int(gt_bbox[i][1]//16)
                # w = int(gt_bbox[i][2]//16)
                # h = int(gt_bbox[i][3]//16)
                #
                # mask_per_img[(y-h//2):y+(h//2), (x-w//2):x+(w//2)] +=1

                # for det head mimic
                positive_thresh2 = max_overlap_per_gbox * self._mimic_trhesh2
                # [24, 78, 9]
                mask_per_gbox_per_anchor2 = (np.reshape(overlaps, [24, 78, 9]) > positive_thresh2) \
                    .astype(int)
                mask_per_img2 += mask_per_gbox_per_anchor2

                # overlaps_batch.append(np.transpose(np.reshape(overlaps,[12,35,9]),[2,0,1]))

                aidx = len(mc.ANCHOR_BOX)
                for ov_idx in np.argsort(overlaps)[::-1]:
                    if overlaps[ov_idx] <= 0:
                        if mc.DEBUG_MODE:
                            min_iou = min(overlaps[ov_idx], min_iou)
                            num_objects += 1
                            num_zero_iou_obj += 1
                        break
                    if ov_idx not in aidx_set:
                        aidx_set.add(ov_idx)
                        aidx = ov_idx
                        if mc.DEBUG_MODE:
                            max_iou = max(overlaps[ov_idx], max_iou)
                            min_iou = min(overlaps[ov_idx], min_iou)
                            avg_ious += overlaps[ov_idx]
                            num_objects += 1
                        break

                if aidx == len(mc.ANCHOR_BOX):
                    # even the largeset available overlap is 0, thus, choose one with the
                    # smallest square distance
                    dist = np.sum(np.square(gt_bbox[i] - mc.ANCHOR_BOX), axis=1)
                    for dist_idx in np.argsort(dist):
                        if dist_idx not in aidx_set:
                            aidx_set.add(dist_idx)
                            aidx = dist_idx
                            break

                box_cx, box_cy, box_w, box_h = gt_bbox[i]
                delta = [0] * 4
                delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0]) / mc.ANCHOR_BOX[aidx][2]
                delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1]) / mc.ANCHOR_BOX[aidx][3]
                delta[2] = np.log(box_w / mc.ANCHOR_BOX[aidx][2])
                delta[3] = np.log(box_h / mc.ANCHOR_BOX[aidx][3])

                aidx_per_image.append(aidx)
                delta_per_image.append(delta)

            mask_per_batch[num_idx] = mask_per_img > 0
            mask_per_batch2[num_idx] = mask_per_img2 > 0

            delta_per_batch.append(delta_per_image)
            aidx_per_batch.append(aidx_per_image)

        if mc.DEBUG_MODE:
            print ('max iou: {}'.format(max_iou))
            print ('min iou: {}'.format(min_iou))
            print ('avg iou: {}'.format(avg_ious / num_objects))
            print ('number of objects: {}'.format(num_objects))
            print ('number of objects with 0 iou: {}'.format(num_zero_iou_obj))

        return image_per_batch, label_per_batch, delta_per_batch, \
               aidx_per_batch, bbox_per_batch, mask_per_batch, mask_per_batch2

    def load_images_and_encode_target(self):

        def sparse_to_dense(sp_indices, output_shape, values, default_value=0):
            """Build a dense matrix from sparse representations.

            Args:
              sp_indices: A [0-2]-D array that contains the index to place values.
              shape: shape of the dense matrix.
              values: A {0,1}-D array where values corresponds to the index in each row of
              sp_indices.
              default_value: values to set for indices not specified in sp_indices.
            Return:
              A dense numpy N-D array with shape output_shape.
            """

            assert len(sp_indices) == len(values), \
                'Length of sp_indices is not equal to length of values'

            array = np.ones(output_shape) * default_value
            for idx, value in zip(sp_indices, values):
                array[tuple(idx)] = value
            return array

        image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
        bbox_per_batch, mask_per_batch, mask_per_batch2 = self.read_batch()

        label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
            = [], [], [], [], []
        aidx_set = set()
        num_discarded_labels = 0
        num_labels = 0
        for i in range(len(label_per_batch)):  # batch_size
            for j in range(len(label_per_batch[i])):  # number of annotations
                num_labels += 1
                if (i, aidx_per_batch[i][j]) not in aidx_set:
                    aidx_set.add((i, aidx_per_batch[i][j]))
                    label_indices.append(
                        [i, aidx_per_batch[i][j], label_per_batch[i][j]])
                    mask_indices.append([i, aidx_per_batch[i][j]])
                    bbox_indices.extend(
                        [[i, aidx_per_batch[i][j], k] for k in range(4)])
                    box_delta_values.extend(box_delta_per_batch[i][j])
                    box_values.extend(bbox_per_batch[i][j])
                else:
                    num_discarded_labels += 1


        image_input = image_per_batch,
        input_mask = np.reshape(
            sparse_to_dense(
                mask_indices, [1, self.mc.ANCHORS],
                [1.0] * len(mask_indices)),
            [1, self.mc.ANCHORS, 1]),
        box_delta_input= sparse_to_dense(
            bbox_indices, [1, self.mc.ANCHORS, 4],
            box_delta_values),
        box_input= sparse_to_dense(
            bbox_indices, [1, self.mc.ANCHORS, 4],
            box_values),
        labels= sparse_to_dense(
            label_indices,
            [1, self.mc.ANCHORS, self.mc.CLASSES],
            [1.0] * len(label_indices)),


        return image_input[0],input_mask[0],box_delta_input[0],box_input[0],labels[0], mask_per_batch, mask_per_batch2

    def next_batch(self):
        while True:
            image_input, input_mask, box_delta_input, box_input, labels, mask_per_batch, mask_per_batch2 = self.load_images_and_encode_target()
            yield (image_input, input_mask, box_delta_input, box_input, labels, mask_per_batch, mask_per_batch2)

    def evaluate_detections(self):
        raise NotImplementedError

    def visualize_detections(
            self, image_dir, image_format, det_error_file, output_image_dir,
            num_det_per_type=10):

        # load detections
        with open(det_error_file) as f:
            lines = f.readlines()
            random.shuffle(lines)
        f.close()

        dets_per_type = {}
        for line in lines:
            obj = line.strip().split(' ')
            error_type = obj[1]
            if error_type not in dets_per_type:
                dets_per_type[error_type] = [{
                    'im_idx': obj[0],
                    'bbox': [float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
                    'class': obj[6],
                    'score': float(obj[7])
                }]
            else:
                dets_per_type[error_type].append({
                    'im_idx': obj[0],
                    'bbox': [float(obj[2]), float(obj[3]), float(obj[4]), float(obj[5])],
                    'class': obj[6],
                    'score': float(obj[7])
                })

        out_ims = []
        # Randomly select some detections and plot them
        COLOR = (200, 200, 0)
        for error_type, dets in dets_per_type.iteritems():
            det_im_dir = os.path.join(output_image_dir, error_type)
            if os.path.exists(det_im_dir):
                shutil.rmtree(det_im_dir)
            os.makedirs(det_im_dir)

            for i in range(min(num_det_per_type, len(dets))):
                det = dets[i]
                im = Image.open(
                    os.path.join(image_dir, det['im_idx'] + image_format))
                draw = ImageDraw.Draw(im)
                draw.rectangle(det['bbox'], outline=COLOR)
                draw.text((det['bbox'][0], det['bbox'][1]),
                          '{:s} ({:.2f})'.format(det['class'], det['score']),
                          fill=COLOR)
                out_im_path = os.path.join(det_im_dir, str(i) + image_format)
                im.save(out_im_path)
                im = np.array(im)
                out_ims.append(im[:, :, ::-1])  # RGB to BGR
        return out_ims
