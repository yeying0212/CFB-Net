import numpy as np
import cv2


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x + 1e-6) ** -1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tp = hist[1, 1]
    fn = hist[1, 0]
    fp = hist[0, 1]
    tn = hist[0, 0]
    # recall
    recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # precision
    precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # F1 score
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return f1


def cm2score(confusion_matrix):#计算混淆矩阵的各种指标
    hist = confusion_matrix
    tp = hist[1, 1]
    fn = hist[1, 0]
    fp = hist[0, 1]
    tn = hist[0, 0]
    # acc
    oa = (tp + tn) / (tp + fn + fp + tn + np.finfo(np.float32).eps)
    # recall
    recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # precision
    precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # F1 score
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    # IoU
    iou = tp / (tp + fp + fn + np.finfo(np.float32).eps)
    # pre
    pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    # kappa
    kappa = (oa - pre) / (1 - pre)
    score_dict = {'Kappa': kappa, 'IoU': iou, 'F1': f1, 'OA': oa, 'recall': recall, 'precision': precision, 'Pre': pre}
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""

    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes ** 2).reshape(num_classes, num_classes)
        return hist

    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


class CDMetricsMeter(object):
    """Change Detection Metrics Meter - Boundary F-score.

    Evaluates boundary accuracy with spatial tolerance:
      - Boundary extracted via morphological gradient (fixed 3x3 kernel, ~2px wide)
      - tolerance controls matching distance: PR boundary within `tolerance` pixels of GT = hit
      - Precision / Recall / F1 computed from globally accumulated pixel counts
    """

    def __init__(self, tolerance=2):
        self.tolerance = tolerance

        # 容忍距离用 tolerance 控制
        t_ksize = 2 * tolerance + 1
        self._tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t_ksize, t_ksize))

        self.reset()

    def reset(self):
        self.initialized = False
        self.count = 0
        self.b_pr_total = 0.0
        self.b_gt_total = 0.0
        self.b_tp_p = 0.0
        self.b_tp_r = 0.0

    def _extract_boundary(self, mask):
        # thin contour: 和4邻域任一不同就是边界
        m = mask.astype(np.uint8)
        b = np.zeros_like(m)
        b[:, :-1] |= (m[:, :-1] != m[:, 1:])
        b[:, 1:]  |= (m[:, 1:] != m[:, :-1])
        b[:-1, :] |= (m[:-1, :] != m[1:, :])
        b[1:, :]  |= (m[1:, :] != m[:-1, :])
        return b

    def update(self, pr, gt, weight=1):
        eps = np.finfo(np.float32).eps

        b_pr = self._extract_boundary(pr)
        b_gt = self._extract_boundary(gt)

        n_pr = np.count_nonzero(b_pr)
        n_gt = np.count_nonzero(b_gt)

        tol_gt = cv2.dilate(b_gt, self._tol_kernel)
        tol_pr = cv2.dilate(b_pr, self._tol_kernel)

        tp_p = np.count_nonzero(b_pr & tol_gt)
        tp_r = np.count_nonzero(b_gt & tol_pr)

        self.b_pr_total += n_pr
        self.b_gt_total += n_gt
        self.b_tp_p += tp_p
        self.b_tp_r += tp_r

        b_precision = tp_p / (n_pr + eps)
        b_recall = tp_r / (n_gt + eps)
        b_f1 = 2.0 * b_precision * b_recall / (b_precision + b_recall + 1e-12)
        if n_pr == 0 and n_gt == 0:
            b_f1 = 1.0
            b_precision = 1.0
            b_recall = 1.0

        self.count += weight
        self.initialized = True

        return {
            'boundary_f1': float(b_f1),
            'boundary_precision': float(b_precision),
            'boundary_recall': float(b_recall),
        }

    def get_scores(self):
        eps = np.finfo(np.float32).eps

        global_b_p = self.b_tp_p / (self.b_pr_total + eps)
        global_b_r = self.b_tp_r / (self.b_gt_total + eps)
        global_b_f1 = 2.0 * global_b_p * global_b_r / (global_b_p + global_b_r + 1e-12)

        if self.b_pr_total == 0.0 and self.b_gt_total == 0.0:
            global_b_f1 = 1.0
            global_b_p = 1.0
            global_b_r = 1.0

        return {
            'Boundary_F1': float(global_b_f1),
            'Boundary_Precision': float(global_b_p),
            'Boundary_Recall': float(global_b_r),
        }
