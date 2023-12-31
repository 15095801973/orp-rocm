import torch

from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   multi_reppoints=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)
    
    if multi_reppoints is not None:
        reppoints = multi_reppoints[:, None].expand(-1, num_classes, multi_reppoints.size(-1))
        
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    
    if multi_reppoints is not None:
        reppoints = reppoints[valid_mask] 
        
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        if multi_reppoints is None:
            bboxes = multi_bboxes.new_zeros((0, 5))
        else:
            bboxes = multi_bboxes.new_zeros((0, reppoints.size(-1) + 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1), **nms_cfg_)
    bboxes = bboxes[keep]
    if multi_reppoints is not None:
        reppoints = reppoints[keep]
        bboxes = torch.cat([reppoints, bboxes], dim=1)
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([bboxes, scores[:, None]], 1), labels

# TODO 新的init_points
def multiclass_rnms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   multi_reppoints=None,
                    multi_reppoints_init=None
                    ):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 8:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 8)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 8)
        
    if multi_reppoints is not None:
        reppoints = multi_reppoints[:, None].expand(-1, num_classes, multi_reppoints.size(-1))
        if multi_reppoints_init is not None:
            reppoints_init = multi_reppoints_init[:, None].expand(-1, num_classes, multi_reppoints_init.size(-1))

    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    valid_mask = scores > score_thr
    bboxes = bboxes[valid_mask]
    
    if multi_reppoints is not None:
        reppoints = reppoints[valid_mask]
        if multi_reppoints_init is not None:
            reppoints_init = reppoints_init[valid_mask]

    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = scores[valid_mask]
    labels = valid_mask.nonzero()[:, 1]
    
    if bboxes.numel() == 0:
        if multi_reppoints is None:
            print("错误,在于'if multi_reppoints is None:'")
            bboxes = multi_bboxes.new_zeros((0, 9))
        # else:
            # bboxes = multi_bboxes.new_zeros((0, reppoints.size(-1) + 9))
        elif multi_reppoints_init is None:
            bboxes = multi_bboxes.new_zeros((0, reppoints.size(-1) + 9))
        else:
            bboxes = multi_bboxes.new_zeros((0, reppoints.size(-1) * 2 + 9))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'rnms')
    nms_op = getattr(nms_wrapper, nms_type)
    
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1), **nms_cfg_)
    
    bboxes = bboxes[keep]
    # print('bboxes_nms', bboxes)
    if multi_reppoints is not None:
        reppoints = reppoints[keep]
        if multi_reppoints_init is not None:
            reppoints_init = reppoints_init[keep]
            # bboxes = torch.cat([reppoints, bboxes], dim=1)
            bboxes = torch.cat([reppoints, reppoints_init, bboxes], dim=1)
            # print('bboxes_reppoints', bboxes)
        else:
            bboxes = torch.cat([reppoints, bboxes], dim=1)

    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]

    if keep.size(0) > max_num:
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]

    return torch.cat([bboxes, scores[:, None]], 1), labels
