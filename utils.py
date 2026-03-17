import numpy as np
from torch import nn
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage import distance_transform_edt as distance

def compute_nsd(pred, gt, tolerance=1.0):
    """
    pred, gt: (H, W) binary numpy array
    tolerance: pixel distance
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    # surface extraction
    pred_surface = pred ^ binary_erosion(pred)
    gt_surface = gt ^ binary_erosion(gt)

    # distance maps
    dt_pred = distance_transform_edt(~pred_surface)
    dt_gt = distance_transform_edt(~gt_surface)

    # surface points within tolerance
    pred_match = (pred_surface & (dt_gt <= tolerance)).sum()
    gt_match = (gt_surface & (dt_pred <= tolerance)).sum()

    nsd = (pred_match + gt_match) / (pred_surface.sum() + gt_surface.sum() + 1e-8)
    return nsd

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def compute_dtm(img_gt, out_shape):
    """
    Compute the signed distance transform map for a given binary mask.
    
    Returns a map where:
    - Inside object: Negative values (distance to boundary, becomes more negative deeper inside)
    - Outside object: Positive values (distance to boundary, becomes more positive further outside)
    - At boundary: Zero
    
    This creates a proper gradient signal for boundary learning.
    """
    # posmask: Object=1, Background=0
    posmask = img_gt.astype(bool)
    
    # distance(posmask): Calculates distance from Object(1) to nearest Background(0).
    # Values are >0 inside the object, 0 outside.
    dist_inside = distance(posmask)
    
    # distance(~posmask): Calculates distance from Background(1) to nearest Object(0).
    # Values are >0 outside the object, 0 inside.
    dist_outside = distance(np.logical_not(posmask))
    
    # Combine: Positive outside - Positive inside
    # Result:
    #   Inside: 0 - Positive = Negative (deeper inside = more negative)
    #   Outside: Positive - 0 = Positive (further outside = more positive)
    #   Boundary: 0 - 0 = 0
    dtm = dist_outside - dist_inside
    
    # Optional: Clip to avoid exploding gradients for very large images
    # Uncomment if you see gradient issues with large distance values
    # dtm = np.clip(dtm, -20, 20)
    
    return dtm


class BoundaryLoss(nn.Module):
    """
    Boundary-Aware Loss that penalizes predictions deviating from ground truth contours.
    This is crucial for improving NSD (Normalized Surface Distance) metric performance.
    """
    def __init__(self, n_classes):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        """Convert class indices to one-hot encoding."""
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input_tensor == i).float()
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target):
        """
        Calculate boundary loss.
        
        Args:
            inputs: [B, C, H, W] - Logits from the model
            target: [B, H, W] - Ground truth indices (0, 1, 2)
        
        Returns:
            loss: Scalar boundary loss value
        """
        # 1. Convert logits to probabilities
        inputs = torch.softmax(inputs, dim=1)
        
        # 2. Get shapes
        B, C, H, W = inputs.shape
        
        # 3. Compute Distance Maps on CPU (Scipy is CPU only)
        # We do this on-the-fly because augmentation changes the boundary every epoch
        target_numpy = target.cpu().numpy()
        dtm_batch = np.zeros((B, C, H, W), dtype=np.float32)
        
        for b in range(B):
            for c in range(1, C):  # Skip background (class 0), focus on Plaque (1) and Vessel (2)
                # Create binary mask for this class
                mask_c = (target_numpy[b] == c).astype(np.uint8)
                
                # Compute distance map
                dtm_batch[b, c, :, :] = compute_dtm(mask_c, (H, W))
        
        # 4. Convert DTM back to GPU tensor
        dtm_tensor = torch.from_numpy(dtm_batch).to(inputs.device).float()
        
        # 5. Calculate Loss
        # Multiplying Probabilities * Distance Map
        # 
        # Mathematical formulation:
        #   loss = mean(prob × DTM)
        #   - Inside pixels: DTM < 0, prob should be high → prob × DTM = large negative (good!)
        #   - Outside pixels: DTM > 0, prob should be low → prob × DTM = small positive (good!)
        #   - Result: Overall loss is negative when model predicts correctly
        #
        # Interpretation:
        #   - More negative = Better (predictions are inside objects, not spilling outside)
        #   - Less negative (closer to 0) = Worse (predictions spilling outside or not filling inside)
        #   - Optimizer minimizes this, so it tries to make it MORE negative (better)
        #
        # Gradient Analysis:
        #   ∂loss/∂logit = DTM × prob × (1 - prob)
        #   - Gradients exist as long as DTM ≠ 0 (true for most pixels)
        #   - Even if loss is small (e.g., -0.04), gradients still exist:
        #     * Pixels spilling outside (high prob × positive DTM) → positive gradient → reduce prob
        #     * Pixels not filling inside (low prob × negative DTM) → negative gradient → increase prob
        #   - Learning continues as long as there are boundary errors
        #
        # Note: This is mathematically correct but counterintuitive (negative = good).
        # The loss being negative is expected and correct behavior.
        loss = torch.einsum("bchw, bchw -> bchw", inputs, dtm_tensor)
        
        # Average over the batch and spatial dimensions
        loss = loss.mean()
        
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        # Accept input of shape [B,H,W] or [B,1,H,W] and return [B,C,H,W]
        if input_tensor.ndim == 3:
            # [B,H,W] -> [B,1,H,W]
            input_tensor = input_tensor.unsqueeze(1)
        tensor_list = []
        for i in range(self.n_classes):
            # comparison yields [B,1,H,W]; cast to float
            temp_prob = (input_tensor == i).float()
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)  # -> [B,C,H,W]
        return output_tensor

    def _dice_loss(self, score, target, ignore):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score[ignore != 1] * target[ignore != 1])
        y_sum = torch.sum(target[ignore != 1] * target[ignore != 1])
        z_sum = torch.sum(score[ignore != 1] * score[ignore != 1])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False, ignore=None):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i], ignore)
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count