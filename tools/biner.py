

class CutByThreshold:
    def __init__(self, threshold_list, include_invalid_bin=False):
        self.threshold_list = threshold_list
        self.last_bin_idx = len(self.threshold_list)
        self.invalid_bin_idx = self.last_bin_idx + 1 if include_invalid_bin else None

    def cut(self, numerator, denominator, is_valid_marker_value=None):
        # is_valid_marker_value小于0意味着action无效，直接返回占位值
        if is_valid_marker_value is not None or self.invalid_bin_idx is not None:
            assert self.invalid_bin_idx is not None and self.invalid_bin_idx is not None, f"invalid_bin_idx={self.invalid_bin_idx}, is_valid_marker_value={is_valid_marker_value}"
            if is_valid_marker_value < 0:
                return self.invalid_bin_idx

        # 分母为0意味着值无限大，分到最后一个桶
        if denominator == 0:
            return self.last_bin_idx

        value = numerator / denominator
        target_bin = None
        for current_bin, threshold in enumerate(self.threshold_list):
            if value < threshold:
                target_bin = current_bin
                break

        if target_bin is None:
            target_bin = self.last_bin_idx
        return target_bin
