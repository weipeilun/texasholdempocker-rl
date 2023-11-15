

class CutByThreshold:
    def __init__(self, threshold_list):
        self.threshold_list = threshold_list

        self.num_bins = len(self.threshold_list) + 1

    def cut(self, value):
        target_bin = None
        for current_bin, threshold in enumerate(self.threshold_list):
            if value <= threshold:
                target_bin = current_bin
                break

        if target_bin is None:
            target_bin = self.num_bins
        return target_bin
