def precision(true_positive: int, false_positive: int) -> float:
	denominator = true_positive + false_positive
	if denominator == 0:
		return 0.0
	return true_positive / denominator


def recall(true_positive: int, false_negative: int) -> float:
	denominator = true_positive + false_negative
	if denominator == 0:
		return 0.0
	return true_positive / denominator


def fall_out(true_negative: int, false_positive: int) -> float:
	denominator = true_negative + false_positive
	if denominator == 0:
		return 0.0
	return false_positive / denominator