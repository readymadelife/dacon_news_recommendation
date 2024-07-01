def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set=  set(predicted[:k])
    return len(act_set & pred_set) / float(len(act_set))