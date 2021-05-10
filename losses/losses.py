import torch
import audtorch

def pearsonr(input_, target_):
    return audtorch.metrics.functional.pearsonr(input_, target_)


def concordance_cc(input_, target_):
    return audtorch.metrics.functional.concordance_cc(input_, target_)