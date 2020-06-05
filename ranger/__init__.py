from ranger.radam import RAdam
from ranger.lookahead import Lookahead


def ranger(params, lr=1e-4, weight_decay=0):
    radam = RAdam(params, lr=lr, weight_decay=weight_decay)
    lookahead = Lookahead(radam)
    return lookahead
