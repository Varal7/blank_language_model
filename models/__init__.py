from . inst import InsTLM
from . blm import BLM
from . lblm import LBLM


def get_model_class(model_type):
    if model_type == 'blm':
        return BLM
    elif model_type == 'inst':
        return InsTLM
    elif model_type == 'lblm':
        return LBLM
    else:
        raise ValueError('Unknown model ' + model_type)
