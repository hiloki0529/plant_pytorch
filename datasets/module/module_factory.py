from .pillow import Pillow
from .cv import CV
from .scaling import Scaling
from .bad_condition import BadCondition
from .bad_condition_scaling import BadConditionScaling

ModuleType = {
        "pil": Pillow(),
        "cv": CV(),
        "scaling": Scaling(),
        "bad condition": BadCondition(),
        "bad condition scaling": BadConditionScaling(),
        }

def create_module(module):
    for key in ModuleType.keys():
        if set(module.split()) == set(key.split()):
            return ModuleType[key]
    raise ValueError("{} was not found".format(module))
