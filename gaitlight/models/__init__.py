__all__ = []

import importlib
import inspect
import pkgutil

for _, _name, _ispkg in pkgutil.iter_modules(__path__):
    if _name == 'archive':
        continue
    module = importlib.import_module(f'.{_name}', package=__name__)

    # import attributes
    for _att in dir(module):
        att = getattr(module, _att)
        if inspect.isclass(att):
            globals()[_att] = att
            __all__.append(_att)

    # merge __all__ from module
    if hasattr(module, '__all__'):
        globals().update({_n: getattr(module, _n) for _n in module.__all__})
        __all__.extend(module.__all__)
