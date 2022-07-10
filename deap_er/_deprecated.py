# ====================================================================================== #
#                                                                                        #
#   MIT License                                                                          #
#                                                                                        #
#   Copyright (c) 2022 - Mattias Aabmets, The DEAP Team and Other Contributors           #
#                                                                                        #
#   Permission is hereby granted, free of charge, to any person obtaining a copy         #
#   of this software and associated documentation files (the "Software"), to deal        #
#   in the Software without restriction, including without limitation the rights         #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell            #
#   copies of the Software, and to permit persons to whom the Software is                #
#   furnished to do so, subject to the following conditions:                             #
#                                                                                        #
#   The above copyright notice and this permission notice shall be included in all       #
#   copies or substantial portions of the Software.                                      #
#                                                                                        #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR           #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,             #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE          #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER               #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,        #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE        #
#   SOFTWARE.                                                                            #
#                                                                                        #
# ====================================================================================== #
from typing import Any
from pathlib import Path
import traceback
import warnings
import inspect


warnings.filterwarnings('always', category=DeprecationWarning)


# ====================================================================================== #
def deprecated(old_name: str, obj: Any) -> Any:
    """
    This decorator wraps an object to notify developers of its old deprecated name.

    :param old_name: The old deprecated name of the object.
    :param obj: Object to be wrapped with deprecation warning.
    :return: Function which warns against using the old deprecated name.
    """
    def warn(obj_type: str, new_name: str):
        msg = f'\nWARNING! {obj_type} name \'{old_name}\' is deprecated! ' \
              f'Replace it with \'{new_name}\'!'
        tb = traceback.extract_stack()[0]
        file = Path(tb.filename)
        warnings.warn_explicit(
            message=msg,
            category=DeprecationWarning,
            filename=file.name,
            lineno=tb.lineno
        )

    if isinstance(obj, property):
        def wrapper(self):
            new_name = obj.fget.__name__
            warn(obj_type='Property', new_name=new_name)
            return getattr(self, new_name)
        return property(wrapper)

    elif inspect.isfunction(obj):
        def wrapper(*args, **kwargs):
            warn(obj_type='Function', new_name=obj.__name__)
            return obj(*args, **kwargs)
        return wrapper

    elif inspect.isclass(obj):
        class Wrapper(obj):
            def __init__(self, *args, **kwargs):
                warn(obj_type='Class', new_name=obj.__name__)
                super().__init__(*args, **kwargs)
        return Wrapper

    return None
