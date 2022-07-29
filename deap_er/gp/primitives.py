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
from __future__ import annotations
from collections import defaultdict, deque
from typing import Union, Type, Callable, Iterable, Any
import copy
import abc
import re


__all__ = [
    'Terminal', 'Ephemeral',
    'Primitive', 'PrimitiveTree',
    'PrimitiveSet', 'PrimitiveSetTyped'
]


# ====================================================================================== #
class Terminal:
    """
    Class that encapsulates a terminal primitive in an expression.
    Terminals can be either values or 0-arity functions.

    :param terminal: The terminal value or function.
    :param symbolic: If True, the terminal is a string.
    :param ret_type: The return type of the terminal.

    :type terminal: :ref:`TerminalTypes <datatypes>`
    """
    __slots__ = ('name', 'value', 'ret', 'conv_fct')

    # -------------------------------------------------------- #
    def __init__(self, terminal: Any, symbolic: bool, ret_type: type):
        self.ret = ret_type
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    # -------------------------------------------------------- #
    @property
    def arity(self):
        return 0

    # -------------------------------------------------------- #
    def format(self):
        return self.conv_fct(self.value)

    # -------------------------------------------------------- #
    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


# ====================================================================================== #
class Ephemeral(Terminal):
    """
    Class that encapsulates a terminal which value is set when
    the object is created. This is an abstract base class. When
    subclassing, a staticmethod named *'func'* must be defined.
    """
    # -------------------------------------------------------- #
    def __init__(self):
        Terminal.__init__(self, self.func(), symbolic=False, ret_type=self.ret)

    # -------------------------------------------------------- #
    @staticmethod
    @abc.abstractmethod
    def func():
        raise NotImplementedError


# ====================================================================================== #
class Primitive:
    """
    Class that encapsulates a primitive and when called with arguments it
    returns the Python code to call the primitive with the arguments.
    
    :param name: The name of the primitive.
    :param args: The list of arguments of the primitive.
    :param ret_type: The return type of the primitive.
    """
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    # -------------------------------------------------------- #
    def __init__(self, name: str, args: list, ret_type: type):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret_type
        args = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    # -------------------------------------------------------- #
    def format(self, *args):
        return self.seq.format(*args)

    # -------------------------------------------------------- #
    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


# ====================================================================================== #
class PrimitiveSetTyped:
    """
    Class that contains the primitives which can be
    used to solve a Strongly Typed GP problem.

    :param name: The name of the primitive set.
    :param in_types: The list of input types.
    :param ret_type: The return type.
    :param prefix: The prefix of the primitive set.
    """
    # -------------------------------------------------------- #
    def __init__(self, name: str, in_types: list,
                 ret_type: type, prefix: str = "ARG") -> None:
        self.name = name
        self.ins = in_types
        self.ret = ret_type

        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.context = {"__builtins__": None}
        self.arguments = list()
        self.mapping = dict()
        self.terms_count = 0
        self.prims_count = 0

        for i, type_ in enumerate(in_types):
            arg_str = "{prefix}{index}".format(prefix=prefix, index=i)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, type_)
            self._add_prim(term)
            self.terms_count += 1

    # -------------------------------------------------------- #
    @staticmethod
    def _add_type(mapping: dict, ret_type: Any) -> None:
        if ret_type not in mapping:
            new_list = []
            for type_, list_ in mapping.items():
                if issubclass(type_, ret_type):
                    for item in list_:
                        if item not in new_list:
                            new_list.append(item)
            mapping[ret_type] = new_list

    # -------------------------------------------------------- #
    def _add_prim(self, prim: Union[Primitive, Terminal, Type[Ephemeral]]) -> None:
        self._add_type(self.primitives, prim.ret)
        self._add_type(self.terminals, prim.ret)
        self.mapping[prim.name] = prim

        if isinstance(prim, Primitive):
            for type_ in prim.args:
                self._add_type(self.primitives, type_)
                self._add_type(self.terminals, type_)
            mapping = self.primitives
        else:
            mapping = self.terminals

        for type_ in mapping:
            if issubclass(prim.ret, type_):
                mapping[type_].append(prim)

    # -------------------------------------------------------- #
    def add_primitive(self, primitive: Callable, in_types: list,
                      ret_type: type, name: str = None) -> None:
        """
        Adds a primitive to the set.

        :param primitive: Callable object or a function.
        :param in_types: List of primitives arguments' type.
        :param ret_type: Type returned by the primitive.
        :param name: Alternative name for the primitive
            instead of its __name__ attribute.
        :return: Nothing.
        """
        if name in self.context:
            raise ValueError(
                f'Primitives are required to have a unique name. '
                f'Consider using the argument \'name\' to '
                f'rename your second \'{name}\' primitive.'
            )

        if name is None:
            name = primitive.__name__
        prim = Primitive(name, in_types, ret_type)

        self._add_prim(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    # -------------------------------------------------------- #
    def add_terminal(self, terminal: Callable, ret_type: type,
                     name: str = None) -> None:
        """
        Adds a terminal to the set.

        :param terminal: Callable object or a function.
        :param ret_type: Type returned by the terminal.
        :param name: Alternative name for the terminal
            instead of its __name__ attribute.
        :return: Nothing.
        """
        if name in self.context:
            raise ValueError(
                f'Terminals are required to have a unique name. '
                f'Consider using the argument \'{name}\' to '
                f'rename your second \'{name}\' terminal.'
            )
        symbolic = False
        if name is None and callable(terminal):
            name = terminal.__name__

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            self.context[str(terminal)] = terminal

        prim = Terminal(terminal, symbolic, ret_type)
        self._add_prim(prim)
        self.terms_count += 1

    # -------------------------------------------------------- #
    def add_ephemeral_constant(self, name: str, ephemeral: Callable,
                               ret_type: type) -> None:
        """
        Adds an ephemeral constant to the set. An ephemeral constant
        is a function without arguments that returns a random value.
        The value is immutable, but unique for each Tree.

        :param ephemeral: Callable object or a function.
        :param ret_type: Type returned by the ephemeral.
        :param name: Name of this ephemeral type.
        :return: Nothing.
        """
        module_gp = globals()
        if name not in module_gp:
            attrs = {'func': staticmethod(ephemeral), 'ret': ret_type}
            class_ = type(name, (Ephemeral,), attrs)
            module_gp[name] = class_
        else:
            class_ = module_gp[name]
            if issubclass(class_, Ephemeral):
                if class_.func is not ephemeral:
                    raise TypeError(
                        'Ephemera with different functions should be '
                        'named differently even between psets.'
                    )
                elif class_.ret is not ret_type:
                    raise TypeError(
                        'Ephemera with the same name and function should '
                        'have the same type even between psets.'
                    )
            else:
                raise TypeError(
                    'Ephemera should be named differently '
                    'than classes defined in the gp module.'
                )

        self._add_prim(class_)
        self.terms_count += 1

    # -------------------------------------------------------- #
    def add_adf(self, prim_set: PrimitiveSetTyped) -> None:
        """
        Adds an Automatically Defined Function (ADF) to the set.

        :param prim_set: PrimitiveSetTyped instance containing
            the primitives with which the ADF can be built.
        :return: Nothing.
        """
        prim = Primitive(
            prim_set.name,
            prim_set.ins,
            prim_set.ret
        )
        self._add_prim(prim)
        self.prims_count += 1

    # -------------------------------------------------------- #
    def rename_arguments(self, **kwargs) -> None:
        """
        Renames the arguments in self with new names from *kwargs*.

        :param kwargs: Dictionary of new names for the arguments.
        :return: Nothing.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kwargs:
                new_name = kwargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    # -------------------------------------------------------- #
    @property
    def terminal_ratio(self):
        """
        The ratio between the quantity of terminals and the
        quantity of all the other kinds of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


# ====================================================================================== #
class PrimitiveSet(PrimitiveSetTyped):
    """
    Subclass of 'PrimitiveSetTyped' without the type definition.

    :param name: The name of the primitive set.
    :param arity: The arity of the primitive set.
    :param prefix: The prefix of the primitive set.
    """
    # -------------------------------------------------------- #
    def __init__(self, name: str, arity: int, prefix: str = "ARG"):
        args = [object] * arity
        super().__init__(name, args, object, prefix)

    # -------------------------------------------------------- #
    def add_primitive(self, primitive: Callable, arity: int, name: str = None, *_, **__) -> None:
        if not arity >= 1:
            raise ValueError('arity should be >= 1')
        args = [object] * arity
        super().add_primitive(primitive, args, object, name)

    # -------------------------------------------------------- #
    def add_terminal(self, terminal: Any, name: str = None, *_, **__) -> None:
        super().add_terminal(terminal, object, name)

    # -------------------------------------------------------- #
    def add_ephemeral_constant(self, name: str, ephemeral: Callable, *_, **__) -> None:
        super().add_ephemeral_constant(name, ephemeral, object)


# ====================================================================================== #
class PrimitiveTree(list):
    """
    Tree specifically formatted for the optimization of genetic programming
    operations. This class is a subclass of *'list'*, where the nodes are appended,
    or are assumed to have been appended, when creating an object of this class
    with a list of primitives and terminals. The nodes appended to the tree are
    required to have the *'arity'* attribute, which defines the arity of the primitive.

    :param content: List of primitives and terminals to be added to the tree.
    """
    # -------------------------------------------------------- #
    def __init__(self, content: Iterable):
        super().__init__(content)

    # -------------------------------------------------------- #
    def __deepcopy__(self, memo: dict):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    # -------------------------------------------------------- #
    def __setitem__(self, key, val):
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError(
                    "Trying to set a slice larger than the size "
                    "of the PrimitiveTree is not allowed."
                )
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError(
                    "Insertion of a subtree with an arity smaller "
                    "than the PrimitiveTree is not allowed."
                )
        elif val.arity != self[key].arity:
            raise ValueError(
                "PrimitiveTree node replacement with a node "
                "of a different arity is not allowed."
            )
        list.__setitem__(self, key, val)

    # -------------------------------------------------------- #
    def __str__(self):
        string = str()
        stack = list()
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break
                stack[-1][1].append(string)
        return string

    # -------------------------------------------------------- #
    @classmethod
    def from_string(cls, string: str,
                    prim_set: PrimitiveSetTyped) -> PrimitiveTree:
        """
        Converts a string expression into a PrimitiveTree given a
        PrimitiveSet **p_set**. The primitive set needs to contain
        every primitive present in the expression.

        :param string: String representation of a Python expression.
        :param prim_set: Primitive set from which primitives are selected.
        :return: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = list()
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                ret_type = ret_types.popleft()
            else:
                ret_type = None

            if token in prim_set.mapping:
                primitive = prim_set.mapping[token]
                if ret_type is not None and not issubclass(primitive.ret, ret_type):
                    raise TypeError(
                        f'Primitive {primitive} return type {primitive.ret} '
                        f'does not match the expected one: {ret_type}.'
                    )
                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError(f'Unable to evaluate terminal: {token}.')
                if ret_type is None:
                    ret_type = type(token)
                if not issubclass(type(token), ret_type):
                    raise TypeError(
                        f'Terminal {token} type {type(token)} does '
                        f'not match the expected one: {ret_type}.'
                    )
                prim = Terminal(token, False, ret_type)
                expr.append(prim)
        return cls(expr)

    # -------------------------------------------------------- #
    def search_subtree(self, begin: int) -> slice:
        """
        Returns a slice object that corresponds to the
        range of values that defines the subtree which
        has the element with index 'begin' as its root.

        :param begin: Index of the root of the subtree.
        :return: Slice object that corresponds to the range
            of values that defines the subtree.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)

    # -------------------------------------------------------- #
    @property
    def height(self):
        """
        The height of the tree or the depth of the deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth

    # -------------------------------------------------------- #
    @property
    def root(self):
        """
        The root of the tree (element 0 in the list).
        """
        return self[0]
