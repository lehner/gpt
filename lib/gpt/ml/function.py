#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g
from gpt.ad.reverse.util import container, get_container, is_node
from gpt.ad.reverse.node import node_base


def _check_name(name):
    # a slot name; "." separates the levels of composed functions and the
    # elements of list slots (U.0, U.1, ...), so it may not appear in a name,
    # and pure integers are reserved for list elements
    if not isinstance(name, str) or name == "":
        raise ValueError(f"Invalid name {name!r}: must be a non-empty string")
    if "." in name:
        raise ValueError(f"Invalid name {name!r}: '.' is reserved")
    if name.isdigit():
        raise ValueError(f"Invalid name {name!r}: integers are reserved for list elements")


def _check_names(names):
    for name in names:
        _check_name(name)
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate names in {names}")


def _has_node(x):
    if isinstance(x, list):
        return any(_has_node(y) for y in x)
    return is_node(x)


def _promote(x):
    # a plain value as a constant node (as the AD's nodify does)
    if isinstance(x, list):
        return [_promote(y) for y in x]
    return x if is_node(x) else node_base(x, with_gradient=False)


def _container_of(x):
    # the type of a value without evaluating it (a computed node has no value
    # until its graph runs, but always a container)
    if is_node(x):
        return x._container
    if isinstance(x, list):
        elem = [_container_of(y) for y in x]
        if len(elem) == 0 or any(not e.accumulate_compatible(elem[0]) for e in elem[1:]):
            raise TypeError(f"Not a uniform list: {[str(e) for e in elem]}")
        return container(list, elem[0], len(elem))
    return get_container(x)


class _storage_list(list):
    # a write-through list (the storage of a composite, g.ml.fields): a
    # fixed-length list whose element k lives in another list, entry k =
    # (storage list, index).  Reads and
    # writes go to that storage, so any number of composites share the values
    # of their functions (optimizers replace numbers in the list they are
    # given).  It is a list (GPT checks isinstance(x, list)); the list's own
    # buffer stays empty, and every list method that would use it raises.
    def __init__(self, entries):
        super().__init__()
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return (s[i] for s, i in self.entries)

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[j] for j in range(*k.indices(len(self)))]
        s, i = self.entries[k]
        return s[i]

    def __setitem__(self, k, value):
        s, i = self.entries[k]
        s[i] = value

    def __contains__(self, x):
        return any(x is y for y in self)

    def __bool__(self):
        return len(self.entries) > 0

    def __repr__(self):
        return repr(list(self))

    def _fixed(self, *args, **kwargs):
        raise TypeError("a write-through list has a fixed layout")

    append = extend = insert = pop = remove = clear = sort = reverse = _fixed
    __delitem__ = __iadd__ = __imul__ = _fixed
    copy = index = count = __add__ = __radd__ = __mul__ = __rmul__ = __reversed__ = _fixed
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _fixed
    __hash__ = None


def fields(*lists):
    """One write-through list over several lists (e.g. the links, other fields
    and a function's parameters()): reading and writing an element reads and
    writes the list it comes from, so an optimizer working on the result
    updates the function's storage."""
    entries = []
    for values in lists:
        if isinstance(values, _storage_list):
            entries += values.entries
        else:
            entries += [(values, j) for j in range(len(values))]
    return _storage_list(entries)


class _named_storage:
    # named slots, each holding a value or a list of values, stored in one
    # flat list (the list elements of a slot U are named U.0, U.1, ...)
    def __init__(self, slots):
        _check_names([name for name, _ in slots])
        self.slots = []
        self.values = []
        self.names = []
        for name, value in slots:
            if isinstance(value, list):
                self.slots.append((name, len(self.values), len(value)))
                for i, v in enumerate(value):
                    self.values.append(v)
                    self.names.append(f"{name}.{i}")
            else:
                self.slots.append((name, len(self.values), None))
                self.values.append(value)
                self.names.append(name)

    @classmethod
    def shared(cls, names, entries):
        # single-element slots with dotted names whose values live in other
        # storages (a composite's), see _storage_list
        r = cls([])
        r.slots = [(name, i, None) for i, name in enumerate(names)]
        r.names = list(names)
        r.values = _storage_list(entries)
        return r

    def group(self, values):
        # flat list -> one value (or list) per slot
        assert len(values) == len(self.values)
        return [
            values[offset] if n is None else values[offset : offset + n]
            for _, offset, n in self.slots
        ]

    def find(self, name):
        # -> (offset, n): n is None for a single element, else a list slot
        for slot, offset, n in self.slots:
            if slot == name:
                return offset, n
        if name in self.names:
            return self.names.index(name), None
        return None


def _assign(values, i, value):
    # in place for lattices and tensors (keeps the object), else replace
    if isinstance(values[i], (g.lattice, g.tensor)):
        values[i] @= value
    else:
        values[i] = value


class function:
    """A function of named inputs to named outputs, with named parameters it
    owns (trained) and named constants (fixed, e.g. calibrated references).

    Subclasses call __init__ with the slots and implement

      evaluate(self, inputs, parameters, constants) -> list of outputs
      initialize(self, rng)

    evaluate receives one value per slot (a list for a list slot) and must
    work on plain values and on nodes of any depth.  If any input or
    parameter is a node, all of them (and the constants) are passed as
    nodes, plain values as constant nodes.
    Inputs and outputs are declared as names or (name, type) pairs, with the
    type an AD container or a representative value (None: unchecked).
    Parameters and constants are (name, value) pairs with the allocated
    storage; a list value makes a list slot whose elements are named
    name.0, name.1, ...  Values are numbers, numpy arrays, lattices or
    tensors; their otype selects the update (g.group.compose) of a
    parameter, numbers and arrays are complex additive.  A real parameter is
    expressed in evaluate by g.component.real(p), so that no gradient flows
    into its imaginary part.
    """

    def __init__(self, inputs, outputs, parameters, constants=[]):
        self._inputs = [self._typed_slot(x) for x in inputs]
        self._outputs = [self._typed_slot(x) for x in outputs]
        for slots in [self._inputs, self._outputs]:
            _check_names([n for n, _ in slots])
        self._parameters = _named_storage(parameters)
        self._constants = _named_storage(constants)
        common = set(self._parameters.names) & set(self._constants.names)
        if common:
            raise ValueError(f"Names used for parameters and constants: {sorted(common)}")

    def _typed_slot(self, x):
        # a declared type: None (unchecked), an AD container, or a
        # representative value (lattice, tensor, numpy array, number, or a
        # list of these)
        name, t = x if isinstance(x, tuple) else (x, None)
        return name, t if t is None or isinstance(t, container) else _container_of(t)

    # names
    def input_names(self):
        return [n for n, _ in self._inputs]

    def output_names(self):
        return [n for n, _ in self._outputs]

    def parameter_names(self):
        return list(self._parameters.names)

    def constant_names(self):
        return list(self._constants.names)

    # storage: the live flat lists (an optimizer updates parameters() in place)
    def parameters(self):
        return self._parameters.values

    def constants(self):
        return self._constants.values

    def _lookup(self, name):
        for storage in [self._parameters, self._constants]:
            r = storage.find(name)
            if r is not None:
                return storage, r
        raise KeyError(f"No parameter or constant {name!r}")

    def __getitem__(self, name):
        storage, (offset, n) = self._lookup(name)
        if n is None:
            return storage.values[offset]
        return storage.values[offset : offset + n]

    def __setitem__(self, name, value):
        storage, (offset, n) = self._lookup(name)
        if n is None:
            n, value = 1, [value]
        assert isinstance(value, list) and len(value) == n
        for i, v in enumerate(value):
            _assign(storage.values, offset + i, v)

    # to be implemented by subclasses
    def initialize(self, rng):
        raise NotImplementedError()

    def evaluate(self, inputs, parameters, constants):
        raise NotImplementedError()

    def calibrate(self, samples):
        # data-dependent constants from samples (each a list of plain inputs);
        # default: nothing to calibrate
        pass

    def __call__(self, inputs, parameters=None, name=None):
        # parameters: a flat list in the order of parameter_names(), e.g.
        # node leaves for a training graph; default: the owned values.
        # Called on symbols (see g.ml.symbols), the call is recorded instead:
        # parameters is then a dict {slot: symbol} of connected parameter
        # slots, and name (default: the class name) names the call.
        from gpt.ml.graph import is_symbolic, record_call

        if is_symbolic(inputs, parameters):
            return record_call(self, inputs, parameters, name)
        if name is not None:
            raise ValueError("A name is only given to symbolic calls")
        if len(inputs) != len(self._inputs):
            raise ValueError(f"Expected {len(self._inputs)} inputs, got {len(inputs)}")
        if parameters is None:
            parameters = self._parameters.values
        elif len(parameters) != len(self._parameters.values):
            raise ValueError(
                f"Expected {len(self._parameters.values)} parameters, got {len(parameters)}"
            )

        # node mode: types are checked when a graph is built (plain
        # evaluations are assumed to have been checked then), and evaluate
        # sees nodes only (plain values become constant nodes, so that
        # products need no operand ordering)
        constants = self._constants.values
        check = _has_node(inputs) or _has_node(parameters)
        if check:
            for (name, t), x in zip(self._inputs, inputs):
                self._check_type("input", name, t, x)
            for name, x, ref in zip(self._parameters.names, parameters, self._parameters.values):
                self._check_type("parameter", name, _container_of(ref), x)
            inputs = [_promote(x) for x in inputs]
            parameters = [_promote(x) for x in parameters]
            constants = [_promote(x) for x in constants]

        outputs = self.evaluate(
            list(inputs),
            self._parameters.group(parameters),
            self._constants.group(constants),
        )

        if len(outputs) != len(self._outputs):
            raise ValueError(f"Expected {len(self._outputs)} outputs, got {len(outputs)}")
        if check:
            for (name, t), x in zip(self._outputs, outputs):
                self._check_type("output", name, t, x)
        return outputs

    def _check_type(self, kind, name, t, x):
        if t is None:
            return
        c = _container_of(x)
        if c != t:
            raise TypeError(f"{self.__class__.__name__}: {kind} {name!r} is {c}, expected {t}")
