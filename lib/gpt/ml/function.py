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
import numpy as np
from gpt.ad.reverse.util import constant, container, get_container, is_node
from gpt.ad.reverse.primitive import has_node


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


def _promote(x):
    # a plain value as a constant node (as the AD's nodify does)
    if isinstance(x, list):
        return [_promote(y) for y in x]
    return constant(x)


def _box(v):
    # a number is stored as a 0-d array (of its dtype), so that every value
    # of a function is updated in place and never replaced
    return np.array(v) if g.util.is_num(v) else v


def _unbox(x):
    # the value a stored (or passed) value has in evaluate: a number for a
    # 0-d array, plain or as a node (linear.element)
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return x.item()
    if is_node(x) and x._container.tag[0] is np.ndarray and x._container.tag[1] == ():
        return x[()]
    return x


class _values(list):
    # the values of a function or composite: fixed objects, updated in place
    # (f[name] = value, the optimizers), so that composites share them by
    # holding the same objects.  Replacing an element raises (augmented
    # assignments such as x[i] @= y reassign the same object, which is
    # allowed); a list (GPT checks isinstance(x, list)).
    def __setitem__(self, k, value):
        if isinstance(k, slice) or value is not list.__getitem__(self, k):
            raise TypeError(
                "the values of a g.ml function are updated in place (f[name] = value), "
                "not replaced"
            )

    def _fixed(self, *args, **kwargs):
        raise TypeError("the values of a g.ml function have a fixed layout")

    append = extend = insert = pop = remove = clear = sort = reverse = _fixed
    __delitem__ = __iadd__ = __imul__ = _fixed


class _named_storage:
    # named slots, each holding a value or a list of values, stored in one
    # flat list (the list elements of a slot U are named U.0, U.1, ...)
    def __init__(self, slots):
        _check_names([name for name, _ in slots])
        self.slots = []
        values = []
        self.names = []
        for name, value in slots:
            if isinstance(value, list):
                self.slots.append((name, len(values), len(value)))
                for i, v in enumerate(value):
                    values.append(_box(v))
                    self.names.append(f"{name}.{i}")
            else:
                self.slots.append((name, len(values), None))
                values.append(_box(value))
                self.names.append(name)
        self.values = _values(values)

    @classmethod
    def shared(cls, names, values):
        # single-element slots with dotted names holding the values of other
        # storages (a composite's)
        r = cls([])
        r.slots = [(name, i, None) for i, name in enumerate(names)]
        r.names = list(names)
        r.values = _values(values)
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
    # in place (the object is kept, see _values); arrays (and numbers, stored
    # as 0-d arrays) keep their shape
    x = values[i]
    if isinstance(x, np.ndarray):
        if np.shape(value) != x.shape:
            raise ValueError(f"array of shape {np.shape(value)} assigned to {x.shape}")
        x[...] = value
    else:
        x @= value


def _same_type(a, b):
    # loaded values may live on new (equal) grid objects: compare layouts; a
    # number is a 0-d array
    a, b = _box(a), _box(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape
    if isinstance(a, g.lattice) and isinstance(b, g.lattice):
        return a.grid.describe() == b.grid.describe() and a.otype.__name__ == b.otype.__name__
    if isinstance(a, g.tensor) and isinstance(b, g.tensor):
        return a.otype.__name__ == b.otype.__name__
    return False


def _type_text(x):
    if isinstance(x, g.lattice):
        return f"lattice({x.otype.__name__}, {x.grid.describe()})"
    if isinstance(x, np.ndarray):
        return "number" if x.ndim == 0 else f"array{x.shape}"
    return type(x).__name__


class function:
    """A function of named inputs to named outputs, with named parameters it
    owns (trained) and named constants (fixed, e.g. calibrated references).

    Subclasses call __init__ with the slots and implement

      evaluate(self, inputs, parameters, constants) -> list of outputs
      initialize(self, rng, scale=None)   (if it has parameters)

    evaluate receives one value per slot (a list for a list slot) and must
    work on plain values and on nodes.  If any input or
    parameter is a node, all of them (and the constants) are passed as
    nodes, plain values as constant nodes.
    Inputs and outputs are declared as names or (name, type) pairs, with the
    type an AD container or a representative value (None: unchecked).
    Parameters and constants are (name, value) pairs with the allocated
    storage; a list value makes a list slot whose elements are named
    name.0, name.1, ...  Values are numbers, numpy arrays, lattices or
    tensors; their otype selects the update (g.group.compose) of a
    parameter, numbers and arrays are complex additive.  Values are updated
    in place, never replaced (numbers are stored as 0-d arrays and passed to
    evaluate as numbers), so composites share them.  A real parameter is
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
        return name, t if t is None or isinstance(t, container) else get_container(t)

    # names
    def input_names(self):
        return [n for n, _ in self._inputs]

    def output_names(self):
        return [n for n, _ in self._outputs]

    def parameter_names(self):
        return list(self._parameters.names)

    def constant_names(self):
        return list(self._constants.names)

    # storage: the flat lists of the value objects (updated in place by
    # assignments f[name] = value and by the optimizers; see _values)
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
        # the value (a number for a number, else the live object)
        storage, (offset, n) = self._lookup(name)
        if n is None:
            return _unbox(storage.values[offset])
        return [_unbox(x) for x in storage.values[offset : offset + n]]

    def __setitem__(self, name, value):
        storage, (offset, n) = self._lookup(name)
        if n is None:
            n, value = 1, [value]
        assert isinstance(value, list) and len(value) == n
        for i, v in enumerate(value):
            _assign(storage.values, offset + i, v)

    # serialization: g.save(filename, f.state()) and f.set_state(g.load(filename))
    def describe(self):
        return type(self).__name__

    def state(self):
        # the live values (no copies) by name, and the graph as text
        return {
            "parameters": dict(zip(self.parameter_names(), self.parameters())),
            "constants": dict(zip(self.constant_names(), self.constants())),
            "graph": self.describe(),
        }

    def set_state(self, state, strict=True):
        # assigns the values of state by name (lattices and tensors in place).
        # strict: the names must agree exactly, otherwise the names present in
        # both are assigned; the types must agree in any case.  A different
        # graph text only warns.  (A prefix for loading the state of a
        # sub-network into a composite, e.g. "inner.", may be added later.)
        for kind, names in [
            ("parameters", self.parameter_names()),
            ("constants", self.constant_names()),
        ]:
            values = state.get(kind, {})
            missing = [n for n in names if n not in values]
            unknown = [n for n in values if n not in names]
            if strict and (missing or unknown):
                raise KeyError(f"set_state: {kind} missing {missing}, unknown {unknown}")
            for name in names:
                if name in values and not _same_type(self[name], values[name]):
                    raise TypeError(
                        f"set_state: {name} is {_type_text(values[name])}, expected {_type_text(self[name])}"
                    )
        for kind, names in [
            ("parameters", self.parameter_names()),
            ("constants", self.constant_names()),
        ]:
            values = state.get(kind, {})
            for name in names:
                if name in values:
                    self[name] = values[name]
        if "graph" in state and state["graph"] != self.describe():
            g.message("set_state: warning: the graph differs from the one the state was saved from")

    # to be implemented by subclasses
    def initialize(self, rng, scale=None):
        # set the parameters (convention: near the identity / neutral
        # element, scale = the distance from it, None = the default);
        # nothing to do for a function without parameters
        if len(self._parameters.values) > 0:
            raise NotImplementedError()

    def evaluate(self, inputs, parameters, constants):
        raise NotImplementedError()

    def calibrate(self, samples):
        # data-dependent constants from samples (each a list of plain inputs);
        # default: nothing to calibrate
        pass

    def __call__(self, inputs, parameters=None, name=None, label=None):
        # parameters: a flat list in the order of parameter_names(), e.g.
        # node leaves for a training graph; default: the owned values.
        # Called on symbols (see g.ml.symbols), the call is recorded instead:
        # parameters is then a dict {slot: symbol} of connected parameter
        # slots, name (default: the class name) names the call, and label
        # (optional, may contain matplotlib mathtext) is shown by draw().
        from gpt.ml.graph import is_symbolic, record_call

        if is_symbolic(inputs, parameters):
            return record_call(self, inputs, parameters, name, label)
        if name is not None or label is not None:
            raise ValueError("A name or label is only given to symbolic calls")
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
        # sees nodes only (plain values become constant nodes: one kind of
        # value inside evaluate)
        # numbers are stored as 0-d arrays and enter evaluate as numbers
        constants = [_unbox(x) for x in self._constants.values]
        check = has_node(inputs) or has_node(parameters)
        parameters = [_unbox(x) for x in parameters]
        if check:
            for (name, t), x in zip(self._inputs, inputs):
                self._check_type("input", name, t, x)
            for name, x, ref in zip(self._parameters.names, parameters, self._parameters.values):
                self._check_type("parameter", name, get_container(_unbox(ref)), x)
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
        c = get_container(x)
        if c != t:
            raise TypeError(f"{self.__class__.__name__}: {kind} {name!r} is {c}, expected {t}")
