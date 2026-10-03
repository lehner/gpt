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
#
# Composite functions from symbolic calls:
#
#   x1, x2 = g.ml.symbols("x1", "x2")
#   a, = f1([x1], name="enc")
#   b, z = f2([x1, x2])                       # call named "f2" (class name)
#   y, = f3([a, b], parameters={"w": z})      # f3's parameter w computed by f2
#   net = g.ml.pack(y=y, z=z).function()      # inputs [x1, x2]
#
# A function called on symbols records the call and returns symbols for its
# outputs.  pack names the outputs; function() collects the calls the outputs
# depend on and returns a g.ml.composite.
#
import itertools
from gpt.ml.function import function, _check_name, _check_names, _named_storage, _storage_list

_creation = itertools.count()


class _symbol:
    # a free symbol (an input; call is None) or output index of a call
    def __init__(self, name=None, call=None, index=None):
        self.name = name
        self.call = call
        self.index = index
        self.id = next(_creation)

    def __repr__(self):
        return f"symbol({self.label()})"

    def label(self):
        # the input's name, or call.output
        if self.call is None:
            return self.name
        return f"{self.call.name}.{self.call.function.output_names()[self.index]}"

    def describe(self):
        return pack._of([self]).describe()

    def draw(self, ax=None):
        return pack._of([self]).draw(ax)


class _call:
    def __init__(self, name, function, inputs, connections):
        self.name = name
        self.function = function
        self.inputs = inputs
        # flat parameter index of the function -> (symbol, list element or None)
        self.connections = connections
        self.id = next(_creation)
        self.outputs = [_symbol(call=self, index=i) for i in range(len(function.output_names()))]


def symbols(*names):
    _check_names(list(names))
    return [_symbol(name=n) for n in names]


def is_symbolic(inputs, parameters):
    return isinstance(parameters, dict) or any(isinstance(x, _symbol) for x in inputs)


def record_call(f, inputs, parameters, name):
    name = type(f).__name__ if name is None else name
    _check_name(name)
    parameters = {} if parameters is None else parameters
    if not isinstance(parameters, dict):
        raise TypeError(f"{name}: parameters of a symbolic call are a dict {{slot: symbol}}")
    if len(inputs) != len(f.input_names()):
        raise ValueError(f"{name}: expected {len(f.input_names())} inputs, got {len(inputs)}")
    for s in list(inputs) + list(parameters.values()):
        if not isinstance(s, _symbol):
            raise TypeError(f"{name}: symbolic calls take symbols only, got {type(s)}")
    connections = {}
    for slot, s in parameters.items():
        r = f._parameters.find(slot)
        if r is None:
            raise KeyError(f"{name}: no parameter {slot!r}")
        offset, n = r
        for j, k in [(offset, None)] if n is None else [(offset + k, k) for k in range(n)]:
            if j in connections:
                raise ValueError(f"{name}: parameter {slot!r} connected twice")
            connections[j] = (s, k)
    return _call(name, f, list(inputs), connections).outputs


def _subgraph(roots):
    # the calls the roots depend on (in order of creation, which is a
    # dependency order) and the free symbols they use (in order of creation)
    calls, free = {}, {}
    stack = list(roots)
    while stack:
        s = stack.pop()
        if s.call is None:
            free[s.id] = s
        elif s.call.id not in calls:
            calls[s.call.id] = s.call
            stack += s.call.inputs + [t for t, _ in s.call.connections.values()]
    return [calls[k] for k in sorted(calls)], [free[k] for k in sorted(free)]


class _layout:
    # the parameters and constants of a set of calls: each function once, in
    # the order of its first call, whose name prefixes its slots; a parameter
    # slot is owned if at least one call of its function leaves it
    # unconnected.  index[id(f)][j] is the position of slot j of f, prefix[id(f)]
    # the name of its first call.
    def __init__(self, calls):
        self.prefix = {}
        for c in calls:
            self.prefix.setdefault(id(c.function), c.name)
        self.functions = list({id(c.function): c.function for c in calls}.values())
        self.names, self.entries, self.index = [], [], {}
        self.constant_names, self.constant_entries = [], []
        for f in self.functions:
            prefix = self.prefix[id(f)]
            f_calls = [d for d in calls if d.function is f]
            owned = [
                j
                for j in range(len(f._parameters.values))
                if any(j not in d.connections for d in f_calls)
            ]
            self.index[id(f)] = {j: len(self.entries) + k for k, j in enumerate(owned)}
            self.names += [f"{prefix}.{f._parameters.names[j]}" for j in owned]
            self.entries += [_entry(f._parameters.values, j) for j in owned]
            self.constant_names += [f"{prefix}.{name}" for name in f._constants.names]
            self.constant_entries += [_entry(f._constants.values, j) for j in range(len(f._constants.values))]


class pack:
    """Named outputs (pack(y=a, z=b)) of a function under construction: the
    graph of the calls they depend on.  function() turns it into a
    g.ml.composite; describe() and draw() show it."""

    def __init__(self, **outputs):
        _check_names(list(outputs.keys()))
        for name, s in outputs.items():
            if not isinstance(s, _symbol):
                raise TypeError(
                    f"Output {name!r} must be a symbol, got {type(s)} (unpack calls: a, = f([x]))"
                )
        self.outputs = outputs

    @classmethod
    def _of(cls, roots):
        # an unnamed pack (for inspecting symbols), labeled by the symbols
        r = cls()
        r.outputs = {s.label(): s for s in roots}
        return r

    def _inputs(self, inputs):
        # inputs: the free symbols in the order of the function's inputs;
        # default: those the outputs depend on, in order of creation
        calls, free = _subgraph(self.outputs.values())
        if inputs is None:
            return calls, free
        for s in inputs:
            if not isinstance(s, _symbol) or s.call is not None:
                raise TypeError(f"Inputs must be free symbols, got {s!r}")
            if not any(s is t for t in free):
                raise ValueError(f"Input {s!r} is not used by the outputs")
        unbound = [s for s in free if not any(s is t for t in inputs)]
        if unbound:
            raise ValueError(f"Symbols {unbound} are used but not inputs")
        return calls, list(inputs)

    def function(self, inputs=None):
        calls, inputs = self._inputs(inputs)
        names = [c.name for c in calls]
        duplicates = sorted(set(n for n in names if names.count(n) > 1))
        if duplicates:
            raise ValueError(f"Duplicate call names {duplicates}: name the calls (name=...)")
        return composite(self, inputs, calls)

    def describe(self, inputs=None):
        """A text listing of the graph: the inputs, one line per call (its
        arguments, connected parameters after ';', and the parameters and
        constants it takes from storage, by their names in the composite),
        and the outputs."""
        calls, inputs = self._inputs(inputs)
        layout = _layout(calls)
        rows = []
        for c in calls:
            f = c.function
            args = [f"{n}={s.label()}" for n, s in zip(f.input_names(), c.inputs)]
            connected = [
                f"{f._parameters.names[j]}={s.label() if k is None else s.label() + '[' + str(k) + ']'}"
                for j, (s, k) in sorted(c.connections.items())
            ]
            call = f"{type(f).__name__}({', '.join(args)}{'; ' + ', '.join(connected) if connected else ''})"
            prefix = layout.prefix[id(f)]
            stored = [f"{prefix}.{n}" for j, n in enumerate(f._parameters.names) if j not in c.connections]
            constants = [f"{prefix}.{n}" for n in f._constants.names]
            rows.append((c.name, call, stored, constants))
        w0 = max([len(r[0]) for r in rows], default=0)
        w1 = max([len(r[1]) for r in rows], default=0)
        lines = [f"inputs: {', '.join(s.label() for s in inputs)}"]
        for name, call, stored, constants in rows:
            line = f"  {name.ljust(w0)} = {call.ljust(w1)}"
            if stored:
                line += f"  parameters: {', '.join(stored)}"
            if constants:
                line += f"  constants: {', '.join(constants)}"
            lines.append(line.rstrip())
        lines.append(
            f"outputs: {', '.join(n if n == s.label() else f'{n}={s.label()}' for n, s in self.outputs.items())}"
        )
        return "\n".join(lines)

    def draw(self, ax=None):
        """Planned: draw the graph with matplotlib (imported here only, an
        optional dependency): inputs left, outputs right, a box per call
        (nested composites as single boxes), solid input and dashed
        parameter edges labeled with their slots, a common marker for the
        calls of a shared function.  Returns the figure."""
        raise NotImplementedError("g.ml: draw is planned; use describe() for now")


class composite(function):
    """The function of a pack (see g.ml.pack).  Its parameters are the
    parameter slots of its functions that at least one call leaves
    unconnected, named call.slot with the first call of each function; its
    constants are all constants of its functions.  It holds no values: its
    storage lists point into the storages of its functions, so any number of
    composites share the values of a function.  f["call.name"] addresses the
    function of a call."""

    def __init__(self, graph, inputs, calls):
        super().__init__([s.name for s in inputs], list(graph.outputs.keys()), [])
        self._graph = graph
        self._input_symbols = inputs
        self._output_symbols = list(graph.outputs.values())
        self._calls = calls
        self._functions = {c.name: c.function for c in calls}
        self._layout = layout = _layout(calls)
        self._parameters = _named_storage.shared(layout.names, layout.entries)
        self._constants = _named_storage.shared(layout.constant_names, layout.constant_entries)

    def graph(self):
        return self._graph

    def calls(self):
        return self._calls

    def describe(self):
        return self._graph.describe(self._input_symbols)

    def draw(self, ax=None):
        return self._graph.draw(ax)

    def initialize(self, rng):
        for f in self._layout.functions:
            f.initialize(rng)

    def _lookup_function(self, name):
        call, _, rest = name.partition(".")
        if call not in self._functions or rest == "":
            raise KeyError(f"No parameter or constant {name!r}")
        return self._functions[call], rest

    def __getitem__(self, name):
        f, rest = self._lookup_function(name)
        return f[rest]

    def __setitem__(self, name, value):
        f, rest = self._lookup_function(name)
        f[rest] = value

    def _run(self, c, values, parameters):
        # evaluate call c: values maps symbol ids to values, parameters is
        # flat in the order of the composite's slots
        def value(s, k=None):
            v = values[s.id]
            return v if k is None else v[k]

        index = self._layout.index[id(c.function)]
        p = [
            value(*c.connections[j]) if j in c.connections else parameters[index[j]]
            for j in range(len(c.function._parameters.values))
        ]
        for s, y in zip(c.outputs, c.function([value(s) for s in c.inputs], p)):
            values[s.id] = y

    def evaluate(self, inputs, parameters, constants):
        values = {s.id: x for s, x in zip(self._input_symbols, inputs)}
        for c in self._calls:
            self._run(c, values, parameters)
        return [values[s.id] for s in self._output_symbols]

    def calibrate(self, samples):
        # the calls in order: each function is calibrated (at its first call)
        # on the inputs it receives from the samples, then evaluated, so later
        # functions see calibrated earlier ones
        values = [{s.id: x for s, x in zip(self._input_symbols, inputs)} for inputs in samples]
        calibrated = set()
        for c in self._calls:
            if id(c.function) not in calibrated:
                c.function.calibrate([[v[s.id] for s in c.inputs] for v in values])
                calibrated.add(id(c.function))
            for v in values:
                self._run(c, v, self.parameters())


def _entry(values, j):
    # where element j of a storage lives (a composite's entries point through)
    if isinstance(values, _storage_list):
        return values.entries[j]
    return (values, j)
