# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Pattern matcher for TensorFlow graphs in the Python object model.

Writing Python to crawl through TensorFlow graphs can be a pain, and the
resulting code is often hard to adapt, extend, and reuse. Instead of
hand-writing that code, we should automatically generate it from a simple
pattern-matching language. This package provides one such system.

More precisely, this package defines a pattern language for matching and
extracting nodes from TensorFlow graphs as represented in the Python object
model. Patterns can be defined in Python code with a simple syntax and are
compiled into compositions of continuation-passing matcher combinators. The
mechanism for compiling the pattern language into combinators looks like an
analyzing Scheme interpreter. The design comes from GJS's 6.945 at MIT.

The pattern language compiler can be extended by registering new handlers at
runtime, and new pattern compilers can be made by instantiating the
PatternEvaluator class.

The grammar for the pattern language implemented in this file is:

  pattern ::= element | choice | list | internal_node | negated_pattern | any
  patterns ::= pattern, patterns | ()

  element ::= ('?', element_name, restrictions)
  element_name ::= PYTHON_STRING
  restrictions ::= PYTHON_FUNCTION, restrictions | ()

  choice ::= ('?:choice', patterns)

  list ::= ('List', patterns)

  internal_node ::= (pattern, neighbor_constraints)
  neighbor_constraints ::= input_list | output_list | input_list, output_list
  input_list ::= ('In', patterns)
  output_list ::= ('Out', patterns)

  negated_pattern ::= ('?:not', pattern)

  any ::= ('?:any',)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.util import tf_inspect

from kfac.python.ops.tensormatch import tensorflow_graph_util as util


def _any(itr):
  """Similar to Python's any, but returns the first value that matches."""
  for val in itr:
    if val:
      return val
  return False


def _all(itr):
  """Similar to Python's all, but returns the first value that doesn't match."""
  any_iterations = False
  val = None
  for val in itr:
    any_iterations = True
    if not val:
      return val
  return val if any_iterations else True


def is_seq(obj):
  return isinstance(obj, (tuple, list))


def is_nonempty_seq(obj):
  return is_seq(obj) and bool(obj)


def is_empty_seq(obj):
  return is_seq(obj) and not bool(obj)


## define the syntax of the pattern language


is_pattern = is_nonempty_seq


def is_element_pattern(pat):
  return is_pattern(pat) and pat[0] == '?'


def element_name(pat):
  return pat[1]


def element_restrictions(pat):
  return pat[2:]


def is_choice_pattern(pat):
  return is_pattern(pat) and pat[0] == '?:choice'


def choice_patterns(pat):
  return pat[1:]


def is_list_pattern(pat):
  return is_pattern(pat) and pat[0] == 'List'


def list_patterns(pat):
  return pat[1:]


def is_not_pattern(pat):
  return is_pattern(pat) and pat[0] == '?:not'


def negated_pattern(pat):
  return pat[1]


def is_any_pattern(pat):
  return is_pattern(pat) and pat[0] == '?:any'


def is_any_noconsume_pattern(pat):
  return is_pattern(pat) and pat[0] == '?:any_noconsume'


def is_internal_node_pattern(pat):
  def is_neighbor_constraints(lst):
    tags = tuple(item[0] for item in lst)
    return tags in {('In',), ('Out',), ('In', 'Out')}
  return (is_pattern(pat) and all(is_pattern(item) for item in pat)
          and is_neighbor_constraints(pat[1:]))


def internal_node_pattern(pat):
  return pat[0]


def internal_node_input_pattern(pat):
  for item in pat[1:]:
    if item[0] == 'In':
      return ('List',) + tuple(item[1:])
  return ('?:any_noconsume',)


def internal_node_output_pattern(pat):
  for item in pat[1:]:
    if item[0] == 'Out':
      return ('List',) + tuple(item[1:])
  return ('?:any_noconsume',)


def internal_patterns(pat):
  return [internal_node_pattern(pat), internal_node_input_pattern(pat),
          internal_node_output_pattern(pat)]


## constructors for pattern-matching combinators


def match_eqv(pattern):
  def eqv_match(data, bindings, consumed, succeed):
    return data == pattern and succeed(bindings, consumed | {data})
  return eqv_match


def match_any(data, bindings, consumed, succeed):
  try:
    consumed = consumed | {data}  # pylint: disable=g-no-augmented-assignment
  except TypeError:
    consumed = consumed | set(data)  # pylint: disable=g-no-augmented-assignment
  return succeed(bindings, consumed)


def match_any_noconsume(data, bindings, consumed, succeed):  # pylint: disable=unused-argument
  # this combinator succeeds (but does not append to the consumed set)
  # regardless of the value of 'data', though the caller still passes 'data'
  # (since all combinators have the same signature)
  return succeed(bindings, consumed)


def match_element(variable_name, restrictions):
  """Matches an element."""
  def element_match(data, bindings, consumed, succeed):
    consumed = consumed | {data}  # pylint: disable=g-no-augmented-assignment
    if _all(restriction(data) for restriction in restrictions):
      if not variable_name:
        return succeed(bindings, consumed)
      elif variable_name in bindings:
        return bindings[variable_name] == data and succeed(bindings, consumed)
      return succeed(dict(bindings, **{variable_name: data}), consumed)
    return False
  return element_match


def match_choice(*match_combinators):
  def choice_match(data, bindings, consumed, succeed):
    return _any(matcher(data, bindings, consumed, succeed)
                for matcher in match_combinators)
  return choice_match


def match_list(*match_combinators):
  """Matches a list."""
  def list_match(data, bindings, consumed, succeed):
    return _list_match(data, match_combinators, bindings, consumed, succeed)

  def _list_match(data, matchers, bindings, consumed, succeed):
    """Apply matchers elementwise to a list, collecting bindings sequentially.

    Args:
      data: The list on which to apply the matcher list.
      matchers: The corresponding list of matchers to apply, element-by-element.
      bindings: The dictionary of bindings to be consistent with.
      consumed: The list of graph nodes consumed so far.
      succeed: The continuation function to call when there is a match.

    Returns:
      False if there is no match, or succeed(bindings) if there is one.
    """
    def match_first_then_subsequent(combinator, datum):
      return combinator(datum, bindings, consumed, match_subsequent_elements)

    def match_subsequent_elements(bindings, consumed):
      return _list_match(data[1:], matchers[1:], bindings, consumed, succeed)

    if is_empty_seq(matchers) and is_empty_seq(data):
      return succeed(bindings, consumed)
    return (is_nonempty_seq(matchers) and is_nonempty_seq(data)
            and match_first_then_subsequent(matchers[0], data[0]))
  return list_match


def match_not(match_combinator):
  def not_match(data, bindings, consumed, succeed):
    return (not match_combinator(data, bindings, set(),
                                 lambda bindings, _: True)
            and succeed(bindings, consumed))
  return not_match


def match_internal(*match_combinators):
  expanded_matcher = match_list(*match_combinators)
  def internal_node_match(data, bindings, consumed, succeed):
    try:
      expanded = [data, util.expand_inputs(data), util.expand_outputs(data)]
    except ValueError:
      return False
    return expanded_matcher(expanded, bindings, consumed, succeed)
  return internal_node_match


## parsing the pattern language into compositions of combinators


class PatternEvaluator(object):
  """Pattern evaluator class."""

  def __init__(self, default_operation=None):
    self.default_operation = default_operation
    self.handlers = []

  def defhandler(self, predicate, handler):
    self.handlers.append((predicate, handler))

  def __call__(self, pat):
    for predicate, handler in self.handlers:
      if predicate(pat):
        return handler(pat)
    if self.default_operation:
      return self.default_operation(pat)
    raise ValueError

make_combinators = PatternEvaluator(match_eqv)
make_combinators.defhandler(
    is_element_pattern,
    lambda pat: match_element(element_name(pat), element_restrictions(pat)))
make_combinators.defhandler(
    is_list_pattern,
    lambda pat: match_list(*map(make_combinators, list_patterns(pat))))
make_combinators.defhandler(
    is_choice_pattern,
    lambda pat: match_choice(*map(make_combinators, choice_patterns(pat))))
make_combinators.defhandler(
    is_not_pattern,
    lambda pat: match_not(make_combinators(negated_pattern(pat))))
make_combinators.defhandler(
    is_any_pattern,
    lambda pat: match_any)
make_combinators.defhandler(
    is_any_noconsume_pattern,
    lambda pat: match_any_noconsume)
make_combinators.defhandler(
    is_internal_node_pattern,
    lambda pat: match_internal(*map(make_combinators, internal_patterns(pat))))


## utility function so the patterns require fewer parentheses


def expand_thunks(pat):
  """Expands thunks (zero-argument functions) in a pattern by calling them.

  Args:
    pat: The pattern to expand, possibly containing thunks.

  Returns:
    The expanded pattern.
  """
  def is_thunk(x):
    if hasattr(x, '__call__'):
      spec = tf_inspect.getargspec(x)
      num_free_args = len(set(spec.args)) - len(set(spec.defaults or {}))
      return num_free_args == 0
    return False
  while is_thunk(pat):
    pat = pat()
  if isinstance(pat, (tuple, list)):
    return type(pat)(map(expand_thunks, pat))
  return pat


## main matcher interface functions


def matcher(pattern):
  combinators = make_combinators(expand_thunks(pattern))
  def match(node):
    return combinators(node, {}, set(), lambda bindings, _: bindings or True)
  return match


def all_matcher(pattern):
  combinators = make_combinators(expand_thunks(pattern))
  results = []

  def all_matches(node):
    combinators(node, {}, set(),
                lambda bindings, _: results.append(bindings or True))
    return results

  return all_matches


def matcher_with_consumed(pattern):
  combinators = make_combinators(expand_thunks(pattern))
  def match(node):
    return combinators(node, {}, set(),
                       lambda bindings, consumed: (bindings, consumed))
  return match
