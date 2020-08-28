# Guidelines for contributing to gpt
- A pull request (PR) should never decrease the test coverage.  If a feature is added, the corresponding code should be used in at least one of the tests of [tests/run](https://github.com/lehner/gpt/blob/master/tests/run).
- A good test checks the expected behavior of the function and does not *only* compare to precomputed values.  A good example of this is the test of the covariant gaussian smearing operator [tests/smear/smear.py](https://github.com/lehner/gpt/blob/master/tests/create/smear.py) which first tests covariance and then tests the expected eigenspace of the operator.
- A good test should avoid to use external files.  This may not be possible if a new file format is implemented. 
  In this case, the [gpt-repository](https://github.com/lehner/gpt-repository) should be used to store such files.  The file sizes for such tests should be kept minimal.
- It is sufficient and in many cases preferable to add a high-level test of functionality that internally uses all of the added lines of code. 
  The interplay between tests and code stability is explained also below.
- A new feature needs to address an immediate need, i.e., code that may not be used for a long time should not be added at this time.  A corollary of this is that a good PR removes lines of code instead of adding them (without affecting the functionality).
- In order to adhere to our Python code style, run [scripts/black](https://github.com/lehner/gpt/blob/master/scripts/black) before creating the PR.
- Class, function, and variable names should be written in snake_case.
- Abbreviations such as *repr* for *representation* when *defining* classes, functions, and variables should be avoided.  Abbreviations for algorithm names may be acceptable if they are frequently used in the literature (such as CG for Conjugate Gradient).
  It is acceptable to define shortcuts when *referring* to classes, functions, and variables such as
```python
inv = gpt.algorithms.inverter
cg = inv.cg
```
- Different parts of the code are expected to be stable to different degrees:
  - Frequent changes to cgpt are expected and may be needed, e.g., due to performance improvements or to enable new hardware platforms.
  - The implementation of functions in gpt may also change on a frequent basis to add new features or make the code faster or more maintainable.
  - The gpt interface should only change if needed.  The interface is defined implicitly through the [tests](https://github.com/lehner/gpt/blob/master/tests).  
  The preferred way to indicate that an interface design choice is intended to be stable is to add it to the tests.

These guidelines may be violated in the short term by different parts of the code base, however, we shall try to remove such violations as we progress.
