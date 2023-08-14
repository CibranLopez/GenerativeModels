# Tests

This folder contains tests for the different routines implemented in the project, namely:

- XDATCAR reading (SPR).
- GCNN implementation.
- Verification functions.
- XDATCAR dumping (SPR).

In order to find all tests from the home directory of the project (parent directory of tests) just run:

```bash
python3 -m unittest discover -v
```

and to run, for instance xx tests, execute:

```bash
python3 -m unittest tests.test_xx -v
```
