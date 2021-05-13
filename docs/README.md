#### Auto-generate the rst Files
This step must be done each time a new module is implemented.

```
sphinx-apidoc -o <OUTPUT-PATH> <MODULE-PATH> ../*run* ../*test_*
```

This command will exclude the run and test modules from documentation.
