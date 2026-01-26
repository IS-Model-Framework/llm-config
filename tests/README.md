# HLO Config System Tests

This directory contains system-level tests for the `hlo_config` tool.

## Test Coverage

The test suite covers all functionality described in `docs/design_draft.md`:

### Layer Parameter Management
- ✅ Adding layer components from file (MLA, MoE)
- ✅ Adding layer components from base
- ✅ Updating layer parameters
- ✅ Deleting layer parameters

### Model/Subgraph Parameter Management
- ✅ Adding models from file
- ✅ Adding models from base with nested attributes
- ✅ Adding models with `--base-layer` option
- ✅ Updating model parameters (including nested attributes)
- ✅ Deleting model parameters

### Information Display
- ✅ Listing all models/layers
- ✅ Showing detailed information
- ✅ Showing specific attributes
- ✅ Showing nested attributes (e.g., `mlp.dim`)
- ✅ Listing attribute descriptions

### Error Handling
- ✅ Duplicate name detection
- ✅ Missing parameter validation

## Running Tests

### Run all tests:
```bash
python tests/test_system.py
```

### Run from project root:
```bash
cd /path/to/llm-config
python tests/test_system.py
```

## Test Environment

The test script:
1. **Sets database name** to `test.sqlite` for testing (default is `configs.sqlite`)
2. **Creates a clean** test environment
3. **Runs all tests** sequentially using the test database
4. **Deletes** the test database after tests complete
5. **Restores** the default database name (`configs.sqlite`)

**Note**: The test database (`test.sqlite`) is completely separate from your development database (`configs.sqlite`). Your development data will not be affected by running tests.

## Test Output

The test script provides:
- Individual test results (✓ PASSED / ✗ FAILED)
- Detailed error messages for failed tests
- Summary statistics at the end

Example output:
```
============================================================
HLO Config System Test Suite
============================================================

============================================================
Test 1: Database Initialization
============================================================
✓ PASSED: Database Initialization

...

============================================================
Test Summary
============================================================
Total Tests: 17
Passed: 15
Failed: 2

Failed Tests:
  - Add Model from Base
  - Update Model with Nested Attributes
```

## Notes

- Tests use the templates in `config/templates/` directory
- Tests create temporary configurations with names like `test-*`
- The test database is separate from your development database
- All test data is cleaned up after tests complete

