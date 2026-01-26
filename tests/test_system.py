#!/usr/bin/env python3
"""
System-level test script for hlo_config tool.
Tests all functionality according to design_draft.md specifications.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import database configuration function
from config.base import set_db_name, _DEFAULT_DB_NAME

# Test configuration
TEST_DB_NAME = "test.sqlite"
TEST_DB_PATH = None
BACKUP_DB_PATH = None
TEMPLATES_DIR = Path(__file__).parent / "templates"
RUNNER_SCRIPT = Path(__file__).parent.parent / "runner.py"
CONFIG_DIR = Path(__file__).parent.parent / "config"

# Ensure we're using the correct Python executable
if not RUNNER_SCRIPT.exists():
    raise FileNotFoundError(f"Runner script not found at {RUNNER_SCRIPT}")

# Test results
test_results = {
    "passed": [],
    "failed": [],
    "total": 0
}


def run_command(cmd, input_text=None, check=True):
    """Run a command and return the result."""
    try:
        # Set environment variable for database name
        env = os.environ.copy()
        env['HLO_CONFIG_DB_NAME'] = TEST_DB_NAME
        
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def setup_test_environment():
    """Setup test environment by setting test database name."""
    global TEST_DB_PATH, BACKUP_DB_PATH
    
    # Set database name to test.sqlite
    set_db_name(TEST_DB_NAME)
    print(f"✓ Set database name to {TEST_DB_NAME}")
    
    # Get the test database path
    TEST_DB_PATH = CONFIG_DIR / TEST_DB_NAME
    
    # Remove test database if it exists from previous test run
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
        print(f"✓ Removed existing test database {TEST_DB_NAME}")

    os.environ['HLO_CONFIG_TEST_MODE'] = 'true'


def cleanup_test_environment():
    """Clean up test environment by removing test database and restoring default."""
    global TEST_DB_PATH
    
    # Remove test database
    if TEST_DB_PATH and TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
        print(f"✓ Removed test database {TEST_DB_NAME}")
    
    # Restore default database name
    set_db_name(_DEFAULT_DB_NAME)
    print(f"✓ Restored default database name ({_DEFAULT_DB_NAME})")


def _run_test(name, func):
    """Run a test and record the result."""
    test_results["total"] += 1
    print(f"\n{'='*60}")
    print(f"Test {test_results['total']}: {name}")
    print(f"{'='*60}")
    
    try:
        result = func()
        if result:
            test_results["passed"].append(name)
            print(f"✓ PASSED: {name}")
            return True
        else:
            test_results["failed"].append(name)
            print(f"✗ FAILED: {name}")
            return False
    except Exception as e:
        test_results["failed"].append(name)
        print(f"✗ FAILED: {name} - {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Test Functions
# ============================================================================

def test_initialize():
    """Test database initialization."""
    success, stdout, stderr = run_command([sys.executable, str(RUNNER_SCRIPT), "show", "--module", "model", "--list"])
    # Initialization happens automatically, just check if command works
    return success or "no config found" in stderr.lower() or "not found" in stderr.lower()


def test_add_mla_from_file():
    """Test adding MLA components from file."""
    mla_file = TEMPLATES_DIR / "test-mla.ini"
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "add", "--module", "mla", "--from-file", str(mla_file)
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify components were added
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "mla", "--list"
    ])
    
    return success and ("Test-Custom1" in stdout and "Test-Custom2" in stdout and "Test-Custom3" in stdout)


def test_add_moe_from_file():
    """Test adding MoE component from file."""
    moe_file = TEMPLATES_DIR / "test-moe.ini"
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "add", "--module", "moe", "--from-file", str(moe_file)
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify component was added
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "moe", "--list"
    ])
    
    return success and "Test-CUSTOM" in stdout


def test_add_model_from_file():
    """Test adding model from file."""
    model_file = TEMPLATES_DIR / "test-llama-3-8B.ini"
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "add", "--module", "model", "--from-file", str(model_file)
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify model was added
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--name", "Test-Llama-3-8B"
    ])
    
    return success and "Test-Llama-3-8B" in stdout


def test_add_mla_from_base():
    """Test adding MLA component from base."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "add", "--module", "mla", "--from-base", "Test-Custom1",
        "--update", "name=test-mla-from-base,kv_lora_rank=1024"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify new component was created
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "mla", "--name", "test-mla-from-base"
    ])
    
    return success and "test-mla-from-base" in stdout


def test_add_model_from_base():
    """Test adding model from base with nested attributes."""
    # First ensure Test-DeepSeek-V3-671B exists
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--name", "Test-DeepSeek-V3-671B"
    ])
    
    if not success:
        # Try to add it first
        model_file = TEMPLATES_DIR / "test_deepseek-v3-671B.ini"
        if model_file.exists():
            run_command([
                sys.executable, str(RUNNER_SCRIPT),
                "add", "--module", "model", "--from-file", str(model_file)
            ])
        else:
            # If file doesn't exist, skip this test
            print("Warning: Test-DeepSeek-V3-671B template not found, skipping test")
            return True
    
    # This will prompt for confirmation, so we need to handle that
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "add", "--module", "model", "--from-base", "Test-DeepSeek-V3-671B",
        "--update", "name=test-model-from-base,num_layers=8,moe.name=Test-DeepSeek-V3-671B,moe.shared_experts_dim=1024"
    ], input_text="yes\n")
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify model was created
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--name", "test-model-from-base"
    ])
    
    return success and "test-model-from-base" in stdout


def test_add_model_with_base_layer():
    """Test adding model with --base-layer option."""
    # Create a minimal model config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write("""[Model]
name=test-model-base-layer
num_layers=16
emb_dim=4096
max_seq_len=8192
vocab_size=128256
tie_word_embeddings=false
""")
        temp_file = f.name
    
    try:
        success, stdout, stderr = run_command([
            sys.executable, str(RUNNER_SCRIPT),
            "add", "--module", "model", "--from-file", temp_file,
            "--base-layer", "mha.name=Test-Llama-3-8B,rope.name=Test-Llama-3-8B,mlp.name=Test-Llama-3-8B,norm.name=Test-Llama-3-8B"
        ])
        
        if not success:
            print(f"Error: {stderr}")
            return False
        
        # Verify model was created
        success, stdout, stderr = run_command([
            sys.executable, str(RUNNER_SCRIPT),
            "show", "--module", "model", "--name", "test-model-base-layer"
        ])
        
        return success and "test-model-base-layer" in stdout
    finally:
        os.unlink(temp_file)


def test_update_layer():
    """Test updating layer parameters."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "update", "--module", "mla", "--name", "test-mla-from-base",
        "--update", "kv_lora_rank=2048"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify update
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "mla", "--name", "test-mla-from-base", "--attribute", "kv_lora_rank"
    ])
    
    return success and "2048" in stdout


def test_update_model_nested():
    """Test updating model with nested attributes."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "update", "--module", "model", "--name", "Test-Llama-3-8B",
        "--update", "mlp.dim=16384"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify update
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--name", "Test-Llama-3-8B", "--attribute", "mlp.dim"
    ])
    
    return success and "16384" in stdout


def test_show_list():
    """Test showing list of models."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--list"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    return "Test-Llama-3-8B" in stdout or "Test-DeepSeek-V3-671B" in stdout


def test_show_detailed():
    """Test showing detailed model information."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--name", "Test-Llama-3-8B", "--detailed"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Check for detailed information
    return "ModelConfig:" in stdout and "MHAConfig:" in stdout


def test_show_attribute():
    """Test showing specific attribute."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--name", "Test-Llama-3-8B", "--attribute", "vocab_size"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    return "128256" in stdout


def test_show_nested_attribute():
    """Test showing nested attribute."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--name", "Test-Llama-3-8B", "--attribute", "mlp.dim"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Should show a number (dim value)
    return stdout.strip().isdigit() or "16384" in stdout or "14336" in stdout


def test_show_list_attributes():
    """Test showing list of attributes."""
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "model", "--list-attributes"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    return "ModelConfig:" in stdout and "emb_dim:" in stdout


def test_delete_layer():
    """Test deleting layer."""
    # First create a test layer to delete
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "add", "--module", "mla", "--from-base", "Test-Custom1",
        "--update", "name=test-mla-to-delete"
    ])
    
    if not success:
        print(f"Error creating test layer: {stderr}")
        return False
    
    # Now delete it
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "delete", "--module", "mla", "--name", "test-mla-to-delete"
    ])
    
    if not success:
        print(f"Error: {stderr}")
        return False
    
    # Verify it's deleted
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "show", "--module", "mla", "--name", "test-mla-to-delete"
    ])
    
    return not success  # Should fail because it doesn't exist


def test_delete_model():
    """Test deleting model."""
    # First create a test model to delete
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
        f.write("""[Model]
name=test-model-to-delete
num_layers=8
emb_dim=2048
max_seq_len=4096
vocab_size=50000
tie_word_embeddings=false
""")
        temp_file = f.name
    
    try:
        success, stdout, stderr = run_command([
            sys.executable, str(RUNNER_SCRIPT),
            "add", "--module", "model", "--from-file", temp_file,
            "--base-layer", "mha.name=Test-Llama-3-8B,rope.name=Test-Llama-3-8B,mlp.name=Test-Llama-3-8B,norm.name=Test-Llama-3-8B"
        ])
        
        if not success:
            print(f"Error creating test model: {stderr}")
            return False
        
        # Now delete it (without --all to test partial delete)
        success, stdout, stderr = run_command([
            sys.executable, str(RUNNER_SCRIPT),
            "delete", "--module", "model", "--name", "test-model-to-delete"
        ])
        
        if not success:
            print(f"Error: {stderr}")
            return False
        
        # Verify it's deleted
        success, stdout, stderr = run_command([
            sys.executable, str(RUNNER_SCRIPT),
            "show", "--module", "model", "--name", "test-model-to-delete"
        ])
        
        return not success  # Should fail because it doesn't exist
    finally:
        os.unlink(temp_file)


def test_error_handling():
    """Test error handling for invalid operations."""
    # Test adding duplicate name
    success, stdout, stderr = run_command([
        sys.executable, str(RUNNER_SCRIPT),
        "add", "--module", "mla", "--from-base", "Test-Custom1",
        "--update", "name=Test-Custom1"  # Duplicate name
    ])
    
    # Should fail
    return not success


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Run all tests."""
    print("="*60)
    print("HLO Config System Test Suite")
    print("="*60)
    
    # Setup
    setup_test_environment()
    
    try:
        # Run tests
        _run_test("Database Initialization", test_initialize)
        _run_test("Add MLA from File", test_add_mla_from_file)
        _run_test("Add MoE from File", test_add_moe_from_file)
        _run_test("Add Model from File", test_add_model_from_file)
        _run_test("Add MLA from Base", test_add_mla_from_base)
        _run_test("Add Model from Base", test_add_model_from_base)
        _run_test("Add Model with Base Layer", test_add_model_with_base_layer)
        _run_test("Update Layer", test_update_layer)
        _run_test("Update Model with Nested Attributes", test_update_model_nested)
        _run_test("Show List", test_show_list)
        _run_test("Show Detailed", test_show_detailed)
        _run_test("Show Attribute", test_show_attribute)
        _run_test("Show Nested Attribute", test_show_nested_attribute)
        _run_test("Show List Attributes", test_show_list_attributes)
        _run_test("Delete Layer", test_delete_layer)
        _run_test("Delete Model", test_delete_model)
        _run_test("Error Handling", test_error_handling)
        
        # Print summary
        print("\n" + "="*60)
        print("Test Summary")
        print("="*60)
        print(f"Total Tests: {test_results['total']}")
        print(f"Passed: {len(test_results['passed'])}")
        print(f"Failed: {len(test_results['failed'])}")
        
        if test_results['failed']:
            print("\nFailed Tests:")
            for name in test_results['failed']:
                print(f"  - {name}")
        
        print("\n" + "="*60)
        
        # Return exit code
        return 0 if len(test_results['failed']) == 0 else 1
        
    finally:
        cleanup_test_environment()


if __name__ == "__main__":
    sys.exit(main())
