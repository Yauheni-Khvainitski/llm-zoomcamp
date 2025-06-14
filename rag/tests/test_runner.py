"""
Comprehensive test runner for all RAG system components.

This module provides functionality to run all tests with detailed reporting.
"""

import sys
import time
import unittest


def run_all_tests(verbosity=2):
    """
    Run all tests in the RAG system.

    Args:
        verbosity: Test output verbosity level (0-2)

    Returns:
        True if all tests pass, False otherwise
    """
    print("🚀 Running Comprehensive RAG System Tests")
    print("=" * 60)

    # Discover and load all tests
    loader = unittest.TestLoader()
    start_dir = "rag/tests"
    suite = loader.discover(start_dir, pattern="test_*.py")

    # Count total tests
    total_tests = suite.countTestCases()
    print(f"📊 Found {total_tests} test cases")
    print("-" * 60)

    # Run tests with custom result class for better reporting
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout, resultclass=DetailedTestResult)

    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()

    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    duration = end_time - start_time
    print(f"⏱️  Total time: {duration:.2f} seconds")
    print(f"🧪 Tests run: {result.testsRun}")
    print(f"✅ Successes: {result.testsRun - len(result.failures) - len(result.errors)}")

    if result.failures:
        print(f"❌ Failures: {len(result.failures)}")

    if result.errors:
        print(f"💥 Errors: {len(result.errors)}")

    if result.skipped:
        print(f"⏭️  Skipped: {len(result.skipped)}")

    # Success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"📈 Success rate: {success_rate:.1f}%")

    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! The RAG system is working correctly.")
        return True
    else:
        print(f"\n❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")

        # Print detailed failure information
        if result.failures:
            print("\n📝 FAILURE DETAILS:")
            for i, (test, traceback) in enumerate(result.failures, 1):
                print(f"\n{i}. {test}")
                print("-" * 40)
                print(traceback)

        if result.errors:
            print("\n💥 ERROR DETAILS:")
            for i, (test, traceback) in enumerate(result.errors, 1):
                print(f"\n{i}. {test}")
                print("-" * 40)
                print(traceback)

        return False


def run_module_tests(module_name, verbosity=2):
    """
    Run tests for a specific module.

    Args:
        module_name: Name of the module to test (e.g., 'course', 'query_builder')
        verbosity: Test output verbosity level

    Returns:
        True if all tests pass, False otherwise
    """
    print(f"🧪 Running tests for {module_name} module")
    print("=" * 50)

    try:
        # Load specific test module
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(f"rag.tests.test_{module_name}")

        # Run tests
        runner = unittest.TextTestRunner(verbosity=verbosity)
        result = runner.run(suite)

        if result.wasSuccessful():
            print(f"\n✅ All {module_name} tests passed!")
            return True
        else:
            print(f"\n❌ {module_name} tests failed!")
            return False

    except Exception as e:
        print(f"💥 Error running {module_name} tests: {e}")
        return False


class DetailedTestResult(unittest.TextTestResult):
    """Custom test result class with enhanced reporting."""

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_start_time = None

    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()

    def addSuccess(self, test):
        super().addSuccess(test)
        if self.verbosity > 1:
            duration = time.time() - self.test_start_time
            self.stream.write(f" ({duration:.3f}s)")
            self.stream.flush()

    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 0:
            self.stream.write(" ERROR")
            self.stream.flush()

    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 0:
            self.stream.write(" FAIL")
            self.stream.flush()


def run_coverage_tests():
    """
    Run tests with coverage reporting (requires coverage.py).

    Returns:
        True if tests pass and coverage is acceptable
    """
    try:
        import coverage
    except ImportError:
        print("❌ Coverage.py not installed. Install with: pip install coverage")
        return False

    print("📊 Running tests with coverage analysis...")
    print("=" * 50)

    # Start coverage
    cov = coverage.Coverage(source=["rag"])
    cov.start()

    # Run tests
    success = run_all_tests(verbosity=1)

    # Stop coverage and generate report
    cov.stop()
    cov.save()

    print("\n📈 COVERAGE REPORT:")
    print("-" * 30)
    cov.report(show_missing=True)

    # Generate HTML report
    try:
        cov.html_report(directory="htmlcov")
        print("\n📄 HTML coverage report generated in 'htmlcov/' directory")
    except Exception as e:
        print(f"⚠️  Could not generate HTML report: {e}")

    return success


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument("--module", "-m", help="Run tests for specific module only")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage analysis")
    parser.add_argument("--verbose", "-v", action="count", default=2, help="Increase verbosity")

    args = parser.parse_args()

    if args.coverage:
        success = run_coverage_tests()
    elif args.module:
        success = run_module_tests(args.module, args.verbose)
    else:
        success = run_all_tests(args.verbose)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
