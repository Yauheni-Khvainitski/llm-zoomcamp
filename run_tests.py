#!/usr/bin/env python3
"""
Simple test runner script for the RAG system.

This script provides an easy way to run tests without requiring make or complex commands.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"🔄 {description}")
    print(f"   Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} - SUCCESS")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        if e.stdout:
            print(f"   Output: {e.stdout.strip()}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - COMMAND NOT FOUND")
        print(f"   Make sure the required tools are installed")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--format", action="store_true", help="Format code")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--module", help="Run tests for specific module")
    
    args = parser.parse_args()
    
    # Change to script directory
    script_dir = Path(__file__).parent
    print(f"📁 Working directory: {script_dir.absolute()}")
    
    success_count = 0
    total_count = 0
    
    if args.format or args.all:
        total_count += 2
        if run_command(["python", "-m", "black", "rag/"], "Format code with Black"):
            success_count += 1
        if run_command(["python", "-m", "isort", "rag/"], "Sort imports with isort"):
            success_count += 1
    
    if args.lint or args.all:
        total_count += 1
        if run_command(["python", "-m", "flake8", "rag/", "--max-line-length=127", "--extend-ignore=E203,W503"], "Lint with flake8"):
            success_count += 1
    
    if args.module:
        total_count += 1
        if run_command(["python", "-m", "pytest", f"rag/tests/test_{args.module}.py", "-v"], f"Test {args.module} module"):
            success_count += 1
    elif args.unit or args.all:
        total_count += 1
        if run_command(["python", "-m", "pytest", "rag/tests/", "-v"], "Run unit tests"):
            success_count += 1
    
    if args.coverage or args.all:
        total_count += 1
        if run_command(["python", "-m", "pytest", "rag/tests/", "--cov=rag", "--cov-report=term"], "Run tests with coverage"):
            success_count += 1
    
    # If no specific options, run basic tests
    if not any([args.unit, args.coverage, args.lint, args.format, args.all, args.module]):
        total_count += 1
        if run_command(["python", "-m", "pytest", "rag/tests/", "-v"], "Run unit tests"):
            success_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {total_count - success_count}")
    print(f"📈 Success rate: {(success_count/total_count)*100:.1f}%" if total_count > 0 else "No tests run")
    
    if success_count == total_count:
        print("\n🎉 ALL CHECKS PASSED!")
        return 0
    else:
        print(f"\n💥 {total_count - success_count} CHECKS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 