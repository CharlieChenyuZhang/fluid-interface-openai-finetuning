#!/usr/bin/env python3
"""
Test script to verify the fine-tuning installation

This script checks if all dependencies are installed correctly
and if the OpenAI API key is configured.
"""

import sys
import os
from pathlib import Path


def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import requests: {e}")
        return False
    
    try:
        import tiktoken
        print("✅ tiktoken imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import tiktoken: {e}")
        return False
    
    try:
        import numpy
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import numpy: {e}")
        return False
    
    try:
        import openai
        print("✅ openai imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import openai: {e}")
        return False
    
    return True


def test_api_key():
    """Test if OpenAI API key is configured"""
    print("\n🔑 Testing API key configuration...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    if len(api_key) < 20:
        print("❌ API key appears to be too short")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  API key doesn't start with 'sk-' (this might be normal for some key formats)")
    
    print("✅ API key is configured")
    return True


def test_files():
    """Test if required files exist"""
    print("\n📁 Testing required files...")
    
    required_files = [
        "openai_finetuning.py",
        "requirements.txt",
        "config.json",
        "data/toy_chat_fine_tuning.jsonl"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist


def test_script_execution():
    """Test if the main script can be executed"""
    print("\n🚀 Testing script execution...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "openai_finetuning.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Script executes successfully")
            return True
        else:
            print(f"❌ Script execution failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Script execution error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 OpenAI Fine-Tuning Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("API Key Configuration", test_api_key),
        ("Required Files", test_files),
        ("Script Execution", test_script_execution)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\n📋 Next steps:")
        print("1. Try: python openai_finetuning.py validate data/toy_chat_fine_tuning.jsonl")
        print("2. Try: python example_usage.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above before proceeding.")
        print("\n💡 Common fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Set API key: export OPENAI_API_KEY='your-api-key-here'")
        print("- Check file paths and permissions")


if __name__ == "__main__":
    main()
