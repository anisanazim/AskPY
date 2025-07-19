#!/usr/bin/env python3
"""
Debug script to test .env file loading and Groq API key
Run this from your project root to diagnose the issue
"""

import os
import sys
from pathlib import Path

def test_env_loading():
    """Test different ways of loading environment variables"""
    
    print("🔍 Environment Variables Debug Test")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check if .env file exists
    env_path = os.path.join(current_dir, '.env')
    print(f"📄 .env file path: {env_path}")
    print(f"📄 .env file exists: {os.path.exists(env_path)}")
    
    if os.path.exists(env_path):
        print(f"📄 .env file size: {os.path.getsize(env_path)} bytes")
        
        # Read raw .env content
        print("\n📝 Raw .env file content:")
        print("-" * 30)
        try:
            with open(env_path, 'r') as f:
                content = f.read()
                print(repr(content))  # Shows hidden characters
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    
    print("\n🔑 Environment Variables (before loading .env):")
    print("-" * 50)
    groq_key_before = os.getenv("GROQ_API_KEY")
    print(f"GROQ_API_KEY: {groq_key_before}")
    
    # Try loading with python-dotenv
    print("\n📦 Testing python-dotenv loading...")
    try:
        from dotenv import load_dotenv
        result = load_dotenv(env_path)
        print(f"✅ load_dotenv result: {result}")
    except ImportError:
        print("❌ python-dotenv not installed")
        return
    except Exception as e:
        print(f"❌ Error loading .env: {e}")
        return
    
    print("\n🔑 Environment Variables (after loading .env):")
    print("-" * 50)
    groq_key_after = os.getenv("GROQ_API_KEY")
    print(f"GROQ_API_KEY: {groq_key_after}")
    print(f"GROQ_API_KEY length: {len(groq_key_after) if groq_key_after else 0}")
    print(f"GROQ_API_KEY starts with 'gsk_': {groq_key_after.startswith('gsk_') if groq_key_after else False}")
    
    # Test other variables
    other_vars = ['LLM_MODEL', 'EMBEDDING_MODEL', 'VECTOR_DB_PATH']
    for var in other_vars:
        value = os.getenv(var)
        print(f"{var}: {value}")
    
    # Test Groq API directly
    if groq_key_after:
        print(f"\n🚀 Testing Groq API connection...")
        test_groq_api(groq_key_after)
    else:
        print(f"\n❌ No GROQ_API_KEY found to test")

def test_groq_api(api_key):
    """Test if the Groq API key works"""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        # Try a simple API call
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Use a basic model
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✅ Groq API key is VALID!")
        print(f"✅ Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Groq API key is INVALID!")
        print(f"❌ Error: {e}")
        
        # Check specific error types
        if "401" in str(e) or "Invalid API Key" in str(e):
            print("🔍 This is definitely an API key issue")
        elif "quota" in str(e).lower():
            print("🔍 API key is valid but you've hit rate limits")
        else:
            print("🔍 Unexpected error - might be network or other issue")

def check_file_permissions():
    """Check file permissions on .env"""
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        print(f"\n🔒 File Permissions Check:")
        print("-" * 30)
        try:
            # Check if readable
            with open(env_path, 'r') as f:
                f.read(1)
            print("✅ File is readable")
        except Exception as e:
            print(f"❌ File read error: {e}")
        
        # Check file stats
        import stat
        file_stat = os.stat(env_path)
        print(f"📊 File mode: {stat.filemode(file_stat.st_mode)}")
        print(f"📊 File owner can read: {bool(file_stat.st_mode & stat.S_IRUSR)}")

if __name__ == "__main__":
    test_env_loading()
    check_file_permissions()
    
    print(f"\n💡 Diagnosis Tips:")
    print("-" * 20)
    print("1. If .env file doesn't exist → Create it")
    print("2. If GROQ_API_KEY is None → Check .env format")
    print("3. If GROQ_API_KEY exists but API fails → Get new key from Groq")
    print("4. If load_dotenv returns False → Check .env syntax")