# tests/setup/simple_gemini_test.py
"""Simple direct test of Gemini API without any custom handlers"""

import os
import sys
import json
import requests
from pathlib import Path

def test_direct_gemini():
    """Test Gemini API directly using the format that worked in curl"""
    # Get API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return False
        
    # Use the exact working model and endpoint from your curl test
    model = "gemini-2.5-pro-preview-03-25"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    # Simple payload
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Explain the C4H project in 3 bullet points."
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1024
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"\n===============================================")
    print(f"Testing direct Gemini API access")
    print(f"Model: {model}")
    print(f"===============================================\n")
    
    try:
        # Make the request
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            content = result["candidates"][0]["content"]["parts"][0]["text"]
            
            print(f"\n===============================================")
            print(f"SUCCESS! Got response from Gemini 2.5:")
            print(f"===============================================\n")
            print(content)
            
            return True
        else:
            print(f"ERROR: API request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"ERROR: API request failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_gemini()
    sys.exit(0 if success else 1)