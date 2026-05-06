#!/usr/bin/env python3
"""
Test script for crowd density forecasting API.
Run this after starting the web_app.py server.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:5000"

def test_forecast():
    """Test the forecasting endpoint."""
    print("=" * 60)
    print("CROWD DENSITY FORECASTING API TEST")
    print("=" * 60)
    
    endpoints = [
        {
            "name": "Hybrid (Fast + LSTM)",
            "params": {"method": "hybrid", "steps": 120}
        },
        {
            "name": "Fast Only (Linear + Exponential)",
            "params": {"method": "fast", "steps": 120}
        },
    ]
    
    for test in endpoints:
        print(f"\n▶ Testing: {test['name']}")
        print(f"  Params: {json.dumps(test['params'], indent=2)}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/forecast",
                json=test["params"],
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✓ Status: {response.status_code}")
                print(f"  Method Used: {data.get('method', 'unknown')}")
                print(f"  Current Density: {data.get('current_density', 0):.2f}%")
                print(f"  Current Count: {data.get('current_count', 0)}")
                print(f"  Forecast Steps: {data.get('steps', 0)}")
                
                forecast = data.get("forecast_density", [])
                if forecast:
                    print(f"  Forecast Range: {min(forecast):.2f}% - {max(forecast):.2f}%")
                    print(f"  First 5 predictions: {[f'{v:.1f}%' for v in forecast[:5]]}")
                    
                confidence_upper = data.get("confidence_upper", [])
                confidence_lower = data.get("confidence_lower", [])
                if confidence_upper and confidence_lower:
                    print(f"  Confidence Band (95%): [{confidence_lower[0]:.1f}%, {confidence_upper[0]:.1f}%]")
            else:
                print(f"  ✗ Status: {response.status_code}")
                print(f"  Error: {response.text}")
                
        except requests.ConnectionError:
            print(f"  ✗ Connection Error: Unable to connect to {BASE_URL}")
            print(f"    Make sure web_app.py is running on port 5000")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nAPI Endpoint: POST /api/forecast")
    print("\nRequest Parameters:")
    print("  - method: 'fast', 'lstm', or 'hybrid' (default: 'hybrid')")
    print("  - steps: forecast horizon in frames (10-600, default: 120)")
    print("\nResponse:")
    print("  - forecast_density: predicted density percentages")
    print("  - confidence_upper/lower: 95% confidence interval")
    print("  - forecast_count: predicted people count")
    print("  - method: forecasting method used")
    print("  - current_count/density: current statistics")

if __name__ == "__main__":
    # Wait a moment to ensure server is ready
    print("Waiting for server connection...")
    time.sleep(1)
    test_forecast()
