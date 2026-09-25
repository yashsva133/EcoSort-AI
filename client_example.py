"""
EcoSort AI - Smart Bin & IoT Client Example
Demonstrates how an edge device (e.g. Raspberry Pi, Jetson Nano, ESP32 gateway)
interacts with the EcoSort AI REST API to actuate physical sorting bins.
"""

import requests
import json
import base64
import sys

SERVER_URL = "http://localhost:5000"

def classify_waste_image(image_path: str):
    """Send an image to the EcoSort AI inference endpoint."""
    print(f"[*] Sending '{image_path}' to EcoSort AI Engine at {SERVER_URL}...")
    
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    b64_str = base64.b64encode(image_bytes).decode('utf-8')
    payload = {"image_base64": f"data:image/jpeg;base64,{b64_str}"}

    try:
        response = requests.post(f"{SERVER_URL}/api/detect", json=payload, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        
        print("\n=== ECOSORT AI INFERENCE RESULT ===")
        print(f"Stream:          {data.get('stream')} ({data.get('stream_display', '')})")
        print(f"Item:            {data.get('primary_item')}")
        print(f"Confidence:      {data.get('confidence')}%")
        print(f"Estimated Mass:  {data.get('grams')} grams")
        print(f"Directive:       {data.get('bin_directive')}")
        print(f"Action Detail:   {data.get('bin_sub')}")
        print(f"Inference Time:  {data.get('latency_ms')} ms")
        
        # Example Robotic Flap / Servo Actuation Logic:
        stream = data.get('stream')
        if stream == 'ORGANIC':
            print("\n[ROBOT ACTION] Actuate SERVO_1 (0 deg -> 90 deg) -> Drop into GREEN COMPOST BIN")
        elif stream == 'INORGANIC':
            print("\n[ROBOT ACTION] Actuate SERVO_2 (0 deg -> -90 deg) -> Drop into BLUE RECYCLABLE BIN")
        elif stream == 'DUAL':
            print("\n[ROBOT ACTION] DUAL-STREAM CAFETERIA DETECTED:")
            print("               1. Prompt user: Scrape food scraps into GREEN BIN")
            print("               2. Route empty plate/carton into BLUE BIN")
        else:
            print("\n[ROBOT ACTION] No waste detected or uncertain. Keep bin closed.")

        return data
    except requests.exceptions.RequestException as e:
        print(f"[!] Connection failed: {e}")
        return None

if __name__ == "__main__":
    test_img = "demo_samples/sample_canteen_plate_leftovers.jpg"
    if len(sys.argv) > 1:
        test_img = sys.argv[1]
    classify_waste_image(test_img)
