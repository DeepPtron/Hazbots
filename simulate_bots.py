"""
Multi-Bot Simulator for LoRa Bot Monitor
Simulates bots 2-20 sending random sensor data to the dashboard.
Each bot has a FIXED GPS position once deployed.
Random human detection alerts every ~45 seconds.
"""

import requests
import time
import random

# Server URL
SERVER_URL = "http://localhost:5000/api/data"

# Base coordinates (Chennai area)
BASE_LAT = 13.008267
BASE_LNG = 80.005501

# Store fixed positions for each bot (assigned once on first run)
BOT_POSITIONS = {}

# Track last human detection time
last_human_detection = 0

def get_fixed_position(bot_id):
    """Get or create a fixed GPS position for a bot."""
    if bot_id not in BOT_POSITIONS:
        # Assign random position within ~500m radius on first call
        BOT_POSITIONS[bot_id] = {
            "lat": BASE_LAT + random.uniform(-0.005, 0.005),
            "lng": BASE_LNG + random.uniform(-0.005, 0.005)
        }
        print(f"   📍 Bot {bot_id:02d} deployed at: {BOT_POSITIONS[bot_id]['lat']:.6f}, {BOT_POSITIONS[bot_id]['lng']:.6f}")
    return BOT_POSITIONS[bot_id]

def generate_bot_data(bot_id, human_detected=False):
    """Generate random sensor data for a bot with fixed position."""
    pos = get_fixed_position(bot_id)
    
    return {
        "device_id": f"ESP32_BOT_{bot_id:02d}",
        "temperature": round(random.uniform(20, 40), 1),
        "pressure": round(random.uniform(1000, 1020), 1),
        "altitude": round(random.uniform(10, 100), 1),
        "gas_co_ppm": round(random.uniform(5, 50), 0),
        "accel_z": round(random.uniform(9.5, 10.0), 2),
        "audio_rms": round(random.uniform(50, 500), 0),
        "gps_valid": "true",
        "lat": str(round(pos["lat"], 6)),
        "lng": str(round(pos["lng"], 6)),
        "human_detected": human_detected  # New field for human detection
    }

def send_data(data):
    """Send data to the server."""
    try:
        response = requests.post(SERVER_URL, json=data, timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def main():
    global last_human_detection
    
    print("🤖 Multi-Bot Simulator (with Human Detection)")
    print("=" * 50)
    print(f"Simulating bots 2-20 sending data to {SERVER_URL}")
    print("• GPS positions are FIXED once deployed")
    print("• Human detection alerts every ~45 seconds")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    bot_ids = list(range(2, 21))  # Bots 2 to 20
    
    try:
        round_num = 1
        while True:
            print(f"\n📡 Round {round_num}")
            
            current_time = time.time()
            
            # Check if it's time for a human detection alert (~45 seconds)
            human_detect_bot = None
            if current_time - last_human_detection >= 45:
                human_detect_bot = random.choice(bot_ids)
                last_human_detection = current_time
            
            # Send data from random subset of bots each round
            active_bots = random.sample(bot_ids, k=random.randint(3, len(bot_ids)))
            
            for bot_id in sorted(active_bots):
                is_human_detected = (bot_id == human_detect_bot)
                data = generate_bot_data(bot_id, human_detected=is_human_detected)
                result = send_data(data)
                
                if "error" in result:
                    print(f"   ❌ Bot {bot_id:02d}: {result['error']}")
                else:
                    tag = result.get('bot_tag', '??')
                    if is_human_detected:
                        pos = BOT_POSITIONS[bot_id]
                        print(f"   🚨 Bot {bot_id:02d} → Tag #{tag} | 🧑 HUMAN DETECTED at ({pos['lat']:.6f}, {pos['lng']:.6f})")
                    else:
                        print(f"   ✅ Bot {bot_id:02d} → Tag #{tag}")
            
            round_num += 1
            
            # Wait between rounds (2-5 seconds)
            wait_time = random.uniform(2, 5)
            print(f"\n⏳ Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Simulator stopped")

if __name__ == "__main__":
    main()
