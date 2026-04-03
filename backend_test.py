#!/usr/bin/env python3
"""
Comprehensive backend testing for Spacecraft Debris Simulation API
Tests all endpoints, WebSocket functionality, and simulation logic
"""

import requests
import json
import asyncio
import websockets
import sys
import time
from datetime import datetime

class SpacecraftSimulationTester:
    def __init__(self, base_url="https://debris-cascade-view.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.websocket_tests_passed = 0
        self.websocket_tests_run = 0

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED {details}")
        else:
            print(f"❌ {name} - FAILED {details}")
        return success

    def test_api_root(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/", timeout=10)
            success = response.status_code == 200
            data = response.json() if success else {}
            details = f"Status: {response.status_code}, Response: {data}"
            return self.log_test("API Root", success, details)
        except Exception as e:
            return self.log_test("API Root", False, f"Error: {str(e)}")

    def test_simulation_state(self):
        """Test getting simulation state"""
        try:
            response = requests.get(f"{self.base_url}/api/simulation/state", timeout=10)
            success = response.status_code == 200
            if success:
                data = response.json()
                required_keys = ['satellites', 'debris', 'stats', 'predictor_mode']
                has_required = all(key in data for key in required_keys)
                success = has_required
                details = f"Status: {response.status_code}, Keys: {list(data.keys())}"
            else:
                details = f"Status: {response.status_code}"
            return self.log_test("Get Simulation State", success, details)
        except Exception as e:
            return self.log_test("Get Simulation State", False, f"Error: {str(e)}")

    def test_reset_simulation_default(self):
        """Test resetting simulation with default parameters"""
        try:
            response = requests.post(f"{self.base_url}/api/simulation/reset", timeout=10)
            success = response.status_code == 200
            if success:
                data = response.json()
                success = data.get('status') == 'reset'
                details = f"Status: {response.status_code}, Response: {data}"
            else:
                details = f"Status: {response.status_code}"
            return self.log_test("Reset Simulation (Default)", success, details)
        except Exception as e:
            return self.log_test("Reset Simulation (Default)", False, f"Error: {str(e)}")

    def test_reset_simulation_custom(self):
        """Test resetting simulation with custom parameters"""
        try:
            custom_params = {
                "satellite_count": 8,
                "max_debris": 300,
                "kessler_threshold": 80,
                "collision_distance": 20.0,
                "debris_per_collision": 6,
                "simulation_speed": 1.5
            }
            response = requests.post(
                f"{self.base_url}/api/simulation/reset",
                json=custom_params,
                timeout=10
            )
            success = response.status_code == 200
            if success:
                data = response.json()
                success = data.get('status') == 'reset' and data.get('params') is not None
                details = f"Status: {response.status_code}, Params: {data.get('params')}"
            else:
                details = f"Status: {response.status_code}"
            return self.log_test("Reset Simulation (Custom)", success, details)
        except Exception as e:
            return self.log_test("Reset Simulation (Custom)", False, f"Error: {str(e)}")

    def test_toggle_pause(self):
        """Test pause/resume functionality"""
        try:
            response = requests.post(f"{self.base_url}/api/simulation/toggle-pause", timeout=10)
            success = response.status_code == 200
            if success:
                data = response.json()
                success = 'is_paused' in data
                details = f"Status: {response.status_code}, Paused: {data.get('is_paused')}"
            else:
                details = f"Status: {response.status_code}"
            return self.log_test("Toggle Pause", success, details)
        except Exception as e:
            return self.log_test("Toggle Pause", False, f"Error: {str(e)}")

    def test_speed_adjustment(self):
        """Test simulation speed adjustment"""
        try:
            # Test speed increase
            response = requests.post(
                f"{self.base_url}/api/simulation/speed",
                json={"delta": 0.5},
                timeout=10
            )
            success = response.status_code == 200
            if success:
                data = response.json()
                success = 'simulation_speed' in data
                details = f"Status: {response.status_code}, Speed: {data.get('simulation_speed')}"
            else:
                details = f"Status: {response.status_code}"
            return self.log_test("Speed Adjustment", success, details)
        except Exception as e:
            return self.log_test("Speed Adjustment", False, f"Error: {str(e)}")

    def test_add_satellite(self):
        """Test adding a new satellite"""
        try:
            response = requests.post(
                f"{self.base_url}/api/simulation/add-satellite",
                json={"x": 150.0, "y": 100.0},
                timeout=10
            )
            success = response.status_code == 200
            if success:
                data = response.json()
                success = data.get('status') == 'added' and 'satellite' in data
                details = f"Status: {response.status_code}, Satellite ID: {data.get('satellite', {}).get('id', 'N/A')}"
            else:
                details = f"Status: {response.status_code}"
            return self.log_test("Add Satellite", success, details)
        except Exception as e:
            return self.log_test("Add Satellite", False, f"Error: {str(e)}")

    def test_predictor_mode_change(self):
        """Test changing predictor mode"""
        try:
            # Test rule-based mode
            response = requests.post(
                f"{self.base_url}/api/simulation/predictor-mode",
                json={"mode": "rule-based"},
                timeout=10
            )
            success = response.status_code == 200
            if success:
                data = response.json()
                success = data.get('status') == 'success'
                details = f"Status: {response.status_code}, Mode: {data.get('mode')}"
            else:
                details = f"Status: {response.status_code}"
            return self.log_test("Predictor Mode Change", success, details)
        except Exception as e:
            return self.log_test("Predictor Mode Change", False, f"Error: {str(e)}")

    async def test_websocket_connection(self):
        """Test WebSocket connection and real-time updates"""
        try:
            ws_url = self.base_url.replace('https://', 'wss://').replace('http://', 'ws://')
            uri = f"{ws_url}/api/ws/simulation"
            
            async with websockets.connect(uri) as websocket:
                self.websocket_tests_run += 1
                
                # Test initial state reception
                try:
                    initial_data = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(initial_data)
                    
                    required_keys = ['satellites', 'debris', 'stats', 'predictor_mode']
                    has_required = all(key in data for key in required_keys)
                    
                    if has_required:
                        self.websocket_tests_passed += 1
                        print(f"✅ WebSocket Initial State - PASSED (Keys: {list(data.keys())})")
                    else:
                        print(f"❌ WebSocket Initial State - FAILED (Missing keys)")
                        
                except asyncio.TimeoutError:
                    print("❌ WebSocket Initial State - FAILED (Timeout)")
                
                # Test sending commands
                self.websocket_tests_run += 1
                try:
                    # Send pause command
                    await websocket.send(json.dumps({"type": "toggle_pause"}))
                    
                    # Wait for response
                    response_data = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response = json.loads(response_data)
                    
                    if 'stats' in response and 'is_paused' in response['stats']:
                        self.websocket_tests_passed += 1
                        print(f"✅ WebSocket Commands - PASSED (Pause state: {response['stats']['is_paused']})")
                    else:
                        print("❌ WebSocket Commands - FAILED (No pause state in response)")
                        
                except asyncio.TimeoutError:
                    print("❌ WebSocket Commands - FAILED (Timeout)")
                
                # Test real-time updates
                self.websocket_tests_run += 1
                try:
                    # Receive multiple updates to verify real-time streaming
                    updates_received = 0
                    for _ in range(3):
                        update_data = await asyncio.wait_for(websocket.recv(), timeout=2)
                        update = json.loads(update_data)
                        if 'stats' in update and 'frame_count' in update['stats']:
                            updates_received += 1
                    
                    if updates_received >= 2:
                        self.websocket_tests_passed += 1
                        print(f"✅ WebSocket Real-time Updates - PASSED ({updates_received} updates)")
                    else:
                        print(f"❌ WebSocket Real-time Updates - FAILED (Only {updates_received} updates)")
                        
                except asyncio.TimeoutError:
                    print("❌ WebSocket Real-time Updates - FAILED (Timeout)")
                    
        except Exception as e:
            print(f"❌ WebSocket Connection - FAILED (Error: {str(e)})")

    def run_all_tests(self):
        """Run all backend tests"""
        print("🚀 Starting Spacecraft Debris Simulation Backend Tests")
        print("=" * 60)
        
        # API Tests
        print("\n📡 Testing REST API Endpoints:")
        self.test_api_root()
        self.test_simulation_state()
        self.test_reset_simulation_default()
        self.test_reset_simulation_custom()
        self.test_toggle_pause()
        self.test_speed_adjustment()
        self.test_add_satellite()
        self.test_predictor_mode_change()
        
        # WebSocket Tests
        print("\n🔌 Testing WebSocket Functionality:")
        try:
            asyncio.run(self.test_websocket_connection())
        except Exception as e:
            print(f"❌ WebSocket Tests Failed: {str(e)}")
        
        # Results Summary
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"REST API Tests: {self.tests_passed}/{self.tests_run} passed")
        print(f"WebSocket Tests: {self.websocket_tests_passed}/{self.websocket_tests_run} passed")
        
        total_passed = self.tests_passed + self.websocket_tests_passed
        total_run = self.tests_run + self.websocket_tests_run
        success_rate = (total_passed / total_run * 100) if total_run > 0 else 0
        
        print(f"Overall Success Rate: {success_rate:.1f}% ({total_passed}/{total_run})")
        
        if success_rate >= 80:
            print("🎉 Backend tests mostly successful!")
            return 0
        else:
            print("⚠️  Backend has significant issues that need attention")
            return 1

def main():
    """Main test execution"""
    tester = SpacecraftSimulationTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    sys.exit(main())