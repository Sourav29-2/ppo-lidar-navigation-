# tests/run_smoke_tests.py
import sys
from pathlib import Path

# Add project paths to import modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "src" / "urdf_test" / "src" / "scripts"))

from tests.test_preprocessing import (
    test_lidar_preprocessing_sanitization,
    test_lidar_preprocessing_clipping_and_normalization,
    test_lidar_preprocessing_scenarios,
    test_neural_network_compatibility,
    test_rollout_buffer_and_ppo_update
)

if __name__ == "__main__":
    print("Running LiDAR preprocessing pipeline smoke tests...")
    
    try:
        print("1/5 Running test_lidar_preprocessing_sanitization...")
        test_lidar_preprocessing_sanitization()
        
        print("2/5 Running test_lidar_preprocessing_clipping_and_normalization...")
        test_lidar_preprocessing_clipping_and_normalization()
        
        print("3/5 Running test_lidar_preprocessing_scenarios...")
        test_lidar_preprocessing_scenarios()
        
        print("4/5 Running test_neural_network_compatibility...")
        test_neural_network_compatibility()
        
        print("5/5 Running test_rollout_buffer_and_ppo_update...")
        test_rollout_buffer_and_ppo_update()
        
        print("\nSUCCESS: All smoke tests passed successfully!")
        sys.exit(0)
    except AssertionError as e:
        print(f"\nFAILURE: Smoke test failed due to AssertionError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: Smoke test failed due to unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
