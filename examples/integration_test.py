"""Integration test demonstrating the complete QKD system components."""
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import logging
import time
import threading
from queue import Queue
from typing import Dict, Any

# Import all components
from src.alice.laser.laser_controller import LaserController
from src.alice.laser.laser_simulator import SimulatedLaserDriver
from src.utils.data_structures import LaserInfo, Pulse
from src.alice.polarization.polarization_controller import PolarizationController
from src.alice.polarization.polarization_simulator import PolarizationSimulator
from src.alice.qrng.qrng_simulator import OperationMode, QRNGSimulator
from src.quantum_channel.free_space_channel import FreeSpaceChannel
from src.utils.data_structures import Pulse
from configs.hardware_config import SYSTEM_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AliceSimulator:
    """Simulated Alice station for QKD."""
    
    def __init__(self):
        """Initialize Alice's components."""
        self.laser = None
        self.polarization = None
        self.qrng = None
        self.is_running = False
        
    def initialize(self):
        """Initialize all Alice components."""
        try:
            logger.info("Initializing Alice station...")
            
            # Initialize laser
            from queue import Queue
            pulses_queue = Queue()
            laser_info = LaserInfo(
                max_power_mW=1000.0,
                pulse_width_fwhm_ns=1000.0,
                central_wavelength_nm=1550.0
            )
            laser_driver = SimulatedLaserDriver(pulses_queue, laser_info)
            self.laser = LaserController(laser_driver)
            self.laser.initialize()
            
            # Initialize polarization controller with QRNG
            polarization_queue = Queue()
            pol_driver = PolarizationSimulator(pulses_queue, polarization_queue, laser_info)
            qrng_driver = QRNGSimulator()
            self.polarization = PolarizationController(pol_driver, qrng_driver)
            self.polarization.initialize()
            
            logger.info("Alice station initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Alice station: {e}")
            return False
    
    def send_qkd_sequence(self, n_pulses: int = 100) -> Dict[str, Any]:
        """Send a sequence of QKD pulses."""
        logger.info(f"Alice sending {n_pulses} QKD pulses...")
        
        results = {
            "pulses_sent": 0,
            "basis_choices": [],
            "bit_values": [],
            "intensities": [],
            "timing": []
        }
        
        try:
            # Turn on laser
            self.laser.initialize()
            
            # Start pulse sequence
            start_time = time.time()
            
            for i in range(n_pulses):
                # Generate random basis and bit
                basis = self.polarization.qrng.get_random_bit(mode=OperationMode.DETERMINISTIC)
                bit_value = self.polarization.qrng.get_random_bit(mode=OperationMode.DETERMINISTIC)

                pol_angle = self.polarization.calculate_polarization_angle(basis, bit_value)
                # Set polarization
                pulse = Pulse(
                    photons=1,
                    polarization=pol_angle
                )
                
                # Send pulse to polarization controller
                self.polarization.input_queue.put(pulse)
                
                # Trigger laser
                self.laser.trigger_once()
                
                # Record this pulse
                results["pulses_sent"] += 1
                results["basis_choices"].append(basis[0])  # H/V or D/A
                results["bit_values"].append(bit_value)
                results["intensities"].append(pulse.intensity)
                results["timing"].append(pulse.timestamp - start_time)
                
                # Small delay between pulses
                time.sleep(0.001)  # 1kHz rate
            
            # Turn off laser
            self.laser.turn_off()
            
            logger.info(f"Alice completed {results['pulses_sent']} pulses")
            return results
            
        except Exception as e:
            logger.error(f"Error in Alice pulse sequence: {e}")
            return results
    
    def shutdown(self):
        """Shutdown Alice components."""
        logger.info("Shutting down Alice station...")
        
        if self.laser:
            self.laser.shutdown()
        
        if self.polarization:
            self.polarization.shutdown()
        
        logger.info("Alice station shutdown complete")


class BobSimulator:
    """Simulated Bob station for QKD."""
    
    def __init__(self):
        """Initialize Bob's components."""
        self.detection_results = []
        self.is_measuring = False
        
    def initialize(self):
        """Initialize Bob's detection system."""
        logger.info("Initializing Bob station...")
        self.detection_results = []
        logger.info("Bob station initialized successfully")
        return True
    
    def start_detection(self, measurement_time: float = 10.0):
        """Start detection sequence."""
        logger.info(f"Bob starting detection for {measurement_time} seconds...")
        
        self.is_measuring = True
        self.detection_results = []
        
        # Simulate detection events
        start_time = time.time()
        detection_count = 0
        
        while (time.time() - start_time) < measurement_time and self.is_measuring:
            # Simulate random basis choice
            basis = ['H', 'D'][detection_count % 2]  # Alternate bases
            
            # Simulate detection (simplified)
            detected = True if (detection_count % 3) != 0 else False  # 2/3 detection rate
            
            if detected:
                result = {
                    "timestamp": time.time(),
                    "detection_id": detection_count,
                    "basis": basis,
                    "detector": detection_count % 4,  # 4 detectors
                    "measurement": detection_count % 2  # 0 or 1
                }
                self.detection_results.append(result)
            
            detection_count += 1
            time.sleep(0.001)  # Match Alice's rate
        
        self.is_measuring = False
        logger.info(f"Bob detected {len(self.detection_results)} events")
        return self.detection_results
    
    def stop_detection(self):
        """Stop detection."""
        self.is_measuring = False
    
    def shutdown(self):
        """Shutdown Bob components."""
        logger.info("Shutting down Bob station...")
        self.stop_detection()
        logger.info("Bob station shutdown complete")


class QuantumChannelSimulator:
    """Simulated quantum channel between Alice and Bob."""
    
    def __init__(self, loss_db: float = 3.0):
        """Initialize channel with specified loss."""
        self.loss_db = loss_db
        self.transmission = 10 ** (-loss_db / 10)
        logger.info(f"Quantum channel: {loss_db} dB loss, {self.transmission:.3f} transmission")
    
    def transmit_pulse(self, pulse: Pulse) -> bool:
        """Simulate pulse transmission through channel."""
        # Simple loss model
        import random
        return random.random() < self.transmission


class QKDSystemIntegrationTest:
    """Integration test for the complete QKD system."""
    
    def __init__(self):
        """Initialize the test system."""
        self.alice = AliceSimulator()
        self.bob = BobSimulator()
        self.channel = QuantumChannelSimulator(loss_db=3.0)
        
    def run_test(self, n_pulses: int = 100, measurement_time: float = 10.0):
        """Run the complete integration test."""
        logger.info("=== QKD System Integration Test ===")
        
        try:
            # Initialize both stations
            if not self.alice.initialize():
                logger.error("Failed to initialize Alice")
                return False
            
            if not self.bob.initialize():
                logger.error("Failed to initialize Bob")
                return False
            
            # Start Bob's detection in a separate thread
            bob_thread = threading.Thread(
                target=self.bob.start_detection,
                args=(measurement_time,)
            )
            bob_thread.start()
            
            # Small delay to ensure Bob is ready
            time.sleep(0.1)
            
            # Alice sends pulse sequence
            alice_results = self.alice.send_qkd_sequence(n_pulses)
            
            # Wait for Bob to finish
            bob_thread.join()
            
            # Analyze results
            self.analyze_results(alice_results, self.bob.detection_results)
            
            return True
            
        except Exception as e:
            logger.error(f"Integration test failed: {e}")
            return False
        
        finally:
            # Cleanup
            self.alice.shutdown()
            self.bob.shutdown()
    
    def analyze_results(self, alice_results: Dict, bob_results: list):
        """Analyze the QKD test results."""
        logger.info("=== Test Results Analysis ===")
        
        # Alice statistics
        logger.info(f"Alice sent: {alice_results['pulses_sent']} pulses")
        
        if alice_results['basis_choices']:
            h_count = alice_results['basis_choices'].count('H')
            d_count = alice_results['basis_choices'].count('D')
            logger.info(f"Alice basis distribution: H/V={h_count}, D/A={d_count}")
        
        # Bob statistics
        logger.info(f"Bob detected: {len(bob_results)} events")
        
        if bob_results:
            h_detections = sum(1 for r in bob_results if r['basis'] == 'H')
            d_detections = sum(1 for r in bob_results if r['basis'] == 'D')
            logger.info(f"Bob basis distribution: H/V={h_detections}, D/A={d_detections}")
        
        # System performance
        if alice_results['pulses_sent'] > 0:
            detection_rate = len(bob_results) / alice_results['pulses_sent']
            logger.info(f"Overall detection rate: {detection_rate:.3f}")
        
        # Timing analysis
        if alice_results['timing'] and len(alice_results['timing']) > 1:
            avg_interval = sum(alice_results['timing'][i+1] - alice_results['timing'][i] 
                             for i in range(len(alice_results['timing'])-1)) / (len(alice_results['timing'])-1)
            logger.info(f"Average pulse interval: {avg_interval*1000:.1f} ms")
        
        logger.info("Analysis complete")


def main():
    """Main test function."""
    logger.info("Starting QKD System Integration Test...")
    
    # Create and run test
    test_system = QKDSystemIntegrationTest()
    success = test_system.run_test(n_pulses=50, measurement_time=5.0)
    
    if success:
        logger.info("🎉 Integration test completed successfully!")
    else:
        logger.error("💥 Integration test failed!")
    
    return success


if __name__ == "__main__":
    main()
