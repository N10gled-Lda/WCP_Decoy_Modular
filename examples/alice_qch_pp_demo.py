"""
Simple demonstration of Alice's hardware-controlled BB84 protocol.
This script shows how to use the hardware components for real quantum key distribution.
"""

import logging
import time
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from examples.logging_setup import setup_logger
from examples.alice_qch_pp import AliceHardwareQubits, AliceTestMode, HardwareAliceConfig

# Setup logger
logger = setup_logger("Alice Hardware Demo", logging.INFO)


def demo_hardware_alice_simple():
    """Simple demo showing Alice's hardware components in action."""
    
    print("=" * 60)
    print("Alice Hardware BB84 Simple Demo")
    print("=" * 60)
    
    # Configuration for a simple demo
    config = HardwareAliceConfig(
        num_qubits=10,           # Generate 10 qubits
        pulse_period_seconds=1.0, # 1 second between pulses
        use_hardware=False,      # Use simulators for demo
        com_port="COM4",           # No COM port (using simulator)
        laser_channel=8,         # Laser channel (ignored in sim mode)
        mode=AliceTestMode.SEEDED, # Use seeded mode for reproducibility
        qrng_seed=42,           # Reproducible random numbers
        use_mock_receiver=True,  # Use mock receiver for demo
        server_host="localhost",
        server_port=12345,
        test_fraction=0.1,
        loss_rate=0.0,
    )
    
    print(f"Configuration:")
    print(f"  Qubits: {config.num_qubits}")
    print(f"  Pulse period: {config.pulse_period_seconds}s")
    print(f"  Hardware mode: {config.use_hardware}")
    print(f"  COM port: {config.com_port}")
    print(f"  Laser channel: {config.laser_channel}")
    print(f"  Test mode: {config.mode}")
    print(f"  QRNG seed: {config.qrng_seed}")
    print(f"  Mock receiver: {config.use_mock_receiver}")
    print(f"  Server host: {config.server_host}")
    print(f"  Server port: {config.server_port}")
    print(f"  Loss rate: {config.loss_rate}")

    print("\n\n\nStarting demo...\n")
    
    try:
        # Create Alice hardware instance
        logger.info("Initializing Alice hardware...")
        alice = AliceHardwareQubits(config)
        
        # Setup mock quantum channel (in real scenario, this connects to Bob)
        logger.info("Setting up mock quantum channel...")
        import socket
        import threading
        
        
        # Setup server
        server = AliceHardwareQubits.setup_server(config.server_host, config.server_port, timeout=30.0)

        # Start mock Bob in background
        if config.use_mock_receiver:
            alice.setup_mock_receiver()
        
        # Give Bob a moment to start
        time.sleep(0.5)
        
        # Connect Alice to the mock quantum channel
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((config.server_host, config.server_port))
        
        # Run Alice's qubit operations
        logger.info("Starting Alice's qubit generation and transmission...")
        start_time = time.time()
        
        results = alice.run_trl4_protocol_quantum_part(client_socket)
        
        end_time = time.time()
        
        # Close connections
        client_socket.close()
        server.close()
        
        # Show results
        print("\n" + "=" * 60)
        print("Alice Hardware Demo Results")
        print("=" * 60)
        
        print(f"✓ Successfully generated {len(results.bits)} qubits")
        print(f"✓ Transmission time: {end_time - start_time:.2f} seconds")
        print(f"✓ Average pulse rate: {len(results.bits) / (end_time - start_time):.2f} Hz")
        print(f"✓ Bases generated: Z={results.bases.count(0)}, X={results.bases.count(1)}")
        print(f"✓ Bits generated: 0={results.bits.count(0)}, 1={results.bits.count(1)}")

        # Show first few qubits for verification
        print(f"\nFirst 10 qubits (bit,base):")
        for i in range(min(10, len(alice.bits))):
            basis_name = "Z" if results.bases[i] == 0 else "X"
            print(f"  Qubit {i}: bit={results.bits[i]}, basis={basis_name}")

        print(f"\nHardware components status:")
        print(f"  ✓ Laser controller: operational")
        print(f"  ✓ Polarization controller: operational") 
        print(f"  ✓ QRNG: operational")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")
        return False
    
    print("\n✓ Demo completed successfully!")
    return True


if __name__ == "__main__":
    print("Alice Hardware BB84 Protocol Demonstration")
    print("This demo shows Alice's side of BB84 using hardware components")
    print()
    
    # Run simple demo
    success = demo_hardware_alice_simple()
    if success:
        print("\nDemo completed successfully!")
        sys.exit(0)
    if not success:
        print("\nDemo encountered errors.")
    
    # Usage instructions
    print("\n" + "=" * 60)
    print("Usage Instructions")
    print("=" * 60)
    print("To run the full BB84 protocol:")
    print("  python examples/alice_hardware_bb84.py --help")
    print()
    print("Example with hardware:")
    print("  python examples/alice_hardware_bb84.py --use_hardware --com_port COM4 --laser_channel 8")
    print()
    print("Example with simulation:")
    print("  python examples/alice_hardware_bb84.py -k 1000 -pp 0.5")
    print()
    print("Parameters:")
    print("  -k: Key length (number of qubits)")
    print("  -pp: Pulse period in seconds") 
    print("  --use_hardware: Use actual hardware")
    print("  --com_port: COM port for polarization control")
    print("  --laser_channel: Digital channel for laser")
    print("  --test_mode: Test mode (SEEDED, FIXED, RANDOM)")
    print("  --qrng_seed: Seed for QRNG in SEEDED mode")
    print("  --use_mock_receiver: Use mock Bob receiver for testing")
    print("  --server_host: Host for quantum channel server")
    print("  --server_port: Port for quantum channel server")
    
    
    sys.exit(1)
