# WCP Decoy-State QKD Modular Simulator Documentation

## Overview

This project is a modular simulator for a Weak-Coherent-Pulse (WCP) Decoy-State BB84 Quantum Key Distribution (QKD) system. It is designed to be flexible, allowing for both simulation and integration with physical hardware.

## Architecture

The simulator is divided into several key modules:

- **`src/alice`**: Contains all components related to Alice, the sender. This includes the laser, variable optical attenuator (VOA), polarization modulator, and the quantum random number generator (QRNG).
- **`src/bob`**: Contains all components related to Bob, the receiver. This includes the detectors, optical table, and time tagger.
- **`src/quantum_channel`**: Simulates the quantum channel, including effects like attenuation and eavesdropping.
- **`src/protocol`**: Contains the implementation of the classical post-processing protocols, including BB84, error reconciliation, and privacy amplification.
- **`src/utils`**: Provides utility functions and data structures used throughout the simulation.
- **`configs`**: Contains configuration files for the simulation.
- **`tests`**: Contains pytest tests for all modules.
- **`examples`**: Contains example scripts demonstrating how to use the simulator.

## Getting Started

### Installation

1. Clone the repository.
2. Install the required dependencies using Poetry:

    ```bash
    pip install poetry
    poetry install
    ```

### Running the Simulation

The simulation can be run from the command line using the `__main__.py` entrypoint.

```bash
poetry run python wcp_decoy_qkd_modular --config configs/config.yaml
```

You can also specify options to run with physical hardware, set a random seed, and specify an output directory.

```bash
poetry run python wcp_decoy_qkd_modular --physical --seed 1234 --out-dir /path/to/results
```

## Physics Models

The simulator includes several physics models that can be configured and extended.

### Laser Source

The laser is modeled as a coherent state source, where the number of photons in each pulse follows a Poisson distribution. The mean photon number (μ) can be set in the configuration file.

**WARNING: CONFIRM MODEL** - Please confirm that this model is appropriate for your use case.

### Quantum Channel

The quantum channel is modeled with a constant attenuation coefficient (dB/km), which is used to calculate the transmission probability.

**WARNING: CONFIRM MODEL** - This model does not account for atmospheric turbulence or other environmental effects. Please confirm that this simplified model is appropriate.

## Hardware Integration

The simulator is designed to be integrated with physical hardware. The `*Controller` classes in each module are responsible for switching between the simulated and physical hardware. To use physical hardware, you will need to implement the `*Hardware` classes with the specific control logic for your devices.

The `PolarizationHardware` class provides an example of how to integrate with an STM32-based stepper motor controller for polarization modulation.
