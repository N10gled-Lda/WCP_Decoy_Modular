# 4. System control and simulation (ML)

## 4.1. Introduction

### 4.1.1. Purpose and Design Philosophy

Next, we describe a control system for the QKD system that has its core principles modularity, adaptability, and maintainability. The system will prepare and measure the BB84 protocol using weak-coherent pulses (WCP) with decoy states, and control all the necessary components to generate, transmit, and receive the quantum states, as well as to perform the classical post-processing steps required for key generation.
The control system is designed to control the each component of the QKD system, both in hardware with **controlling** the physical devices (the laser, the VOA, the polarization controller, detectors, time tagger, etc.) and in software with the classical post-processing of the data (key sifting, error reconciliation, privacy amplification, etc.), but also to simulate the individual components or even the entire system.

- Modular, meaning that each component can be developed, tested, and replaced independently. This allows for easy evolution of the system as new technologies or methods become available, without requiring complete redesign of the entire system. As example, the laser simulator can be replaced with the real laser atuator on the hardware without changing how alice interacts with the laser controller. For that reason, each physical component and process is modeled as a distinct block with clearly defined interfaces. This allows for an evolutionary development approach, where initial simple models can be progressively replaced with more sophisticated and realistic ones without redesigning the entire system.
- Adaptable, meaning that it can be easily configured to work with different hardware setups or protocols. This is achieved through the use of configuration files (in JSON or YAML format) that define the parameters for each component and the overall workflow in a easy human-readable format. The system can be easily extended to support new hardware components by adding new configuration options and modules.
- Maintainable, meaning that the system is designed to be easy to understand, modify and maintain the codebase (include logging and debugging features), streamlining the process of bug fixes and overall software maintenance. This is achieved through clear documentation, modular design, and the use of high-level programming languages that are easy to read and write. It will also be more efficient in terms of resource utilization, avoiding redundant processes and optimizing resource allocation which will lead to better performance and reduced resource overhead.

### 4.1.2. Overview

The control system consists of several key components that work together to implement the protocol The main components are:

- **Alice**: The sender of the quantum states, responsible for generating weak-coherent pulses (WCP) and performing the necessary classical post-processing steps.
- **Bob**: The receiver of the quantum states, responsible for measuring the received pulses and performing the necessary classical post-processing steps.
- **Quantum Channel**: The medium through which the quantum states are transmitted from Alice to Bob (Simulation).
- **Classical Channel**: The medium through which the classical information is transmitted between Alice and Bob (Simulation).

In both Alice and Bob, the system will control their hardware components or simulated versions of those components, such that their CPU will be able to configure the hardware and perform the classical post-processing steps required for key generation. For that reason, the Alice CPU will have control modules for the laser sources, a polarization controller (which manages basis/bit logic), the variable optical attenuator (VOA) (which manages a fixed attenuator and decoy-state logic), and the QRNG. While the Bob CPU will have control modules for the detectors and time tagger. Both have a classical post-processing module for sharing information necessary for key sifting, error reconciliation, and privacy amplification steps to generate the final key.

The simulation of the quantum channel will include models for loss, depolarization, and other effects that can occur during the transmission of quantum states. The classical channel simulation will include models for the communication protocols by socket used to exchange classical information that takes into account the latency and bandwidth limitations as well as authentication and integrity checks.

To configure the system to test/work in different environments, a JSON or YAML configuration file will be used to define the parameters for each component and the overall workflow. This configuration file will allow users to easily modify the system's behavior without changing the underlying code. The system will also include logging features to store results and debug information, which can be used for statistical security analysis.

In the next image 1 follows a high-level overview of the system architecture, showing the main components and their interactions.

Figure 1: High-level overview of the QKD control system architecture showing the main components and their interactions.

```text
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ALICE (Transmitter)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐     ┌─────────┐    ┌─────────┐     ┌──────────────┐    ┌─────────┐ │
│  │  QRNG   │───▶│  Laser  │───▶│   VOA   │───▶│ Polarization │───▶│ Alice   │ │
│  │(Random  │     │(Trigger │    │(Decoy   │     │ Controller   │    │  CPU    │ │
│  │ Bits)   │     │& Pulse) │    │States)  │     │(BB84 Basis)  │    │(Control)│ │
│  └─────────┘     └─────────┘    └─────────┘     └──────────────┘    └─────────┘ │
└─────────────────────────────────────────┬───────────────────────────────────────┘
                                          │ Quantum Pulses
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                            QUANTUM CHANNEL                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  • Atmospheric Attenuation (weather, turbulence)                        │   │
│  │  • Polarization Drift & Mode Dispersion                                 │   │
│  │  • Background Noise & Timing Jitter                                     │   │
│  │  • Optional Eavesdropping Simulation                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────┬──────────────────────────────────────┘
                                          │ Attenuated Pulses
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                               BOB (Receiver)                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐     ┌──────────────┐    ┌─────────────┐     ┌──────────────┐     │
│  │   Bob    │◀───│ Time Tagger  │◀───│  Detectors  │◀───│ Optical Table│     │
│  │   CPU    │     │(Timestamps & │    │(H,V,D,A +   │     │(Passive 50/50│     │
│  │(Analysis │     │ Coincidence) │    │ Statistics) │     │Basis Choice) │     │
│  │& Control)│     └──────────────┘    └─────────────┘     └──────────────┘     │
│  └──────────┘                                                                  │
└────────────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │ Classical Information
                                          ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                          CLASSICAL CHANNEL                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  • Basis Sifting & Error Estimation                                     │   │
│  │  • Error Reconciliation (Cascade Protocol)                              │   │
│  │  • Privacy Amplification (Universal Hashing)                            │   │
│  │  • Authentication & Message Integrity                                   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
```

## 4.2. Software Architecture

As outlined in the previous section and shown in Figure 1, the software architecture of the QKD control system is divided into several modules, each responsible for a specific part of the system. This modular design ensures a degree of independence, allowing for replacement or modification of one part without impacting the entire system. The modules are grouped by functionality, such as Alice, Bob, Components, Channels, and Classical Post-Processing.

Within the Components, a distinction is made between controllers and actuators/drivers. Controllers handle the configuration and logic for devices, while actuators interface directly with hardware (detectors, modulators, pointing systems) or their simulated counterparts. This separation of concerns allows the system to operate consistently whether connected to real instruments or simulation environments.

Both Alice and Bob maintain their own modules to manage quantum state preparation and detection, and control the hardware or simulated components. The quantum and classical channels will have their own modules that simulate the transmission of quantum states and classical information, respectively. The Quantum Channel models the optical transmission medium, including realistic impairments such as atmospheric turbulence, pointing errors, loss, and depolarization—critical factors in satellite-to-ground links. The Classical Channel is used to exchange relevant information for key generation between Alice and Bob, in particular in post-processing. The Classical Post-Processing module performs the stages required to generate a secure key: basis sifting, error reconciliation, and privacy amplification, based on information exchanged over the classical channel.

**Multi-Threading Component:**
In the ground–satellite QKD system, once qubits are transmitted and detected, the post-processing workflow must transform raw detection events into a secure final key. This involves sequential steps such as basis sifting, error reconciliation, and privacy amplification. If executed as one long sequential process on large blocks of data, this would introduce significant latency and risk producing no key if a satellite pass ends before completion.
To mitigate this, a crucial aspect is the handling of data packets. Rather than processing one very large block of quantum states sequentially and then processing, the transmission is divided into finite-size packets, each containing a defined number of qubits (later converted into classical bits). Once a packet is received and handed off to the FPGA/CPU for post-processing, freeing the detectors to continue receiving new qubits while previous packets are still being processed.
This allows for a continuous flow of quantum states and continuous obtention of a secret key rate. This is particularly advantageous for satellite passes where the contact time is limited or cloudy moments that can interrupt the link, allowing to maximize key generation within the window.

The packet size is subject to a trade-off:

- Smaller packets enable faster parallelization and reduce latency, but the overhead in processing grows with the number of packets and is limited by the number of threads the system can handle in parallel, and the limit in the minimum size of a key from the security proof needs to be respected.
- Larger packets reduce relative overhead but introduce longer delays before keys can be reconciled, risking partial key loss if the satellite-ground contact ends before completion.

The key point is then the use of multi-threaded pipelines, where each Thread represents the full sequential workflow for one packet, and the entire post-processing chain (except for the final privacy amplification) is executed concurrently across multiple threads.

1. Preparation/Sending and Detection of the quantum states of that thread - using the controllers of the hardware - (after finishing the reception of the previous packet, it is ready, after agreement from both sides, the next packet can be sent/next thread)
2. Basis Sifting: selecting the detection results where Alice and Bob chose compatible bases.
3. Parameter Estimation: Estimating the quantum bit error rate (QBER) and other relevant parameters (including the decoy statistics)
4. Error Reconciliation: running an independent instance of the algorithm to correct the noisy sifted key.
5. Corrected Key Output: producing a verified sub-key ready before privacy amplification.

Multiple threads run concurrently, processing different packets in parallel. This pipelined architecture ensures that while one packet is being reconciled, another may be sifting, and yet another may already have produced a corrected sub-key. At the end of the session, all corrected sub-keys are collected and passed to a single Privacy Amplification stage. This step compresses the information into a shorter but more secure Final Secret Key. This process can be demonstrated in the following figure X.

Figure X: Demonstration of the multi-threaded pipeline architecture of the classical channel post processing steps.

### 4.2.1. Modules

Each module has a specific purpose, and will be described in detail in a subsequent section, but here is a brief overview of the main modules, some key inputs/outputs and their responsibilities

Table 1: Overview of the main modules and their responsibilities

| Block                                      | Brief Description/Responsibilities                                                                                    | Key Inputs                                                      | Key Outputs                                                   |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| **Control CPU Alice** | Orchestrates Alice's simulator/hardware components, generates per‑pulse commands by coordinating the transmission process. Controls the post processing process (BS, PE, decoy analysis, ER, PA) with Bob | Run/Protocol configuration setting | Final Secret Key, run metadata |
| **Laser Source** | Modular drivers (Simulation / Hardware) that controls the generation of individual pulses (triggering of a specific laser from the array, each pre-configured with a specific state). | Sim: Lasers info, pulse queue Har: port device, laser index to trigger | A stream of pulses with pre-set polarization and intensity and their timing |
| **Quantum Random Number Generator (QRNG)** | Source of pseudo random bits(simulation) for protocol choices - type/basis/bit. Provide 2 modes: obtain all necessary bits before or one-by-one as is necessary. | (Entropy source/seed), Key length.                              | Random bits for type of pulse, basis, and bit value.      |
| **Polarization Encoder** | Modular drivers (simulation/hardware) that encodes BB84 bit and basis onto polarization state. | Basis probabilities, basis, bit. Sim: Laser pulse and polarized pulse queue | Hard: A logical choice of polarization state for laser selection. Sim: A stream of pulses with the corrected polarization |
| **VOA Encoder** | Modular drivers (simulation/hardware) that sets the attenuation for decoy state selected (signal, weak, vacuum) based on probabilities. At initialization, its driver sets the session's overall fixed attenuation. | Target `μ_type` and corresponding probabilities. Hard: port for fixed attenuator. Sim: Laser Info, polarized pulses and attenuated pulses queue | Hard: A logical choice of decoy type (e.g., 'signal') for laser selection. Sim: A stream of pulses with the corrected attenuation |
| **Quantum Channel Model + Atmosphere**     | Simulation of the quantum channel transmission that models effects like loss from several effects (link budget), depolarization?, optional eavesdropper model, ...                                      | Alice simulation pulses, channel/atmosphere parameters (`α`, `e_d`, `p_dc`, ...).         | Attenuated and/or noisy pulses.                |
| **Classical Communication Channel**        | Simulation of the classical communication channel between Alice and Bob, including effects like delay, and bandwidth limitations. | Channel parameters (e.g., bandwidth, use of authentication). | N.A. |
| **Optical Table Components (OT) and Detectors**          | Simulation of the passive basis selection and routing to the corresponding detectors. As well as simulation of efficiency, dark counts, dead time, ... effects on the detectors.                                  | Incoming simulated pulses from quantum channel and Detector information .                                                 | Detection event: `⟨detector_id, time⟩` |
| **Time Tagger & Data Acquisition**         | Records the real detection events with Hardware (like swabian instruments) and saves for post processing. | Necessary protocol information to record events. | Time‑tags list/log of the detected events |
| **Control CPU Bob**                        | Orchestrates Bob's simulator/hardware components coordinating the receiving process. Controls the post processing process (BS, PE, decoy analysis, ER, PA) with Alice. | Run/Protocol configuration setting | Final Secret Key, run metadata (gains `Q_μ`/ bounds Y₀,Y₁ `Y_0`, `Y_1` / error `E_μ` & QBER).    |

### 4.2.2. Workflow and Control Flow

The system operates according to the following high-level workflow:

1. **Initialization Phase**: Both Alice and Bob initialize their hardware/simulator components, establish classical communication channel, and confirm the run protocol settings.

2. **Quantum Transmission Phase**:
   - Alice's CPU requests random bits from the QRNG.
   - The CPU uses these bits to query the *VOAController* to logically select a decoy type (signal, weak, or vacuum).
   - The CPU queries the *PolarizationController* to logically select a polarization state (H, V, D, or A).
   - The CPU commands the *LaserController* with the complete state information. The controller triggers the single, corresponding pre-configured laser (or no laser for a vacuum state).
   - The generated pulse is transmitted through the quantum channel.
   - Bob performs passive basis selection and detects the incoming pulses (in case of simulation)
   - Bob records detection events with precise timing, ready for post-processing.

3. **Classical Post-Processing Phase (through classical channel)**:
   - Basis Sifting (BS) and Parameter Estimation (PE)
   - Decoy State Analysis to confirm the presence of eavesdropping with PNS (through statistical methods)
   - Error Reconciliation (ER) using cascade protocol
   - Privacy Amplification (PA) for Final Secure Key generation.

## 4.3. Module Breakdown / Detailed Module Specifications

### 4.3.1. Transmitter Modules (Alice)

#### 4.3.1.1. Laser Controller & Laser Hardware/Simulator Driver - OLD VERSION

The laser source subsystem generates the optical pulses that encode the quantum states transmitted by Alice. It exposes a single high-level control surface to the rest of the transmitter stack (Alice CPU), while delegating hardware-specific behavior to pluggable drivers. This separation allows seamless switching between simulated and hardware modes for development and testing, including simulation, without changes at the call sites. In the overall architecture, the laser is the first actuation block in Alice’s optical chain and feeds the polarization and VOA encoder along the preparation pipeline.

***LaserController* Architecture**:
At the top level, a LaserController maintains protocol-level state (initialized/active/continuous flags, counters) and exposes a minimal set of operations used by the orchestrator: single-shot emission, and continuous pulse trains. The controller implements defensive checks (e.g., “initialized before use”, “no overlapping activities”) and centralizes status reporting of the controller and own driver-reported telemetry. This controller delegates all physical timing and I/O to a concrete driver selected at construction time (simulation or hardware).

- **Purpose**: High-level laser control abstraction independent of implementation
- **Potential Key Methods**:
  - `initialize()/shutdown()`: Prepares the controller for operation and handles power control with safety interlocks
  - `trigger_once()`: Single pulse emission
  - `start_continuous(rep_rate_hz)`: Continuous emission at specified rate
  - `stop_continuous()`: Cease continuous emission
  - `get_status()`: Current state and telemetry of the controller and driver
- **Inputs**: Driver in cause (simulated or hardware)

**Driver Implementation**:

1. ***SimulatedLaserDriver***:
Designed to mimic the behavior of a real laser system for development and testing purposes. It generates pulse events based on the laser information provided as input to best reproduce the behaviour of a real laser. It will use a queue-based architecture to manage pulse events and timing and easily deliver them to the next stage in the optical chain. In the end it initializes each pulse with a initial polarization state, and a certain number of photons given the laser characteristics.

2. ***HardwareLaserDriver***:
Using a arbitraty waveform generator to genreate rising-edge triggers to fire the laser. Allows for managing duty cycle and frequency of the send signal, with a simple internal state machine (OFF → ON/READY → FIRING/CONTINUOUS). A background monitor thread polls device status for basic health and counter updates.

#### 4.3.1.1 Laser Controller & Laser Hardware/Simulator Driver - NEW ARCHITECTURE

The laser source subsystem is the core of state generation, responsible for emitting optical pulses that encode the quantum states. In this architecture, it consists of an array of eight individual lasers. Each laser's output is pre-calibrated to a specific BB84 state (H, V, D, A) and intensity level (signal or weak decoy). This design replaces a single laser and dynamic modulators with a direct selection model, where choosing a state translates to triggering a specific laser. This subsystem acts as the final actuator in the state preparation chain. It receives commands that specify a complete quantum state and translates them into a low-level trigger for the single correct laser in the bank.

***LaserController* Architecture**:
The *LaserController* orchestrates the entire laser bank. It exposes a simple interface to be commanded by higher-level controllers (like the Polarization and Intensity controllers via the Alice CPU). Its sole responsibility is to map a fully specified state to a physical laser and fire it. For a vacuum state, it ensures no laser is triggered. The controller implements defensive checks (e.g., “initialized before use”, “no overlapping activities”) and centralizes status reporting of the controller and own driver-reported telemetry.

- **Purpose:** High-level control abstraction for an array of lasers.
- **Potential Key Methods:**
  - *initialize()/shutdown()*: Prepares all lasers for operation, including safety interlocks.
  - *trigger_laser()*: The core function. It takes the basis, bit, decoy_type state parameters that were set by the polarization and VOA controllers, maps them to the correct laser index, and triggers it. If decoy_type is vacuum, no laser is fired.
  - *set_polarization(basis, bit)*: Configures the polarization state for the next pulse.
  - *set_intensity(decoy_type)*: Configures the intensity level (signal or weak or vacuum decoy) for the next pulse.
  - *get_status()*: Reports the status of the controller and the individual lasers.
- **Inputs:** A driver instance (hardware or simulated), and a mapping configuration that links (basis, bit, decoy_type) tuples to physical laser indices.

**Driver Implementation**:

1. ***SimulatedLaserDriver***: Mimics the behavior of the laser. It generates a pulse event corresponding to the selected laser's pre-configured polarization and intensity. It uses a queue-based architecture to deliver the resulting pulse to the next stage in the simulation pipeline. (USE OF SIMULATOR OF DYNAMIC POLARIZATION AND ATTENUATION AND A SINGLE LASER???)

2. ***HardwareLaserDriver***: Interfaces with the physical hardware that controls the 8 lasers (e.g., a multi-channel pulse generator or FPGA). It translates the *trigger_laser* command into the appropriate low-level signals to fire the correct laser for a single pulse duration.

#### 4.3.1.2. Quantum Random Number Generator (QRNG)

The QRNG subsystem has its purpose to supply the randomness necessary to Alice’s orchestration in: (i) BB84 basis/bit selection in the polarization encoder, and (ii) decoy-state selection in the VOA. The design will accommodate a simulation with a pseudo-random number generator for the BB06 phase, but also easily adapt to include a control of the hardware of a real QRNG for TAQS.
The QRNG exposes a simple interface so that Alice can request a single bit unbiased or with a specified probability, request batches of bits, and for testing (3 modes). The simulator additionally supports explicit seeding to enable reproducible end-to-end tests and traceable failures during integration.

- **Operation Modes**:
  - `STREAMING`: For single bit generation in real-time operation
  - `BATCH`: For generating batches of random bits (e.g. pre-generated bits for optimal performance)
  - `DETERMINISTIC`: Seeded PRNG for reproducible testing
- **Potential Key Methods**:
  - `get_random_bit(mode, size, probability)`: with parameter size>1 for batch mode, and probability for biased bits
  - `set_seed(seed)`: set the seed for the random number generator (for deterministic mode)
- **Input**: Mode, Optional seed and size (depending on the mode)

#### 4.3.1.3. Polarization Controller & Polarization Hardware/Simulator Driver - OLD VERSION

The polarization subsystem prepares BB84 polarization states for Alice given the random bits from the QRNG it maps a (basis, bit) pair into one of {H, V, D, A} polarizations and controls the corresponding linear polarization angle on the actuation hardware. This module sits after the pulse generation stage and before the attenuation of the pulses channel launch, ensuring that every emitted pulse carries the intended state in the selected basis.

**BB84 Efficient Protocol Implementation**:
Maps basis and bit selections to specific polarization angles, each base with a different probability. MISSING EXPLAINING THE BB84 EFFICIENT PROTOCOL?:

- **Z Basis (Rectilinear)**: Bit 0: H polarization (0°); Bit 1: V polarization (90°).
- **X Basis (Diagonal)**:Bit 0: D polarization (45°); Bit 1: A polarization (135°).

***PolarizationController* Architecture**:

A PolarizationController coordinates two concerns: (1) selection of basis/bit (by default via QRNG), and (2) issuing the appropriate angle/state command to an interchangeable driver. Internally it tracks the selected basis, bit, state, and angle, and surfaces context-manager semantics for safe init/shutdown. The controller implements the previous BB84 efficient mapping (map_basis_bit_to_polarization).

- **Purpose**: High-level polarization control abstraction independent of implementation
- **Potential Key Methods**:
  - `initialize()/shutdown()`: Prepares the controller for operation and handles power control with safety interlocks
  - `map_basis_bit_to_polarization(basis, bit)`: Maps basis/bit pair to polarization angle/state
  - `set_polarization_angle(angle_deg)`: Direct angle setting in degrees
  - `set_polarization_random()`: Randomized polarization state setting through QRNG and given the probabilities
  - `add_noise()`: Adds simulated noise to the polarization state
  - `get_status()`: Retrieves operational telemetry
- **Inputs**: Driver in cause (simulated or hardware), QRNG instance (if applicable), Basis probabilities

This similarity ensures the simulator and the physical interface drop-in replacements, enabling SIL/HIL testing with identical controller code paths.

**Driver Implementation**:

1. ***SimulatorPolarizationDriver***: The simulator mirrors the hardware API and adds two queues—raw pulses in from laser, polarized pulses out—so higher layers can exercise end-to-end pipelines without lab hardware. It tracks current angle/state and saves for later use, applies the current polarization to queued pulses, and can add an estimate rotation time to better model the time it takes to control the real hardware. In the end it initializes each pulse with the correct polarization state as the hardware would, with the possibility to add some random statistical noise like the real photons would have.

2. ***HardwarePolarizationDriver***: The hardware driver interfaces with a physical polarization controller via serial commands. It implements the same interface as the simulator, ensuring compatibility. The driver manages connection setup, output format of the given angle (e.g. voltage, motor position, ...), and response parsing, providing real-time feedback.

#### 4.3.1.3 Polarization Controller & Polarization Hardware/Simulator Driver - NEW ARCHITECTURE

The polarization subsystem is responsible for selecting the BB84 polarization state for each pulse. While the current hardware uses a bank of pre-polarized lasers, this controller is maintained as an abstraction layer. This ensures modularity, allowing the underlying hardware to be changed to a dynamic modulator without altering the high-level control flow. In this configuration, the controller's role is to process a random choice, determine the target polarization, log it, and provide this information for the final laser selection.

**BB84 Efficient Protocol Implementation**: Maps basis and bit selections to specific polarization angles, each base with a different probability

- **Z Basis (Rectilinear - Bit 0)**: Bit 0: H polarization (0°); Bit 1: V polarization (90°).
- **X Basis (Diagonal - Bit 1)**: Bit 0: D polarization (45°); Bit 1: A polarization (135°).

***PolarizationController* Architecture:**
A PolarizationController coordinates the logical choice of polarization. It takes random inputs and maps them to one of the four BB84 states, saving the necessary information for post-processing. It does not directly control a dynamic modulator but serves as the authoritative source for the polarization choice for each pulse.

- **Purpose:** High-level, implementation-independent manager for polarization state selection.
- **Potential Key Methods:**
  - *initialize()/shutdown()*: Verifies the system is ready for polarization selection.
  - *map_basis_bit_to_polarization(basis, bit)*: Maps basis/bit pair to polarization state/angle
  - *select_polarization_state()*: Uses the QRNG to select a basis and bit, determines the corresponding polarization state {H, V, D, A}, and returns this choice.
  - *add_noise()*: Adds simulated noise to the polarization state in the simulation
  - *get_last_state()*: Retrieves the most recently selected polarization state for logging and control purposes.
  - *get_status()*: Reports operational statistics.
- **Inputs:** Driver in cause (simulated or hardware), QRNG instance, basis probabilities.  

**Driver Implementation:**

1. ***SimulatorPolarizationDriver:*** The simulator adds two queues—raw pulses in from laser, polarized pulses out—so higher layers can exercise end-to-end pipelines without lab hardware. It tracks current angle/state and saves for later use, applies the current polarization to queued pulses. In the end it initializes each pulse with the correct polarization state as the hardware would when selecting the laser, with the possibility to add some random statistical noise.
<!-- A simple pass-through driver. It mirrors the hardware API but performs no physical action. Its methods confirm that a logical polarization has been selected and are ready to be logged. -->

1. ***HardwarePolarizationDriver:*** In this architecture, this driver does not affect the physical polarization, at maximum actuates the components before the protocol to initialize them the hardware. During operation, its functions serve to validate and log the polarization choice that will ultimately be used to select a laser by giving the information of the selected polarization state to the LaserController.

#### 4.3.1.4. Variable Optical Attenuator (VOA) Controller & VOA Hardware/Simulator Driver - OLD VERSION

The VOA subsystem implements decoy-state intensity control for Alice by applying pulse-by-pulse attenuation that realizes the target mean photon numbers for signal, weak, and vacuum states. It exposes a unified control surface to the orchestrator (Alice CPU) and supports both hardware and simulator back-ends without changes at call sites. The VOA sits directly after the polarization preparation and before sending through the quantum channel.

**Decoy State Protocol Implementation**:
Mapping of decoy states to their corresponding mean photon numbers is achieved through a combination of QRNG-based selection and attenuation calculation.

- First, a biased bit with the probability $p_{signal}$ is generated to determine if the selected state is a signal or decoy (weak/vacuum).
- Second, a biased bit with the probability $p_{weak}/(p_{weak}+p_{vacuum})$ is generated to determine the specific decoy state (weak or vacuum).
- Finally, for the given state, the corresponding attenuation is calculated based on the target mean photon number $\mu$ of the state and the laser's output characteristics: NOT SURE IF THIS IS THE RIGHT WAY BUT CHANGE WHAT IS NECESSARY
  $$
  A_{dB}=10\log_{10}(\bar{\mu}/\mu_{target}) \qquad , \quad \bar{\mu} = \frac{E_\text{pulse}}{h\nu}= \frac{P_\text{avg}\,λ}{h c\,f_\text{rep}}.
  $$

Example of a 3-intensity decoy state configuration (compressed in a `DecoyInfo` data structure):

| State   | Mean Photon Number (μ) | Probability (p) | Typical Purpose           |
|---------|------------------------|-----------------|---------------------------|
| Signal  | 0.5                    | 0.7             | Key generation bits       |
| Weak    | 0.1                    | 0.2             | Channel estimation of single-photon yield and error rate (Y1 and e1 bound)          |
| Vacuum  | 0.0001                 | 0.1             | Background noise / dark counts estimation (Y0 bound)        |

***VOAController* Architecture**:

A VOAController coordinates three concerns: (1) decoy-state selection via a QRNG interface, (2) conversion from target mean photon number μ to attenuation in dB, and (3) issuing attenuation commands to a pluggable driver (either the simulator or the hardware). It will keep track of each state, attenuation, and an initialization flag; it can also provides setters to update the intensities and probabilities (with validation) of the decoy states configuration. A DecoyInfo data structure is provided when system is initiated and helps with easy getters/setter for intensities and probabilities and validation checks.

- **Purpose**: High-level VOA control abstraction independent of implementation
- **Potential Key Methods**:
  - `initialize()/shutdown()`: Prepares the controller for operation and handles power control with safety interlocks
  - `set_decoy_info (DecoyInfo)`: Sets the decoy state configuration (intensities and probabilities)
  - `update_intensities(new_intensities)`: Update target mean photon numbers with validation
  - `update_probabilities(new_probabilities, normalize=True)`: Update state probabilities with optional normalization check
  - `generate_pulse_with_probability_selection()`: Selects state based on configured probabilities using QRNG
  - `calculates_attenuation_for_state(state)`: Computes required attenuation for the selected state
  - `get_status()`: Retrieves operational telemetry
  - `get_output_from_attenuation()`: Converts dB attenuation to output format needed (depends on the hardware/simulator)
- **Inputs**: Driver in cause (simulated or hardware), QRNG instance (if applicable), Decoy state configuration (*DecoyInfo* - intensities and probabilities)

**Driver Implementation**:

All concrete drivers conform to a common BaseVOADriver contract (initialize/shutdown, set_attenuation, get_attenuation, and get_output_from_attenuation), ensuring interchangeability and enabling SIL/HIL parity.

1. ***SimulatorVOADriver***: It mirrors the hardware API and adds two queues—raw pulses in from the polarizer, attenuated pulses out - to emulate pipeline flow. Attenuation is applied via binomial “thinning” with probability $10^{−A_{dB}/10}$ to the number of photons of the simulated pulses, aligning with attenuation level by the hardware. In the end it initializes each pulse with the correct attenuation as the hardware would, similarly with the possibility to add some random statistical noise like the real photons would have.

2. ***HardwareVOADriver***: It binds a physical VOA via serial commands, implementing the same interface as the simulator, ensuring compatibility. The driver manages connection setup, output format of the given attenuation (e.g. voltage, ...), and response parsing, providing real-time feedback.

#### 4.3.1.4. Variable Optical Attenuator (VOA) Controller & VOA Hardware/Simulator Driver - NEW ARCHITECTURE

This subsystem manages the decoy-state intensity control for Alice. Similar to the PolarizationController, it functions as a high-level abstraction layer. Its responsibility is to select the intensity level (signal, weak, or vacuum) for each pulse based on the decoy state protocol probabilities. In the current hardware, this choice is realized by selecting a laser with a pre-calibrated intensity, but this controller decouples that logic from the main workflow. Meaning it will still be available to easily adapt the hardware for it to dynamically adjust the VOA for different operational scenarios. It is also responsible for setting the session's overall fixed attenuation during initialization.

**Decoy State Protocol Mapping**: Mapping of decoy states to their corresponding mean photon numbers is achieved through a combination of QRNG-based selection and attenuation calculation.

- First, a biased bit with the probability $p_{signal}$ is generated to determine if the selected state is a signal or decoy (weak/vacuum).
- Second, a biased bit with the probability $p_{weak}/(p_{weak}+p_{vacuum})$ is generated to determine the specific decoy state (weak or vacuum).

***VOAController* Architecture:**
The *VOAController* orchestrates decoy state selection. It uses a QRNG to choose a state, logs the choice, and provides this information to the Alice CPU for laser selection. It does not directly control a dynamic modulator but serves as the authoritative source for the intensity choice for each pulse. During initialization, it sets the fixed attenuation level for the session.

- **Purpose:** High-level, implementation-independent manager for decoy state selection and overall intensity configuration.
- **Potential Key Methods:**
  - *initialize()*: Configures the system for the run. Critically, this is where it would command its driver to set the fixed optical attenuator to the correct value for the session.
  - *select_decoy_state()*: Uses the QRNG and configured probabilities to select a state (Signal, Weak, Vacuum) and returns this choice.
  - *set_fixed_attenuation()*: Sets the fixed attenuation level for the hardware session.
  - *add_noise()*: Adds noise to the selected pulse.
  - *get_status()*: Retrieves operational telemetry.
- **Inputs:** Driver in cause (simulated or hardware), QRNG instance, Decoy state configuration (DecoyInfo), Optional fixed attenuation level for the session.

**Driver Implementation:**

1. ***SimulatorVOADriver:*** It adds two queues—raw pulses in from the polarizer, attenuated pulses out - to emulate pipeline flow. In the end it initializes each pulse with the correct attenuation as the hardware would do by selecting the appropriate laser. Similarly with the possibility to add some random statistical noise.

2. ***HardwareVOADriver:*** Its initialize method connects to the physical fixed optical attenuator and sets its value for the duration of the session. During operation, its *select_decoy_state* function is a logical step; it does not send per-pulse commands to hardware but ultimately selects a laser by giving the information of the selected decoy state to the LaserController.

#### 4.3.1.5. Alice CPU (System Orchestrator)

**Purpose/Role:**
The Alice CPU coordinates all transmitter-side subsystems (QRNG, Laser, VOA, Polarization) to produce a stream of BB84 decoy‑state pulses and the corresponding information required for coordinating the steps involved in post-processing. It provides a single orchestration surface for both simulation and hardware operation initiating, closing, and calling the transmission of each of the modules in the correct order, recording per‑pulse metadata (decoy class, basis, bit, polarization angle, timing) for downstream analysis and key distillation with Bob. In summary, this module acts as the “conductor” of the transmitter pipeline and the integration point for all components.

Since this system is design to handle multiple threads, the Alice CPU will also be in charge of managing the threads and each corresponding process involved. That way the control and launch of the different modules for the transmission of the smaller portion of qubits can be done at the same time of the post-processing elements via classical communication channel.

**Architecture Overview:**  
The orchestrator is configured via a general dataclass‑like settings (e.g., `AliceConfig`) with all of the necessary input for the construction of each module, both the controllers and the selected drivers (hardware or simulator), with the corresponding necessary information for them and for the post-processing. A queue-and-thread model decouples component latencies while sustaining the configured repetition rate.

**WorkFlow & Potential Key Functions**:

1) **Initialization:** establish classical channel session with Bob, initiate session-level handshake with Bob to establish parameters, thus building drivers per configuration, initialize controllers, clear queues. (`initialize_components`)
2) **Transmission Loop Threads:** starts and manages the individual threads for each module (QRNG, Laser, VOA, Polarization) to ensure synchronized operation and data flow in the right order, calling the relevant functions to operate each module. (`start_transmission`, `transmission_loop_individual_thread`, individual module caller (e.g. `set_laser`), `stop_transmission`). And a `get_data` function to retrieve the relevant data for the post-processing
3) **Post-Processing Threads:** similarly manages the individual threads for each post-processing step (BS, PE, ER, PA) via classical channel independent of the transmission loop, calling the relevant functions to operate each step. (`start_post_processing`, individual module caller (e.g. `do_basis_sifting`)).
4) **Terminating:** Terminates all process/threads/controllers and clean up the resources (`shutdown_components`). And getters to retrieve the final key and the relevant statistics (`get_final_key`, `get_simulation_results`, `get_component_statistics`).

**Inputs:**

**Summary Key Capabilities**:

- **Threading Architecture**: Separate threads for pulse generation, processing, and data collection
- **Timing Control**: Precise pulse period maintenance including processing overhead
- **Data Recording**: Complete transmission history for post-processing analysis
- **Error Handling**: Robust error recovery and logging
- **Performance Monitoring**: Real-time statistics and throughput measurement for easy debugging and error handling.

### 4.3.2. Channel Simulation Modules

#### 4.3.2.1. Quantum Channel Simulator

**Purpose and Scope**: This module has the main focus to emulate the effects of a real quantum link between Alice and Bob. It applies a link-budget–derived attenuation (in dB) and optional stochastic losses to the pulse stream, producing the same aggregate impact on mean photon number that a free-space path would impose. At run time, attenuation may be supplied directly (from a link model) or set by parameters with the model inside the simulator; ALSO AFFECTS THE POLARIZATION???

**Architecture Overview**: The simulator wraps a simple, configurable channel with two behaviors: (1) pass-through (no modification), (2) fixed attenuation in dB based on a link model. The simulator exposes a minimal interface to set parameters and apply the channel to individual pulses. The fixed attenuation can varied along the time passed in the process to simulate dynamic conditions of a overpass with varying distance of the satellite to the ground station. To apply to the pulses, an input queue (from the output of alice) is used and processed into a output queue accessed by bob detection modules.

**Potential Key Methods**:

- `set_attenuation(dB) / set_model(LinkBudgetModel)`: Sets/Updates the attenuation level in dB or the link budget model to use that was provided when initialized.
- `set_pass_through_mode(on/off)`: Enables or disables pass-through mode.
- `transmit_pulse(p/list[p])`: Transmits a single or a list of pulses through the channel returning the modified pulse or None if lost.

Effect models that can be included?????:

**Atmospheric Effects**:

- **Distance-based attenuation**: Configurable loss coefficient (0.2 dB/km baseline)
- **Weather modeling**: Clear, hazy, rain, fog conditions with appropriate loss factors
- **Atmospheric turbulence**: Time-varying transmission with realistic fluctuations
- **Background noise**: Poisson-distributed background photons based on detection window

**Polarization Effects**:

- **Drift modeling**: Linear drift rate with random fluctuations
- **Mode dispersion**: Small timing variations due to polarization mode coupling
- **Depolarization**: Gradual loss of polarization coherence over distance

#### 4.3.2.2. Classical Channel Simulator

**Purpose and Scope**: This module provides a controllable, authenticated classical link between Alice and Bob for necessary steps. It is intentionally simulation-first: transport, latency, bandwidth, and buffering are configurable so the rest of the stack can be validated under realistic constraints before any network deployment. Additionally the design allows for the addition of Authentication

**Architecture Overview**:

- Endpoints & Transport Abstraction. Each participant runs a “role” endpoint that binds a local receive address and connects to its peer. The transport is a framed, length-prefixed byte stream over TCP-like sockets, with explicit peer configuration and lifecycle (init, connect, send, receive, shutdown). This design of python like sockets allows for easy integration with existing Python libraries and tools, facilitating rapid development and testing. However due to the limitations of Python's performance in high-speed scenarios, this python architecture is suitable for BB06 but not TAQS.
- Framing & Flow Control. Application payloads are split into frames with a compact length header; frames are reassembled on the receiver before delivery to the application inbox. Bounded, byte-accurate queues model finite buffers to create realistic back-pressure and overflow behaviors.
- Bandwidth & Latency Modeling. The simulator can enforces a configurable throughput cap and injects one-way latency (and thus RTT) so that protocol timings (e.g., Cascade round-trips) can be profiled.

**Operating Behavior**:

1. Initialization: endpoints start with configured capacities, limits, and authentication; peers exchange a session handshake, agree on parameters, then begin framed exchange.
2. Send Path: application messages → outbox (byte-bounded) → split into frames (length header) → timestamp → MAC → shaped by bandwidth/latency → socket.
3. Receive Path: socket → reframe/deframe → MAC verify → timestamp check (windowed) → reassembly → inbox delivery (byte-bounded).
4. Testing Harness: in-process mock sockets emulate a perfect link to isolate logic; latency/bandwidth knobs then reintroduce realistic conditions.

**Authentication & Anti-Replay**:

Communications on the classic channel generally operate over frequencies vulnerable to interception and transmissions from unknown parties. Because of this, without robust authentication mechanisms, a message sent by an attacker can be considered genuine if its details match messages exchanged in an authentic communication process. This is concerning, especially considering that key post-processing steps of the quantum key exchange occur through it.
Message Authentication Codes (MACs) are a common way to authenticate communications. MAC works by processing a message along with a symmetric key, which is shared between the sender and receiver, through a cryptographic algorithm. This generates a unique code that is appended to the message. Upon arrival, the receiver separates the code from the message and re-runs the algorithm. If the computed code does not match the received one, the message's integrity cannot be verified.
For our MAC algorithm, we selected AES-GMAC, an authentication-only variant of AES-GCM, which is widely used in TLS, the secure communication protocol on which most of the internet is based. It has proven cryptographic strength, high performance, and hardware acceleration support on modern Intel-based CPUs with AES-NI.

- **Message Types (BB06 scope)**: The simulator defines message classes that map directly to post-processing steps and orchestration:
  - **Session / Control**: handshake, capability negotiation (frame size, auth mode), heartbeat, orderly shutdown.
  - **Controller**: start/stop transmission by acknowledgment/"handshake"; report status, errors, and telemetry.
  - **Basis Sifting (BS)**: basis vectors or compressed basis masks; acknowledgments and pagination for large blocks.
  - **Parameter Estimation (PE)**: sampled indices and bit values; decoy-class statistics; sampling seeds.
  - **Error Reconciliation (ER)**: Cascade parity vectors, block indices, pass markers, verification hashes.
  - **Privacy Amplification (PA)**: Toeplitz descriptors; output length negotiation.
- **Potential Key Methods**:
  - `role.get_instance(...)` → Role: Factory that returns either a MAC-enabled or MAC-disabled role depending on whether a MAC configuration is provided. Key parameters include local socket info (`ConnectionInfo(ip, port)`), bandwidth cap, inbox/outbox capacities, simulated latency, and option to use authentication.
  - `role.peer_connection_info = ConnectionInfo(ip, port)`: Sets the remote peer’s address. This must be set before sending; setting it launches the internal sender/receiver worker threads that move frames over the socket.
  - `put_in_outbox(data: bytes, block=True, timeout=None) -> None`: Queues an application payload (bytes) to be sent. Respects an optional byte-capacity limit (back-pressure) and updates transfer/latency accounting when a bandwidth cap is configured.
  - `get_from_inbox(block=True, timeout=None) -> bytes`: Retrieves a reassembled application payload received from the peer (blocking or non-blocking).
  - `clean() -> None`: Gracefully stops all CCC threads (shutdown).
- **Inputs**: Local/Remote addressing, Bandwidth & capacity, Framing & latency, Authentication
  - **Addresses/ports:** `ConnectionInfo(ip, port)` for self; peer_connection_info for the other side.
  - **Bandwidth cap:** `bandwidth_limit_megabytes_per_second` (bytes/sec limiter).
  - **Latency:** `latency_seconds` (one-way), also affects derived RTT.
  - **Authentication:** `MAC_Config(key tag)` with a shared secret key of 16 bytes (128 bits) minimum.

### 4.3.3. Receiver Modules (Bob)

#### 4.3.3.1. Optical Table and Detector Simulator

**Purpose and Role**
This block models Bob’s front-end optics and detectors so the receiver pipeline can be exercised without lab hardware. It takes the polarized, attenuated pulses from the quantum channel, projects them onto the selected measurement basis (Z or X) using Malus’ law, and produces detector events on {H,V,D,A} consistent with efficiency, thresholds, and noise (like if it would passed through the beam splitter and other optics, and into the SNSPD detectors). In the overall system it sits between the quantum channel simulator and the time-tagger/data-acquisition stage.

**Architecture Overview**
The simulator is split into two cooperating components:

- **Optical Table Model** — given the incoming pulse, it computes which arm (H/V or D/A) the pulse would exit and the corresponding detection probability given the number of photons, with options for perfect vs. imperfect analysis and angular/basis misalignment. It exposes setters to choose basis and to inject controlled angular deviations.
- **Detector Model** — converts optical outcomes into actual clicks with a timestamp, with the option to apply quantum efficiency, dark counts, dead-time as noise. It returns detector numbers 0/1/2/3 (H,V,D,A) and maintains basic statistics and dark-count generators.

**Potential Key Methods:**

- `measure_detect_pulse(pulse)`: Based on the incoming pulse’s polarization - if closer to H/V, applies Malus’ law (cos²(Δθ)) to compute the detection probability of going to the detector of H or V (0 or 2), due to imperfections of the incoming pulse - if closer to D/A, applies Malus’ law (cos²(Δθ)) but dephased of $-45\degree$ for those detectors instead (1 or 3). Returns the detector ID measured depending only if the number of photons is above 1.
- `set_alignment_error(deg: float)`: Fixed basis misalignment (added to Δθ in Malus’ law).
- `set_random_deviation(std_deg: float)`: Enables per-pulse angular jitter (Gaussian, std in degrees).
- `set_efficiency(d: float)`: Sets the quantum efficiency (0–1).
- `set_dark_counts(rate: float, prob: float)`: Sets the dark count rate (Hz) and the probability to have a dark count per pulse.
- `set_dead_time(dt: float)`: Sets the dead time (s).
- `add_optical_noise()`: Adds simulated noise to the polarization state.
- `add_detector_noise()`: Adds simulated noise to the detector clicks.
- `get_data()`: Returns current measurement counts data for the timetagger / Bob CPU.

#### 4.3.3.2. Time Tagger Controller & TimeTagger Hardware/Simulator Driver

**Purpose and Role**:
The time-tagger subsystem records detection events from Bob’s front-end as timestamped clicks, one consolidated outcome per transmitted pulse.  It serves as a bridge between simulated/hardware detectors and the post-processing pipeline, offering a unified API to start/stop acquisition, configure channels, and retrieve (channel\_id, timestamp) events. In this design, time gating is performed by the detectors; the tagger simply collects edges within the broader per-pulse window, and—if more than one detetor reports for the same pulse—selects one at random for that pulse’s record.

**Controller Architecture**:
A *TimeTaggerController* selects and manages a concrete driver (hardware vs. simulator). It centralizes lifecycle (init, start/stop), selecting channel to the correspondent detector, and device info, while keeping call-sites identical across back-ends.

- **Potential Key Methods:**
  - `initialize()/shutdown()`: Prepares the controller for operation and handles power control with safety interlocks
  - `start_measurement()/stop_measurement()`: Arms/disarms the tagger for data acquisition
  - `get_timestamps(nb_events)`: Retrieves a batch of timestamped events (channel, time_ps)
  - `enable_channel(ch, enable)`: Enables/disables a specific channel
  - `get_count_information()`: Returns count information for each channel as a list of tuples (channel_id, timestamp).
- **Inputs**: Driver in cause (simulated or hardware), TimeTaggerConfig defines the configuration parameters (time of counter, channels used, ...)

**Driver Implementation**:

1. The hardware driver will bind a Swabian Instruments time-tagger. Integration focuses on channel setup (trigger/impedance), streaming timestamps at the configured resolution, and returning device health/throughput.
2. The simulator conforms to the same driver contract. Given detector outcomes for each pulse, it attaches timestamps outputting (channel, time_ps) events, and aggregates one “winner” per pulse if multiple arrive (random tie-break), since via security proof in these cases they can not be discarded. This mirrors how the hardware view is later post-processed.

#### 4.3.3.3. Bob CPU (Measurement Controller)

**Purpose and Role:**
The Bob CPU orchestrates all receiver-side subsystems—the quantum channel adapter, optical table, photon detectors simulators, and time-tagger controller to convert incoming detected pulses into timestamped and detector identification events. Similar to Alice CPU it provides a single orchestration operating the initializing, closing, and calling detection module to start/end measurements, maintaing data for post-processing. Classical-channel coupling for post-processing (BS/PE/ER/PA) is also planned and will bind to the same session/handshake model specified for Alice.

Since this system is design to handle multiple threads, the Bob CPU will also be in charge of managing the threads and each corresponding process involved. That way the control and launch of the different modules for the detection of the smaller portion of qubits can be done at the same time of the post-processing elements via classical communication channel.

**Architecture Overview:**
Bob is configured via a general dataclass-like setting (e.g. `BobConfig`) with all of the necessary input for the construction of each module, both the controllers and the selected drivers (hardware or simulator), with the corresponding necessary information for them and for the post-processing. A queue-and-thread model decouples component latencies while sustaining the configured repetition rate.

**WorkFlow & Potential Key Functions**:

1) **Initialization:** Establish classical channel session with Alice, build all necessary components from BobConfig: (i) the quantum channel in case of simulation, (ii) the optical table and detectors in case of simulation, and (iii) a time-tagger controller (sim or hardware) under a unified driver contract. Internally it tracks statistics and the data. A queue-and-thread loop drains pulses and drives the measurement pipeline at run-time.  (`initialize_components`)
2) **Measurement Loop Threads:** For each received pulse, in case of simulation apply quantum channel effect, and measurement process with some noise of the optical table and detector, then finalize with starting a measurement on the timetagger, and if a click occurs, it emits/saves a time-tag event on the mapped channel; Record per-pulse telemetry (time, basis and bit given the detector id) for later post processing. (`start_measurement`, `measurement_loop_individual_thread`, individual module caller (e.g. `measure_pulse`), `stop_measurement`). And a `get_data` function to retrieve the relevant data for the post-processing.
3) **Post-Processing Threads:** Once the measurement loop is complete, trigger the start of post-processing pipeline (BS, PE, ER, PA) via classical channel independent of the measurement loop, calling the relevant functions to operate each step. (`start_post_processing`, individual module caller (e.g. `do_basis_sifting`)).
4) **Terminating:** Ensure all threads are properly joined and resources released on shutdown. (`shutdown_components`). And getters to retrieve the final key and the relevant statistics (`get_final_key`, `get_simulation_results`, `get_component_statistics`).

### 4.4 Technical Considerations

The development of the QKD control and simulation platform requires careful attention to several technical aspects that extend beyond the implementation of individual modules. A primary consideration is the configuration and management of system parameters. Human-readable formats such as YAML or JSON provide a flexible means to define hardware connections, simulation modes, and run parameters. This ensures that the same framework can be adapted to different experimental setups or simulation campaigns with minimal modification of the codebase.

Equally important is the integration of monitoring and visualization tools. A graphical interface allows operators to supervise the status of Alice and Bob in real time, track error rates, and observe trends in key generation. Beyond convenience, this layer is critical for diagnosing anomalies, triggering alarms in case of degraded security margins, and presenting clear metrics such as QBER evolution, throughput, and finite-size effects. The GUI therefore functions as both a management console and a validation environment.

Finally, the platform must incorporate a robust validation and testing framework.
Individual components are subject to unit testing and hardware validation, while the integrated system must undergo end-to-end protocol verification. Stress tests—such as deliberate error injection, synchronization drift, or attack scenario emulation—serve to probe resilience and highlight points of improvement. From a security standpoint, the accurate estimation of decoy-state bounds, finite-key effects, and information leakage is vital to certify that the generated keys satisfy theoretical guarantees.

In combination, these technical considerations ensure that the QKD control and simulation platform is not only functional but also reliable, secure, and extensible. They provide the foundation for translating research prototypes into deployable systems that can withstand the operational and security requirements of satellite-based quantum communications.
