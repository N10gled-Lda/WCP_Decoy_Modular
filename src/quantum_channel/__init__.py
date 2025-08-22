"""Quantum channel components."""

# Enhanced channel simulator
from .free_space_channel import (
    FreeSpaceChannelSimulator,
    ChannelConfig,
    ChannelStatistics,
    FreeSpaceChannel  # Legacy class
)

# Simple channel
from .simple_channel import (
    SimpleQuantumChannel,
    SimpleChannelConfig
)

# Eavesdropping simulator
from .eavesdropping import Eavesdropping

__all__ = [
    # Enhanced channel
    "FreeSpaceChannelSimulator",
    "ChannelConfig",
    "ChannelStatistics", 
    "FreeSpaceChannel",  # Legacy
    
    # Simple channel
    "SimpleQuantumChannel",
    "SimpleChannelConfig",
    
    # Security
    "Eavesdropping",
]