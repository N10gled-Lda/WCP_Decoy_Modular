"""Utility functions for statistics."""
import logging
import numpy as np

def calculate_qber(sent_bits: list[int], received_bits: list[int], basis_match: list[bool]) -> float:
    """
    Calculates the Quantum Bit Error Rate (QBER).

    Args:
        sent_bits: The list of bits sent by Alice.
        received_bits: The list of bits received by Bob.
        basis_match: A list of booleans indicating if the bases matched for each bit.

    Returns:
        The calculated QBER.
    """
    logging.info("Calculating QBER.")
    
    mismatched_bits = 0
    compared_bits = 0
    
    for i in range(len(sent_bits)):
        if basis_match[i]:
            compared_bits += 1
            if sent_bits[i] != received_bits[i]:
                mismatched_bits += 1
    
    if compared_bits == 0:
        return 0.0
        
    qber = mismatched_bits / compared_bits
    logging.info(f"QBER calculated: {qber:.4f}")
    return qber
