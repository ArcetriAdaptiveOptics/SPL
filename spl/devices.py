# SPL/spl/devices.py
# -*- coding: utf-8 -*-
"""
SPL Hardware Device Management.

This module is responsible for initializing and providing access to the hardware
components used by the SPL system, specifically the filter (motor) and the camera.
It centralizes the client creation for these devices, utilizing configuration
parameters defined in `spl.conf.configuration`.

Key functionalities:
- Initializes plico_motor client for the filter.
- Initializes pysilico client for the camera.
- Provides getter functions (e.g., `get_filter_client`, `get_camera_client`)
  that cache client instances to ensure a single instance is used throughout
  the application (singleton-like behavior per client).
- Offers a convenience function `initialize_hardware()` to set up all devices.
- Offers a `shutdown_hardware()` function to clear cached clients (and could
  be extended for explicit device disconnection if required).

This module helps in decoupling hardware interaction logic from the application
logic (e.g., data acquisition, GUI), promoting reusability and easier
maintenance.
"""

__author__ = "INAF Arcetri Adaptive Optics" # Or your specific name/team
__version__ = "0.1.0" # Or align with your project versioning
__date__ = "2025-05-16" # Or current date

import logging
from .conf import configuration as config

# Actual hardware control libraries
from plico_motor import motor as plico_motor_client # aliasing to avoid name clash if motor is a common variable name
import pysilico

logger = logging.getLogger(__name__)

# Global cache for initialized devices to ensure they are singletons if needed
_filter_client = None
_camera_client = None

def get_filter_client(reinitialize: bool = False):
    """
    Initializes and returns the filter client (motor).
    Uses cached instance unless reinitialize is True.
    """
    global _filter_client
    if _filter_client is None or reinitialize:
        try:
            # Use pre-processed attributes from spl.conf.configuration module
            filter_host = config.IPFILTRO
            filter_port = config.PORTFILTRO # This is already an int from configuration.py

            filter_device_index = 0 # Assuming this is static for now, or could be added to config

            logger.info(f"Initializing filter client (motor) at {filter_host}:{filter_port}, device index: {filter_device_index}")
            _filter_client = plico_motor_client(filter_host, filter_port, filter_device_index)
            logger.info("Filter client (motor) initialized successfully.")
        except AttributeError as ae:
            logger.error(f"Failed to get filter server/port from configuration module (spl.conf.configuration): {ae}. Ensure IPFILTRO and PORTFILTRO are defined there.")
            _filter_client = None
            raise
        except Exception as e:
            logger.error(f"Failed to initialize filter client (motor): {e}")
            _filter_client = None # Ensure it's None on failure
            raise # Re-raise the exception so the application knows it failed
    return _filter_client

def get_camera_client(reinitialize: bool = False):
    """
    Initializes and returns the camera client (pysilico).
    Uses cached instance unless reinitialize is True.
    """
    global _camera_client
    if _camera_client is None or reinitialize:
        try:
            # Use pre-processed attributes from spl.conf.configuration module
            camera_host = config.IPCAMERA
            camera_port = config.PORTCAMERA # This is already an int from configuration.py

            logger.info(f"Initializing camera client (pysilico) at {camera_host}:{camera_port}")
            _camera_client = pysilico.camera(camera_host, camera_port)
            logger.info("Camera client (pysilico) initialized successfully.")
        except AttributeError as ae:
            logger.error(f"Failed to get camera server/port from configuration module (spl.conf.configuration): {ae}. Ensure IPCAMERA and PORTCAMERA are defined there.")
            _camera_client = None
            raise
        except Exception as e:
            logger.error(f"Failed to initialize camera client (pysilico): {e}")
            _camera_client = None # Ensure it's None on failure
            raise
    return _camera_client

def initialize_hardware():
    """
    Convenience function to initialize all hardware components.
    Returns:
        tuple: (filter_client, camera_client)
    """
    logger.info("Attempting to initialize all hardware...")
    filter_c = get_filter_client()
    camera_c = get_camera_client()
    if filter_c and camera_c:
        logger.info("All hardware components initialized.")
    else:
        logger.warning("One or more hardware components failed to initialize.")
    return filter_c, camera_c

def shutdown_hardware():
    """
    Placeholder for any necessary hardware shutdown/cleanup.
    For plico clients, this might involve specific disconnection methods.
    Pysilico and plico_motor clients might not require explicit shutdown,
    or might be handled by their destructors or context managers if used.
    """
    global _filter_client, _camera_client
    logger.info("Shutting down hardware (clearing cached clients)...")
    
    # Example of explicit disconnect if the clients have such a method:
    # try:
    #     if _filter_client and hasattr(_filter_client, 'disconnect'):
    #         logger.info("Disconnecting filter client...")
    #         _filter_client.disconnect()
    #     if _camera_client and hasattr(_camera_client, 'disconnect'):
    #         logger.info("Disconnecting camera client...")
    #         _camera_client.disconnect()
    # except Exception as e:
    #     logger.error(f"Error during explicit hardware disconnect: {e}")

    _filter_client = None
    _camera_client = None
    logger.info("Hardware client cache cleared.") 