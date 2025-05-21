# SPL/spl/spl_gui.py
# -*- coding: utf-8 -*-
"""
SPL Graphical User Interface (GUI).

This module provides a Tkinter-based GUI for controlling the 
SPL (Spectro-Polarimeter) data acquisition, processing, and analysis workflows.
It allows users to set parameters for each step and initiate the processes.

Key Features:
- Tabbed interface for Acquisition, Processing, and Analysis stages.
- Input fields for all relevant parameters, w                logger.info(f"Looking for analysis image at: {image_path}")
                
                if os.path.exists(image_path):
                    # Load and display the image using matplotlib
                    img = plt.imread(image_path)
                    self.analysis_figure.clear()
                    ax = self.analysis_figure.add_subplot(111)
                    ax.imshow(img)
                    ax.axis('off')  # Hide axeslts where applicable.
- Uses the `spl.devices` module to obtain initialized hardware clients (filter, camera)
  which are then passed to the `SplAcquirer`.
- Interacts with `SplAcquirer`, `SplProcessor`, and `SplAnalyzer` classes from
  the respective SPL modules to execute the core logic.
- Provides a logging area to display messages, progress, and errors.
- Handles graceful shutdown, including attempting to shut down hardware resources.

This GUI aims to simplify the operation of the SPL system by providing a user-friendly
interface to its command-line functionalities.
"""

__author__ = "Marco Bonaglia & Gemini AI" # Or your specific name/team
__version__ = "0.2.0" # Incrementing version due to significant structure
__date__ = "2025-05-16" # Or current date

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import glob
import re

# Actual script imports using relative paths
from .SPL_data_acquirer import SplAcquirer
from .SPL_data_processer import SplProcessor
from .SPL_data_analyzer import SplAnalyzer
from .conf import configuration as config
from . import devices # Import the new devices module

# --- Filter and Camera Objects are now managed by devices.py and passed to SplGuiApp ---

logger = logging.getLogger(__name__) # Setup logger for the GUI module

class SplGuiApp:
    def __init__(self, master, filter_obj_instance, camera_obj_instance):
        self.master = master
        master.title("SPL Control GUI")
        master.geometry("800x700")

        self.filter_obj = filter_obj_instance
        self.camera_obj = camera_obj_instance
        self.acquisition_running = False  # Add flag to track acquisition state
        self.acquirer = None  # Will store the SplAcquirer instance

        self.notebook = ttk.Notebook(master)

        # --- Shared Variables ---
        self.tt_var = tk.StringVar()

        # --- Acquisition Tab ---
        self.tab_acquirer = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_acquirer, text='Acquisition')
        self._create_acquirer_widgets()

        # --- Processing Tab ---
        self.tab_processor = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_processor, text='Processing')
        self._create_processor_widgets()

        # --- Analyzer Tab ---
        self.tab_analyzer = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_analyzer, text='Analysis')
        self._create_analyzer_widgets()

        self.notebook.pack(expand=1, fill='both', padx=10, pady=5)

        # --- Log Area ---
        log_frame = ttk.LabelFrame(master, text="Logs & Results")
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)
        log_scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        log_scrollbar.pack(side=tk.RIGHT, fill='y')
        self.log_text['yscrollcommand'] = log_scrollbar.set

        # Create reference image display frame
        self.ref_image_frame = ttk.LabelFrame(self.tab_acquirer, text="Reference Image")
        # self.ref_image_frame.pack(padx=10, pady=10, fill='both', expand=True) # Removed initial pack
          # Create matplotlib figure and canvas for reference image display
        self.ref_figure = Figure(figsize=(6, 4), dpi=100)
        self.ref_canvas = FigureCanvasTkAgg(self.ref_figure, master=self.ref_image_frame)
        self.ref_canvas.get_tk_widget().pack(side=tk.TOP, fill='both', expand=True)
        
        # Add toolbar for navigation
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.ref_canvas, self.ref_image_frame)
        toolbar.update()

        if not self.display_reference_var.get():
            self.hide_reference_image()

    def _log_message(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(message) # Also print to console

    def _create_acquirer_widgets(self):
        frame = ttk.LabelFrame(self.tab_acquirer, text="Acquisition Parameters")
        frame.pack(padx=10, pady=10, fill='x')

        # Lambda Vector (simplified: start, stop, step)
        ttk.Label(frame, text="Lambda Start (nm):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.lambda_start_var = tk.StringVar(value="530") # Default from SplAcquirer example
        ttk.Entry(frame, textvariable=self.lambda_start_var, width=10).grid(row=0, column=1, padx=5, pady=2, sticky='ew')

        ttk.Label(frame, text="Lambda Stop (nm):").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.lambda_stop_var = tk.StringVar(value="730") # Default from SplAcquirer example
        ttk.Entry(frame, textvariable=self.lambda_stop_var, width=10).grid(row=1, column=1, padx=5, pady=2, sticky='ew')

        ttk.Label(frame, text="Lambda Step (nm):").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        self.lambda_step_var = tk.StringVar(value="10") # Default from SplAcquirer example
        ttk.Entry(frame, textvariable=self.lambda_step_var, width=10).grid(row=2, column=1, padx=5, pady=2, sticky='ew')
        
        # Use config defaults if fields are empty
        ttk.Label(frame, text=" (Uses config if empty)").grid(row=0, column=2, rowspan=3, padx=5, pady=2, sticky='w')

        ttk.Label(frame, text="Exposure Time (ms):").grid(row=3, column=0, padx=5, pady=2, sticky='w')
        self.exptime_var = tk.StringVar(value="100") # Default from SplAcquirer example
        ttk.Entry(frame, textvariable=self.exptime_var, width=10).grid(row=3, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(frame, text=" (Uses config if empty)").grid(row=3, column=2, padx=5, pady=2, sticky='w')

        ttk.Label(frame, text="Number of Frames:").grid(row=4, column=0, padx=5, pady=2, sticky='w')
        self.numframes_var = tk.StringVar(value="1") # Default from SplAcquirer example (used for reference frame)
        ttk.Entry(frame, textvariable=self.numframes_var, width=10).grid(row=4, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(frame, text=" (Uses config if empty)").grid(row=4, column=2, padx=5, pady=2, sticky='w')

        ttk.Label(frame, text="Mask File (optional):").grid(row=5, column=0, padx=5, pady=2, sticky='w')
        self.mask_file_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.mask_file_var, width=40).grid(row=5, column=1, padx=5, pady=2, sticky='ew')
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.mask_file_var)).grid(row=5, column=3, padx=5, pady=2)

        # Add flux calibration filename field
        ttk.Label(frame, text="Flux Calibration File:").grid(row=6, column=0, padx=5, pady=2, sticky='w')
        self.flux_calib_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.flux_calib_var, width=40).grid(row=6, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(frame, text=" (Uses config if empty, no calib if None)").grid(row=6, column=2, padx=5, pady=2, sticky='w')
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.flux_calib_var)).grid(row=6, column=3, padx=5, pady=2)

        self.display_reference_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="Display Reference Image", variable=self.display_reference_var,
            command=self.on_display_reference_toggle
        ).grid(row=7, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Actuator Positions (nm, optional):").grid(row=8, column=0, padx=5, pady=2, sticky='w')
        self.actuator_pos_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.actuator_pos_var, width=40).grid(row=8, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(frame, text=" (Comma-separated values)").grid(row=8, column=2, padx=5, pady=2, sticky='w')
        
        frame.columnconfigure(1, weight=1)

        # Create a frame for the buttons and progress bar
        control_frame = ttk.Frame(self.tab_acquirer)
        control_frame.pack(pady=10, fill='x')

        # Add Run and Stop buttons side by side
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=(0, 10))
        self.run_button = ttk.Button(button_frame, text="Run Acquisition", command=self.run_acquisition)
        self.run_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(button_frame, text="Stop Acquisition", command=self.stop_acquisition, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Add progress bar
        self.progress_frame = ttk.Frame(control_frame)
        self.progress_frame.pack(fill='x', padx=10)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', expand=True)
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack()

        ttk.Label(self.tab_acquirer, text="Tracking Number (tt):").pack(pady=(10,0))
        ttk.Entry(self.tab_acquirer, textvariable=self.tt_var, width=50, state='readonly').pack()

    def _browse_file(self, var_to_set):
        filename = filedialog.askopenfilename()
        if filename:
            # Convert absolute path to relative path if it's within the workspace
            try:
                workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
                if filename.startswith(workspace_path):
                    rel_path = os.path.relpath(filename, workspace_path)
                    var_to_set.set(rel_path)
                else:
                    var_to_set.set(filename)
            except Exception as e:
                self._logger.warning(f"Could not convert path to relative: {e}")
                var_to_set.set(filename)

    def _browse_folder(self, var_to_set):
        """Browse for a folder and convert its path to be relative to the workspace root if possible."""
        folder_path = filedialog.askdirectory()
        if folder_path:
            # Convert absolute path to relative path if it's within the workspace
            try:
                workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
                if folder_path.startswith(workspace_path):
                    rel_path = os.path.relpath(folder_path, workspace_path)
                    var_to_set.set(rel_path)
                else:
                    var_to_set.set(folder_path)
            except Exception as e:
                logger.warning(f"Could not convert path to relative: {e}")
                var_to_set.set(folder_path)

    def stop_acquisition(self):
        """Stop the ongoing acquisition process."""
        if self.acquisition_running and self.acquirer:
            self._log_message("Stopping acquisition...")
            self.acquirer._stop_acquisition = True  # Set the stop flag in the acquirer
            self.acquisition_running = False
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def update_progress(self, current, total, wavelength):
        """Update the progress bar and label."""
        progress = (current / total) * 100
        self.progress_var.set(progress)
        self.progress_label.config(text=f"Processing wavelength {wavelength} nm ({current}/{total})")
        self.master.update_idletasks()

    def run_acquisition(self):
        self._log_message("Starting acquisition...")
        try:
            # Check if hardware objects are initialized
            if self.filter_obj is None or self.camera_obj is None:
                messagebox.showerror("Hardware Error", "Filter or Camera objects not available.")
                self._log_message("Error: Filter or Camera objects not available in SplGuiApp.")
                return

            # Set acquisition state and update button states
            self.acquisition_running = True
            self.run_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Update the UI
            self.master.update()

            # Parameter Parsing
            lambda_start = self.lambda_start_var.get()
            lambda_stop = self.lambda_stop_var.get()
            lambda_step = self.lambda_step_var.get()
            
            lambda_vector = None
            if lambda_start and lambda_stop and lambda_step:
                try:
                    lambda_vector = np.arange(float(lambda_start), float(lambda_stop) + float(lambda_step)/2, float(lambda_step))
                    self._log_message(f"Using lambda vector: start={lambda_start}, stop={lambda_stop}, step={lambda_step}")
                except ValueError:
                    messagebox.showerror("Error", "Invalid lambda parameters. Must be numbers.")
                    return
            else:
                self._log_message("Using lambda vector from config.")

            exptime_str = self.exptime_var.get()
            exptime = float(exptime_str) if exptime_str else None
            
            numframes_str = self.numframes_var.get()
            numframes = int(numframes_str) if numframes_str else None

            mask_path = self.mask_file_var.get()
            mask = None
            if mask_path:
                try:
                    mask = np.load(mask_path)
                    self._log_message(f"Successfully loaded mask from: {mask_path}")
                except Exception as e:
                    messagebox.showerror("Mask Error", f"Could not load mask from '{mask_path}'. Error: {e}")
                    self._log_message(f"Error loading mask: {e}")
                    return
            
            display_reference = self.display_reference_var.get()
            
            flux_calib_path = self.flux_calib_var.get()
            
            # Parse actuator positions
            actuator_pos_str = self.actuator_pos_var.get()
            actuator_positions = None
            if actuator_pos_str:
                try:
                    actuator_positions = [float(pos.strip()) for pos in actuator_pos_str.split(',')]
                    self._log_message(f"Using actuator positions: {actuator_positions}")
                except ValueError:
                    messagebox.showerror("Error", "Invalid actuator positions. Must be comma-separated numbers.")
                    return

            # Local variables for use in the thread
            master = self.master
            acquirer = SplAcquirer(self.filter_obj, self.camera_obj)
            acquirer._stop_acquisition = False
            tt_var = self.tt_var
            log_message = self._log_message
            update_reference_image = self.update_reference_image if display_reference else None
            update_progress = self.update_progress

            # Define the acquisition function to run in a separate thread
            def acquisition_thread():
                try:
                    # Run acquisition with callback if display_reference is True
                    tt_result = acquirer.acquire(
                        lambda_vector=lambda_vector,
                        exptime=exptime,
                        numframes=numframes,
                        mask=mask,
                        display_reference=display_reference,
                        actuator_positions=actuator_positions,
                        flux_calibration_filename=flux_calib_path if flux_calib_path else None,
                        reference_image_callback=update_reference_image,
                        progress_callback=update_progress
                    )

                    # Update GUI after acquisition is complete (in main thread)
                    master.after(0, lambda: tt_var.set(tt_result))
                    master.after(0, lambda: log_message(f"Acquisition complete. Tracking Number (tt): {tt_result}"))
                    master.after(0, lambda: messagebox.showinfo("Acquisition Complete", f"Acquisition finished. Tracking Number (tt): {tt_result}"))

                except ValueError as ve:
                    logger.exception("ValueError during acquisition:")  # Log the full traceback
                    err_msg = str(ve)  # Capture the error message
                    master.after(0, lambda: messagebox.showerror("Input Error", f"Invalid input: {err_msg}"))
                    master.after(0, lambda: log_message(f"Input Error: {err_msg}"))
                except Exception as e:
                    logger.exception("Exception during acquisition:")  # Log the full traceback
                    err_msg = str(e)  # Capture the error message
                    master.after(0, lambda: messagebox.showerror("Error", f"An error occurred during acquisition: {err_msg}"))
                    master.after(0, lambda: log_message(f"Acquisition Error: {err_msg}"))
                finally:
                    # Reset acquisition state and button states (in main thread)
                    master.after(0, lambda: setattr(self, 'acquisition_running', False))
                    master.after(0, lambda: self.run_button.config(state=tk.NORMAL))
                    master.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
                    master.after(0, lambda: setattr(acquirer, '_stop_acquisition', False) if hasattr(acquirer, '_stop_acquisition') else None)
                    # Reset progress bar
                    master.after(0, lambda: self.progress_var.set(0))
                    master.after(0, lambda: self.progress_label.config(text=""))

            # Start the acquisition thread
            threading.Thread(target=acquisition_thread).start()

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during acquisition setup: {e}")
            self._log_message(f"Acquisition Setup Error: {e}")
            # Reset acquisition state and button states
            self.acquisition_running = False
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            if self.acquirer:
                self.acquirer._stop_acquisition = False

    def _create_processor_widgets(self):
        frame = ttk.LabelFrame(self.tab_processor, text="Processing Parameters")
        frame.pack(padx=10, pady=10, fill='x')

        # TT input with browse button
        tt_frame = ttk.Frame(frame)
        tt_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=2, sticky='ew')
        
        ttk.Label(tt_frame, text="Tracking Number (tt):").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(tt_frame, textvariable=self.tt_var, width=40).pack(side=tk.LEFT, expand=True, fill='x', padx=5)
        ttk.Button(tt_frame, text="Browse...", 
                  command=lambda: self._browse_folder(self.tt_var)).pack(side=tk.LEFT, padx=(5,0))

        self.debug_contours_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Debug Contours", 
                       variable=self.debug_contours_var,
                       command=self.on_debug_contours_toggle).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        
        frame.columnconfigure(1, weight=1)

        # Create a frame for the button and progress bar
        control_frame = ttk.Frame(self.tab_processor)
        control_frame.pack(pady=10, fill='x')

        # Add Run button
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=(0, 10))
        run_button = ttk.Button(button_frame, text="Run Processing", command=self.run_processing)
        run_button.pack()

        # Add progress bar
        self.processor_progress_frame = ttk.Frame(control_frame)
        self.processor_progress_frame.pack(fill='x', padx=10)
        self.processor_progress_var = tk.DoubleVar()
        self.processor_progress_bar = ttk.Progressbar(self.processor_progress_frame, variable=self.processor_progress_var, maximum=100)
        self.processor_progress_bar.pack(fill='x', expand=True)
        self.processor_progress_label = ttk.Label(self.processor_progress_frame, text="")
        self.processor_progress_label.pack()

        # Create frame for debug images
        self.debug_images_frame = ttk.LabelFrame(self.tab_processor, text="Debug Images (600nm)")
        self.debug_images_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Create matplotlib figure and canvas for debug images
        self.debug_figure = Figure(figsize=(10, 8), dpi=100)
        self.debug_canvas = FigureCanvasTkAgg(self.debug_figure, master=self.debug_images_frame)
        self.debug_canvas.get_tk_widget().pack(side=tk.TOP, fill='both', expand=True)
        
        # Add toolbar for navigation
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.debug_canvas, self.debug_images_frame)
        toolbar.update()

        # Initially hide the debug images frame
        self.debug_images_frame.pack_forget()

    def on_debug_contours_toggle(self):
        """Handle debug contours checkbox toggle."""
        if self.debug_contours_var.get():
            self.debug_images_frame.pack(padx=10, pady=10, fill='both', expand=True)
            # If we have a tt value, try to load the debug images
            tt_value = self.tt_var.get()
            if tt_value:
                self.load_debug_images(tt_value)
        else:
            self.debug_images_frame.pack_forget()

    def load_debug_images(self, tt_value):
        """Load and display debug images for 600nm wavelength."""
        try:
            # Get the storage folder
            storage_folder = os.path.join(config.MEASUREMENT_ROOT_FOLDER, tt_value)
            
            # Find all 600nm debug images
            debug_pattern = os.path.join(storage_folder, "image_600nm_pos*_proc_debug.png")
            debug_files = sorted(glob.glob(debug_pattern))
            
            if not debug_files:
                self._log_message("No debug images found for 600nm wavelength.")
                return

            # Calculate grid dimensions
            n_images = len(debug_files)
            n_cols = min(3, n_images)  # Max 3 columns
            n_rows = (n_images + n_cols - 1) // n_cols

            # Clear the figure
            self.debug_figure.clear()

            # Load and display each image
            for idx, file_path in enumerate(debug_files):
                ax = self.debug_figure.add_subplot(n_rows, n_cols, idx + 1)
                img = plt.imread(file_path)
                ax.imshow(img)
                ax.axis('off')
                # Extract position number from filename
                pos_match = re.search(r'pos(\d+)', os.path.basename(file_path))
                if pos_match:
                    pos_num = pos_match.group(1)
                    ax.set_title(f"Position {pos_num}")

            self.debug_figure.tight_layout()
            self.debug_canvas.draw()
            self._log_message(f"Loaded {len(debug_files)} debug images for 600nm wavelength.")

        except Exception as e:
            self._log_message(f"Error loading debug images: {e}")
            logger.error(f"Error loading debug images: {e}", exc_info=True)

    def run_processing(self):
        tt_value = self.tt_var.get()
        if not tt_value:
            messagebox.showerror("Error", "Tracking Number (tt) is required for processing.")
            return

        debug_contours = self.debug_contours_var.get()
        self._log_message(f"Starting processing for tt: {tt_value}, Debug Contours: {debug_contours}")

        try:
            # Reset progress bar
            self.processor_progress_var.set(0)
            self.processor_progress_label.config(text="Starting processing...")

            # --- INTEGRATE YOUR SplProcessor LOGIC HERE ---
            processor = SplProcessor()
            processor.process(tt=tt_value, debug_contours=debug_contours)
            
            # Update progress to 100% when complete
            self.processor_progress_var.set(100)
            self.processor_progress_label.config(text="Processing complete")
            
            self._log_message(f"Processing for tt: {tt_value} complete.")
            messagebox.showinfo("Processing Complete", f"Processing for tt: {tt_value} finished.")
            
            # If debug contours is enabled, load the debug images
            if debug_contours:
                self.load_debug_images(tt_value)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {e}")
            self._log_message(f"Processing Error: {e}")
        finally:
            # Reset progress bar after a delay
            self.master.after(2000, lambda: self.processor_progress_var.set(0))
            self.master.after(2000, lambda: self.processor_progress_label.config(text=""))

    def _create_analyzer_widgets(self):
        # Parameters frame
        frame = ttk.LabelFrame(self.tab_analyzer, text="Analysis Parameters")
        frame.pack(padx=10, pady=10, fill='x')

        # TT input with browse button
        tt_frame = ttk.Frame(frame)
        tt_frame.grid(row=0, column=0, columnspan=3, padx=5, pady=2, sticky='ew')
        
        ttk.Label(tt_frame, text="Tracking Number (tt):").pack(side=tk.LEFT, padx=(0,5))
        ttk.Entry(tt_frame, textvariable=self.tt_var, width=40).pack(side=tk.LEFT, expand=True, fill='x', padx=5)
        ttk.Button(tt_frame, text="Browse...", 
                  command=lambda: self._browse_folder(self.tt_var)).pack(side=tk.LEFT, padx=(5,0))

        self.use_processed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Use Processed FITS Files", 
                       variable=self.use_processed_var).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Matching Method:").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        self.matching_method_var = tk.StringVar(value="original")
        matching_options = ["original", "cross_correlation", "template_matching"]
        ttk.Combobox(frame, textvariable=self.matching_method_var, 
                    values=matching_options, state="readonly", width=20).grid(row=2, column=1, padx=5, pady=2, sticky='ew')
        
        frame.columnconfigure(1, weight=1)

        # Create a frame for the button and progress bar
        control_frame = ttk.Frame(self.tab_analyzer)
        control_frame.pack(pady=10, fill='x')

        # Add Run button
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=(0, 10))
        run_button = ttk.Button(button_frame, text="Run Analysis", command=self.run_analysis)
        run_button.pack()

        # Add progress bar
        self.analysis_progress_frame = ttk.Frame(control_frame)
        self.analysis_progress_frame.pack(fill='x', padx=10)
        self.analysis_progress_var = tk.DoubleVar()
        self.analysis_progress_bar = ttk.Progressbar(self.analysis_progress_frame, variable=self.analysis_progress_var, maximum=100)
        self.analysis_progress_bar.pack(fill='x', expand=True)
        self.analysis_progress_label = ttk.Label(self.analysis_progress_frame, text="")
        self.analysis_progress_label.pack()

        # Results frame
        results_frame = ttk.LabelFrame(self.tab_analyzer, text="Analysis Results")
        results_frame.pack(padx=10, pady=5, fill='both', expand=True)

        # Piston values display
        ttk.Label(results_frame, text="Piston Values:").pack(anchor='w', padx=5, pady=2)
        self.piston_text = tk.Text(results_frame, height=4, width=50)
        self.piston_text.pack(padx=5, pady=2, fill='x')

        # Create a frame for the fringe images
        self.fringes_frame = ttk.LabelFrame(results_frame, text="Fringe Images")
        self.fringes_frame.pack(padx=5, pady=5, fill='both', expand=True)
        
        # Create matplotlib figure and canvas for fringe images
        self.fringes_figure = Figure(figsize=(10, 8), dpi=100)
        self.fringes_canvas = FigureCanvasTkAgg(self.fringes_figure, master=self.fringes_frame)
        self.fringes_canvas.get_tk_widget().pack(side=tk.TOP, fill='both', expand=True)
        
        # Add toolbar for navigation
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(self.fringes_canvas, self.fringes_frame)
        toolbar.update()

    def update_analysis_progress(self, current, total, position):
        """Update the analysis progress bar and label."""
        progress = (current / total) * 100
        self.analysis_progress_var.set(progress)
        self.analysis_progress_label.config(text=f"Processing position {position} ({current}/{total})")
        self.master.update_idletasks()

    def run_analysis(self):
        tt_value = self.tt_var.get()
        if not tt_value:
            messagebox.showerror("Error", "Tracking Number (tt) is required for analysis.")
            return

        use_processed = self.use_processed_var.get()
        matching_method = self.matching_method_var.get()

        self._log_message(f"Starting analysis for tt: {tt_value}, Use Processed: {use_processed}, Matching Method: {matching_method}")

        try:
            # Clear previous results
            self.piston_text.delete('1.0', tk.END)
            self.fringes_figure.clear()

            # Get the storage folder
            storage_folder = os.path.join(config.MEASUREMENT_ROOT_FOLDER, tt_value)
            
            # Find all position numbers from the filenames
            position_pattern = re.compile(r'pos(\d+)')
            positions = set()
            for filename in os.listdir(storage_folder):
                match = position_pattern.search(filename)
                if match:
                    positions.add(int(match.group(1)))
            positions = sorted(list(positions))
            
            if not positions:
                self._log_message("No position data found in the storage folder.")
                return

            # Run analysis for each position
            results = {}
            total_positions = len(positions)
            
            for idx, position in enumerate(positions):
                # Update progress
                self.update_analysis_progress(idx + 1, total_positions, position)
                
                # Run analysis for this position
                analyzer = SplAnalyzer()
                position_result = analyzer.analyzer(
                    tt=tt_value,
                    use_processed=use_processed,
                    matching_method=matching_method,
                    position=position
                )
                results[position] = position_result

                # Display piston values in text widget
                self.piston_text.insert(tk.END, f"Position {position:02d}: {position_result}\n")

                # Try to load and display fringe image for this position
                try:
                    # Look for match images with the correct pattern
                    match_pattern = f"match_Qm_pos{position:02d}_vs_Template_Fringe_*_{matching_method}.png"
                    match_files = glob.glob(os.path.join(storage_folder, match_pattern))
                    
                    if match_files:
                        # Sort files to get the most recent one if multiple exist
                        match_files.sort(reverse=True)
                        image_path = match_files[0]
                        
                        # Calculate grid dimensions with max 3 rows per column
                        max_rows_per_column = 3
                        n_columns = (total_positions + max_rows_per_column - 1) // max_rows_per_column
                        current_column = idx // max_rows_per_column
                        current_row = idx % max_rows_per_column
                        
                        # Add subplot for this position
                        ax = self.fringes_figure.add_subplot(max_rows_per_column, n_columns, 
                                                           current_row * n_columns + current_column + 1)
                        img = plt.imread(image_path)
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(f"Position {position:02d}")
                        self._log_message(f"Loaded match image: {os.path.basename(image_path)}")
                    else:
                        self._log_message(f"No match image found for position {position:02d}")
                except Exception as img_error:
                    self._log_message(f"Error loading match image for position {position:02d}: {img_error}")

            # Adjust layout and draw
            self.fringes_figure.tight_layout()
            self.fringes_canvas.draw()

            self._log_message(f"Analysis for tt: {tt_value} complete.")
            messagebox.showinfo("Analysis Complete", f"Analysis for tt: {tt_value} finished.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")
            self._log_message(f"Analysis Error: {e}")
        finally:
            # Reset progress bar
            self.analysis_progress_var.set(0)
            self.analysis_progress_label.config(text="")

    def update_reference_image(self, image, positions, n_rows, m_cols):
        """Display the reference image in the GUI."""
        self.ref_figure.clear()
        ax = self.ref_figure.add_subplot(111)
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        # Plot spots and labels
        for idx, (x, y) in enumerate(positions):
            ax.plot(x, y, 'go', markersize=10)
            ax.text(x+50, y+50, str(idx), color='white', bbox=dict(facecolor='red', alpha=0.7))
        self.ref_canvas.draw()
        # Ensure the frame is visible - Removed redundant pack call
        # self.ref_image_frame.pack(padx=10, pady=10, fill='both', expand=True)

    def hide_reference_image(self):
        self.ref_image_frame.pack_forget()

    def on_display_reference_toggle(self):
        if self.display_reference_var.get():
            self.ref_image_frame.pack(padx=10, pady=10, fill='both', expand=True)
        else:
            self.hide_reference_image()


if __name__ == '__main__':
    # Configure basic logging for the application
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # CRITICAL: Initialize filter_obj and camera_obj here using the devices module
    filter_hardware = None
    camera_hardware = None
    initialized_successfully = False

    try:
        logger.info("Initializing hardware via devices.py...")
        filter_hardware, camera_hardware = devices.initialize_hardware()
        if filter_hardware and camera_hardware:
            logger.info("Hardware initialized successfully through devices.py.")
            initialized_successfully = True
        else:
            logger.error("Hardware initialization failed. Check devices.py and plico setup.")
            messagebox.showerror("Hardware Initialization Failed",
                                 "Could not initialize camera and/or filter. Check logs. GUI will run with limited functionality.")

    except Exception as e:
        logger.critical(f"Critical error during hardware initialization: {e}", exc_info=True)
        messagebox.showerror("Hardware Initialization Error",
                             f"An error occurred during hardware setup: {e}\n\nCheck logs. GUI will run with limited functionality.")

    root = tk.Tk()
    # Pass the initialized hardware objects to the GUI application
    app = SplGuiApp(root, filter_obj_instance=filter_hardware, camera_obj_instance=camera_hardware)

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            logger.info("Shutdown requested by user.")
            try:
                devices.shutdown_hardware() # Attempt to shutdown hardware
            except Exception as e:
                logger.error(f"Error during hardware shutdown: {e}", exc_info=True)
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing) # Handle window close button
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C).")
        on_closing() # Attempt graceful shutdown
    finally:
        logger.info("Application finished.")