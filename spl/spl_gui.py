import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import logging # Added for GUI logging consistency

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
        self.exptime_var = tk.StringVar(value="0.7") # Default from SplAcquirer example
        ttk.Entry(frame, textvariable=self.exptime_var, width=10).grid(row=3, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(frame, text=" (Uses config if empty)").grid(row=3, column=2, padx=5, pady=2, sticky='w')

        ttk.Label(frame, text="Number of Frames:").grid(row=4, column=0, padx=5, pady=2, sticky='w')
        self.numframes_var = tk.StringVar(value="1") # Default from SplAcquirer example (used for reference frame)
        ttk.Entry(frame, textvariable=self.numframes_var, width=10).grid(row=4, column=1, padx=5, pady=2, sticky='ew')
        ttk.Label(frame, text=" (Uses config if empty)").grid(row=4, column=2, padx=5, pady=2, sticky='w')

        ttk.Label(frame, text="Mask File (optional):").grid(row=5, column=0, padx=5, pady=2, sticky='w')
        self.mask_file_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.mask_file_var, width=40).grid(row=5, column=1, padx=5, pady=2, sticky='ew')
        ttk.Button(frame, text="Browse...", command=lambda: self._browse_file(self.mask_file_var)).grid(row=5, column=2, padx=5, pady=2)

        self.display_reference_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Display Reference Image", variable=self.display_reference_var).grid(row=6, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Actuator Position (nm, optional):").grid(row=7, column=0, padx=5, pady=2, sticky='w')
        self.actuator_pos_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.actuator_pos_var, width=10).grid(row=7, column=1, padx=5, pady=2, sticky='ew')
        
        frame.columnconfigure(1, weight=1)


        run_button = ttk.Button(self.tab_acquirer, text="Run Acquisition", command=self.run_acquisition)
        run_button.pack(pady=10)

        ttk.Label(self.tab_acquirer, text="Tracking Number (tt):").pack(pady=(10,0))
        ttk.Entry(self.tab_acquirer, textvariable=self.tt_var, width=50, state='readonly').pack()

    def _browse_file(self, var_to_set):
        filename = filedialog.askopenfilename()
        if filename:
            var_to_set.set(filename)

    def run_acquisition(self):
        self._log_message("Starting acquisition...")
        try:
            # Check if hardware objects are initialized (now passed to __init__)
            if self.filter_obj is None or self.camera_obj is None:
                messagebox.showerror("Hardware Error", "Filter or Camera objects not available. Initialization might have failed.")
                self._log_message("Error: Filter or Camera objects not available in SplGuiApp.")
                return

            # --- Parameter Parsing ---
            lambda_start = self.lambda_start_var.get()
            lambda_stop = self.lambda_stop_var.get()
            lambda_step = self.lambda_step_var.get()
            
            lambda_vector = None
            if lambda_start and lambda_stop and lambda_step:
                try:
                    lambda_vector = np.arange(float(lambda_start), float(lambda_stop) + float(lambda_step)/2, float(lambda_step)) # +step/2 to be inclusive of stop
                    self._log_message(f"Using lambda vector: start={lambda_start}, stop={lambda_stop}, step={lambda_step}")
                except ValueError:
                    messagebox.showerror("Error", "Invalid lambda parameters. Must be numbers.")
                    return
            else:
                self._log_message("Using lambda vector from config.")

            exptime_str = self.exptime_var.get()
            exptime = float(exptime_str) if exptime_str else None
            if exptime is not None: self._log_message(f"Using exptime: {exptime} ms")
            else: self._log_message("Using exptime from config.")

            numframes_str = self.numframes_var.get()
            numframes = int(numframes_str) if numframes_str else None
            if numframes is not None: self._log_message(f"Using numframes: {numframes}")
            else: self._log_message("Using numframes from config.")

            mask_path = self.mask_file_var.get()
            mask = None # Load mask if path is provided
            if mask_path:
                try:
                    mask = np.load(mask_path) # Assuming .npy file for mask
                    self._log_message(f"Successfully loaded mask from: {mask_path}")
                except Exception as e:
                    messagebox.showerror("Mask Error", f"Could not load mask from '{mask_path}'. Error: {e}")
                    self._log_message(f"Error loading mask: {e}")
                    return # Stop if mask loading fails
            
            display_reference = self.display_reference_var.get()
            self._log_message(f"Display reference: {display_reference}")

            actuator_pos_str = self.actuator_pos_var.get()
            actuator_position = float(actuator_pos_str) if actuator_pos_str else None
            if actuator_position is not None: self._log_message(f"Actuator position: {actuator_position} nm")

            # --- INTEGRATE YOUR SplAcquirer LOGIC HERE ---
            acquirer = SplAcquirer(self.filter_obj, self.camera_obj)
            tt_result = acquirer.acquire(
                lambda_vector=lambda_vector,
                exptime=exptime,
                numframes=numframes,
                mask=mask,
                display_reference=display_reference,
                actuator_position=actuator_position
            )
            self.tt_var.set(tt_result)
            self._log_message(f"Acquisition complete. Tracking Number (tt): {tt_result}")
            messagebox.showinfo("Acquisition Complete", f"Acquisition finished. Tracking Number (tt): {tt_result}")
            # --- End Integration ---
            
            # Dummy result for GUI testing (REMOVE AFTER INTEGRATION)
            # dummy_tt = f"dummy_tt_{np.random.randint(1000, 9999)}"
            # self.tt_var.set(dummy_tt)
            # self._log_message(f"Placeholder: Acquisition would run here. Generated tt: {dummy_tt}")
            # messagebox.showinfo("Acquisition", f"Acquisition process placeholder complete. tt = {dummy_tt}")

        except ValueError as ve:
            messagebox.showerror("Input Error", f"Invalid input: {ve}")
            self._log_message(f"Input Error: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during acquisition: {e}")
            self._log_message(f"Acquisition Error: {e}")

    def _create_processor_widgets(self):
        frame = ttk.LabelFrame(self.tab_processor, text="Processing Parameters")
        frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(frame, text="Tracking Number (tt):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Entry(frame, textvariable=self.tt_var, width=40).grid(row=0, column=1, padx=5, pady=2, sticky='ew')

        self.debug_contours_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Debug Contours", variable=self.debug_contours_var).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        
        frame.columnconfigure(1, weight=1)

        run_button = ttk.Button(self.tab_processor, text="Run Processing", command=self.run_processing)
        run_button.pack(pady=10)

    def run_processing(self):
        tt_value = self.tt_var.get()
        if not tt_value:
            messagebox.showerror("Error", "Tracking Number (tt) is required for processing.")
            return

        debug_contours = self.debug_contours_var.get()
        self._log_message(f"Starting processing for tt: {tt_value}, Debug Contours: {debug_contours}")

        try:
            # --- INTEGRATE YOUR SplProcessor LOGIC HERE ---
            processor = SplProcessor()
            processor.process(tt=tt_value, debug_contours=debug_contours)
            self._log_message(f"Processing for tt: {tt_value} complete.")
            messagebox.showinfo("Processing Complete", f"Processing for tt: {tt_value} finished.")
            # --- End Integration ---

            # Placeholder (REMOVE AFTER INTEGRATION)
            # self._log_message(f"Placeholder: Processing for tt: {tt_value} would run here.")
            # messagebox.showinfo("Processing", f"Processing for tt: {tt_value} placeholder complete.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {e}")
            self._log_message(f"Processing Error: {e}")

    def _create_analyzer_widgets(self):
        frame = ttk.LabelFrame(self.tab_analyzer, text="Analysis Parameters")
        frame.pack(padx=10, pady=10, fill='x')

        ttk.Label(frame, text="Tracking Number (tt):").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        ttk.Entry(frame, textvariable=self.tt_var, width=40).grid(row=0, column=1, padx=5, pady=2, sticky='ew')

        self.use_processed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Use Processed FITS Files", variable=self.use_processed_var).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')

        ttk.Label(frame, text="Matching Method:").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        self.matching_method_var = tk.StringVar(value="original")
        matching_options = ["original", "cross_correlation", "template_matching"]
        ttk.Combobox(frame, textvariable=self.matching_method_var, values=matching_options, state="readonly", width=20).grid(row=2, column=1, padx=5, pady=2, sticky='ew')
        
        frame.columnconfigure(1, weight=1)

        run_button = ttk.Button(self.tab_analyzer, text="Run Analysis", command=self.run_analysis)
        run_button.pack(pady=10)

    def run_analysis(self):
        tt_value = self.tt_var.get()
        if not tt_value:
            messagebox.showerror("Error", "Tracking Number (tt) is required for analysis.")
            return

        use_processed = self.use_processed_var.get()
        matching_method = self.matching_method_var.get()

        self._log_message(f"Starting analysis for tt: {tt_value}, Use Processed: {use_processed}, Matching Method: {matching_method}")

        try:
            # --- INTEGRATE YOUR SplAnalyzer LOGIC HERE ---
            analyzer = SplAnalyzer()
            results = analyzer.analyzer(
                tt=tt_value,
                use_processed=use_processed,
                matching_method=matching_method
            )
            self._log_message(f"Analysis for tt: {tt_value} complete.")
            self._log_message(f"Piston Results: {results}")
            # You might want to display results more structured way
            messagebox.showinfo("Analysis Complete", f"Analysis for tt: {tt_value} finished. Results: {results}")
            # --- End Integration ---

            # Placeholder (REMOVE AFTER INTEGRATION)
            # self._log_message(f"Placeholder: Analysis for tt: {tt_value} would run here.")
            # messagebox.showinfo("Analysis", f"Analysis for tt: {tt_value} placeholder complete.")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during analysis: {e}")
            self._log_message(f"Analysis Error: {e}")


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