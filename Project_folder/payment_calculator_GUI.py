# payment_calculator_GUI.py  (no pydantic)
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading, queue, os, yaml, datetime
from config import (WINDOW_WIDTH, WINDOW_HEIGHT, TITLE, CONSOLE_LINE_WIDTH, 
                    TYPEWRITER_DELAY_MS, TYPEWRITER_BLOCK_THRESHOLD, CONSOLE_QUEUE_MAXSIZE,
                    ROW0, ROW1_TO_6, ROW7, ROW8_PROGRESS, LOG_TIMESTAMP_FORMAT)
from modules.db import connect
from modules.validation_module import ValidationModule
from modules.identification_module import IdentificationModule
from modules.estimation_module import EstimationModule
from schemas import (validate_validation_rules_dict, validate_identification_rules_dict, validate_estimation_rules_dict)
from utils import rule_hash

class PaymentCalculatorGUI:
    def __init__(self, master):
        self.master = master
        master.title(TITLE)
        master.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        master.resizable(False, False)
        ctk.set_widget_scaling(1.0)
        ctk.set_appearance_mode("System")

        for d in ["inputs","validation_outputs","identification_outputs","estimation_outputs","log_files","cache"]:
            os.makedirs(d, exist_ok=True)

        self.db_path = os.path.join("cache","cache.db")
        self.conn, self.cursor, self.db_lock = connect(self.db_path)

        self.validation_rules = self.load_yaml_validated("inputs/validation_rules.yaml", validate_validation_rules_dict)
        self.identification_rules = self.load_yaml_validated("inputs/identification_rules.yaml", validate_identification_rules_dict)
        self.estimation_rules = self.load_yaml_validated("inputs/estimation_rules.yaml", validate_estimation_rules_dict)
        self.validation_rule_hash = rule_hash(self.validation_rules)
        self.identification_rule_hash = rule_hash(self.identification_rules)
        self.estimation_rule_hash = rule_hash(self.estimation_rules)

        self.setup_main_frame()
        self.setup_left_panel()
        self.setup_middle_panel()
        self.setup_right_panel()
        self.setup_progress_row()

        self.thread = None
        self.stop_event = threading.Event()
        self.log_file = None
        self.log_lock = threading.Lock()

        self.console_queue = queue.Queue(maxsize=CONSOLE_QUEUE_MAXSIZE)
        self.console_typing = False

        self.console.configure(state="normal")
        welcome_text = (
            f"{TITLE}\n{'='*CONSOLE_LINE_WIDTH}\n"
            f"Version: 1.1\nDate: {datetime.date.today():%Y-%m-%d}\n"
            f"Time: {datetime.datetime.now():%H:%M:%S}\n"
            f"{'='*CONSOLE_LINE_WIDTH}\n"
            "Please enter a .csv file, select your modules, and press \"Run the Calculator\".\n"
        )
        self.console.insert("end", welcome_text)
        self.console.configure(state="disabled")

    def load_yaml_validated(self, path: str, validator):
        if not os.path.exists(path):
            messagebox.showerror("Missing Rule File", f"Required rules file not found: {path}")
            raise SystemExit(1)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = yaml.safe_load(f)
            validator(obj)
            return obj
        except Exception as e:
            messagebox.showerror("Invalid Rule File", f"{path} failed schema validation.\n{e}")
            raise

    def setup_main_frame(self):
        self.master_frame = ctk.CTkFrame(self.master)
        self.master_frame.pack(fill="both", expand=True)
        self.master_frame.columnconfigure(0, minsize=200)
        self.master_frame.columnconfigure(1, minsize=400)
        self.master_frame.columnconfigure(2, minsize=400)
        self.master_frame.rowconfigure(0, minsize=ROW0)
        for r in range(1,7):
            self.master_frame.rowconfigure(r, minsize=ROW1_TO_6)
        self.master_frame.rowconfigure(7, minsize=ROW7)
        self.master_frame.rowconfigure(8, minsize=ROW8_PROGRESS)

    def setup_left_panel(self):
        title_label = ctk.CTkLabel(self.master_frame, text=TITLE, font=ctk.CTkFont(size=24), anchor="w")
        title_label.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=(20,5), pady=5)

        modules_label = ctk.CTkLabel(self.master_frame, text="Modules", font=ctk.CTkFont(size=18))
        modules_label.grid(row=1, column=0, sticky="nsew", padx=(20,5), pady=5)

        self.module_frame = ctk.CTkFrame(self.master_frame)
        self.module_frame.grid(row=2, column=0, rowspan=3, sticky="nsew", padx=(20,5), pady=5)

        self.switch_validation = ctk.CTkSwitch(self.module_frame, text="Validation"); self.switch_validation.select()
        self.switch_identification = ctk.CTkSwitch(self.module_frame, text="Identification"); self.switch_identification.select()
        self.switch_estimation = ctk.CTkSwitch(self.module_frame, text="Estimation"); self.switch_estimation.select()
        self.switch_validation.pack(anchor="w", pady=(10,5), padx=10)
        self.switch_identification.pack(anchor="w", pady=5, padx=10)
        self.switch_estimation.pack(anchor="w", pady=5, padx=10)

        self.run_button = ctk.CTkButton(self.master_frame, text="Run the calculator", fg_color="green", command=self.on_run)
        self.run_button.grid(row=5, column=0, sticky="nsew", padx=(20,5), pady=5)

        self.exit_button = ctk.CTkButton(self.master_frame, text="Exit & Close", state="disabled", command=self.on_exit)
        self.exit_button.grid(row=6, column=0, sticky="nsew", padx=(20,5), pady=5)

        self.file_path_entry = ctk.CTkEntry(self.master_frame, placeholder_text="Your CSV file here")
        self.file_path_entry.grid(row=7, column=0, columnspan=2, sticky="nsew", padx=(20,5), pady=5)

    def setup_middle_panel(self):
        intro_frame = ctk.CTkScrollableFrame(self.master_frame)
        intro_frame.grid(row=1, column=1, rowspan=3, sticky="nsew", padx=5, pady=5)
        intro_text = ("Welcome to the Payment Calculator GUI.\n\n"
                      "Validate, identify eligibility, and estimate payments.\n"
                      "Use Browse to select a CSV file, choose modules, and click 'Run the calculator'.")
        ctk.CTkLabel(intro_frame, text=intro_text, justify="left", wraplength=380).pack(padx=5, pady=5)

        tabview = ctk.CTkTabview(self.master_frame, width=380, height=230)
        tabview.grid(row=4, column=1, rowspan=3, sticky="nsew", padx=5, pady=5)
        tabview.add("Validation"); tabview.add("Identification"); tabview.add("Estimation")
        ctk.CTkLabel(tabview.tab("Validation"), text="Checks types, missing values, ranges, outliers.", justify="left", wraplength=360).pack(padx=5, pady=5)
        ctk.CTkLabel(tabview.tab("Identification"), text="Determines eligibility based on rules.", justify="left", wraplength=360).pack(padx=5, pady=5)
        ctk.CTkLabel(tabview.tab("Estimation"), text="Calculates payment using a regression-style formula.", justify="left", wraplength=360).pack(padx=5, pady=5)

    def setup_right_panel(self):
        self.console = ctk.CTkTextbox(self.master_frame, width=380, state="disabled")
        self.console.grid(row=1, column=2, rowspan=6, sticky="nsew", padx=(5,20), pady=5)

        self.browse_button = ctk.CTkButton(self.master_frame, text="Browse...", command=self.on_browse)
        self.browse_button.grid(row=7, column=2, sticky="nsew", padx=(5,20), pady=5)

    def setup_progress_row(self):
        self.progress_bar = ctk.CTkProgressBar(self.master_frame)
        self.progress_bar.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=10, pady=(0,5))
        self.progress_bar.set(0.0)

    def _clear_console(self):
        self.console_typing = False
        try:
            with self.console_queue.mutex:
                self.console_queue.queue.clear()
        except Exception:
            pass
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")

    def log_and_console(self, text: str, end: str = "\n"):
        if self.log_file:
            with self.log_lock:
                self.log_file.write(text + end)
                self.log_file.flush()
        msg = text + end
        try:
            self.console_queue.put_nowait(msg)
        except queue.Full:
            try:
                _ = self.console_queue.get_nowait()
            except Exception:
                pass
            self.console_queue.put_nowait(msg)
        self.master.after(0, self._process_console_queue)

    def _process_console_queue(self):
        if self.console_typing:
            return
        if self.console_queue.empty():
            return
        msg = self.console_queue.get_nowait()
        if len(msg) >= TYPEWRITER_BLOCK_THRESHOLD:
            self.console.configure(state="normal")
            self.console.insert("end", msg)
            self.console.yview_moveto(1.0)
            self.console.configure(state="disabled")
            self.master.after(0, self._process_console_queue)
        else:
            self.console_typing = True
            self._typewriter_step(msg, 0)

    def _typewriter_step(self, msg: str, i: int):
        if i == 0:
            self.console.configure(state="normal")
        if i < len(msg):
            self.console.insert("end", msg[i])
            self.console.yview_moveto(1.0)
            self.console.after(TYPEWRITER_DELAY_MS, self._typewriter_step, msg, i+1)
        else:
            self.console.configure(state="disabled")
            self.console_typing = False
            self.master.after(0, self._process_console_queue)

    def set_progress(self, cur: int, tot: int, message: str | None = None):
        def _update():
            val = 0.0
            if tot and tot > 0:
                val = min(max(cur / tot, 0.0), 1.0)
            self.progress_bar.set(val)
            if message:
                self.log_and_console(message)
        try:
            self.master.after(0, _update)
        except Exception:
            pass

    def show_modal_summary(self, title: str, message: str) -> threading.Event:
        done = threading.Event()
        def _show():
            top = ctk.CTkToplevel(self.master)
            top.title(title)
            top.geometry("500x300")
            lbl = ctk.CTkLabel(top, text=message, justify="left", wraplength=460)
            lbl.pack(padx=20, pady=20, fill="both", expand=True)
            def _ok():
                try: top.destroy()
                except: pass
                done.set()
            btn = ctk.CTkButton(top, text="OK", command=_ok)
            btn.pack(pady=(0,20))
            top.grab_set()
        self.master.after(0, _show)
        return done

    def on_browse(self):
        path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV Files","*.csv")])
        if path:
            self.file_path_entry.delete(0,"end")
            self.file_path_entry.insert(0, path)

    def on_run(self):
        file_path = self.file_path_entry.get().strip()
        modules_selected = [
            self.switch_validation.get(),
            self.switch_identification.get(),
            self.switch_estimation.get(),
        ]
        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Input Error", "Please select a valid CSV file.")
            return
        if not any(modules_selected):
            messagebox.showerror("Input Error", "Please select at least one module to run.")
            return

        self.run_button.configure(state="disabled")
        self.exit_button.configure(state="normal")
        self.file_path_entry.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.progress_bar.set(0.0)

        ts = datetime.datetime.now().strftime(LOG_TIMESTAMP_FORMAT)
        log_path = os.path.join("log_files", f"log_{ts}.txt")
        self.log_file = open(log_path, "w", encoding="utf-8")

        self._clear_console()

        start_time = datetime.datetime.now()
        intro = (f"Running Payment Calculator\n{'='*CONSOLE_LINE_WIDTH}\n"
                 f"Data file: {os.path.basename(file_path)}\nModules selected: "
                 f"{', '.join([m for m, sel in zip(['Validation','Identification','Estimation'], modules_selected) if sel])}\n"
                 f"Start time: {start_time:%Y-%m-%d %H:%M:%S}\n{'='*CONSOLE_LINE_WIDTH}\n")
        self.log_and_console(intro)

        self.thread = threading.Thread(target=self.run_modules, args=(file_path, modules_selected), daemon=True)
        self.thread.start()
        self.master.after(100, self.check_thread)

    def on_exit(self):
        if self.thread and self.thread.is_alive():
            if not messagebox.askyesno("Cancel run", "A run is in progress. Do you want to cancel it?"):
                return
            self.stop_event.set()
            self.log_and_console("Cancellation requested by user...")
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass
        if self.log_file:
            try: self.log_file.close()
            except Exception: pass
        self.master.destroy()

    def check_thread(self):
        if self.thread and self.thread.is_alive():
            self.master.after(100, self.check_thread)
        else:
            self.run_button.configure(state="normal")
            self.exit_button.configure(state="disabled")
            self.file_path_entry.configure(state="normal")
            self.browse_button.configure(state="normal")
            end_time = datetime.datetime.now()
            self.log_and_console(f"{'-'*CONSOLE_LINE_WIDTH}\nEnd time: {end_time:%Y-%m-%d %H:%M:%S}\nRun completed.\n")
            if self.log_file:
                try: self.log_file.close()
                except Exception: pass
                self.log_file = None
            self.stop_event.clear()

    def run_modules(self, file_path, modules_selected):
        try:
            def progress(cur, tot, msg=None):
                self.set_progress(cur, tot, msg)
            def event(level, msg):
                self.log_and_console(f"[{level}] {msg}")

            if modules_selected[0]:
                self.log_and_console("Starting Validation module...")
                val = ValidationModule(self.conn, self.cursor, self.validation_rules, progress, event)
                res = val.run(file_path, self.stop_event)
                if res.get("status") == "failed":
                    self.log_and_console("Validation aborted: " + res.get("message","Unknown error") + "\n")
                    return
                warnings_ct = res.get("warnings_count",0); errors_ct = res.get("errors_count",0)
                if res.get("dataset_valid", False):
                    self.log_and_console(f"Validation module completed — dataset validated. Warnings: {warnings_ct}, Errors: {errors_ct}.")
                else:
                    self.log_and_console(f"Validation module completed — failed to validate data. Warnings: {warnings_ct}, Errors: {errors_ct}.")
                    evt = self.show_modal_summary("Validation completed", f"Failed to validate data.\nWarnings: {warnings_ct}\nErrors: {errors_ct}")
                    evt.wait()
                    return
                evt = self.show_modal_summary("Validation completed", f"Dataset validated.\nWarnings: {warnings_ct}\nErrors: {errors_ct}")
                evt.wait()

            if modules_selected[1]:
                self.log_and_console("\nStarting Identification module...")
                ident = IdentificationModule(self.conn, self.cursor, self.identification_rules, progress, event, self.validation_rules)
                res = ident.run(file_path, self.stop_event)
                if res.get("status") == "failed":
                    self.log_and_console("Identification aborted: " + res.get("message","Unknown issue") + "\n")
                    return
                msg = (f"In-scope: {res.get('in_scope_count',0)} ({res.get('in_scope_pct',0)}%), "
                       f"Out-of-scope: {res.get('out_scope_count',0)} ({res.get('out_scope_pct',0)}%).")
                self.log_and_console("Identification completed. " + msg)
                evt = self.show_modal_summary("Identification completed", msg)
                evt.wait()

            if modules_selected[2]:
                self.log_and_console("\nStarting Estimation module...")
                est = EstimationModule(self.conn, self.cursor, self.estimation_rules, progress, event, self.validation_rules)
                res = est.run(file_path, self.stop_event)
                if res.get("status") == "failed":
                    self.log_and_console("Estimation aborted: " + res.get("message","Unknown issue") + "\n")
                    return
                total = res.get("total_payment",0.0); elig = res.get("eligible_payment",0.0); inelig = res.get("ineligible_payment",0.0)
                msg = f"Total payout: £{total:.2f}, Eligible payout: £{elig:.2f}, Ineligible payout: £{inelig:.2f}."
                self.log_and_console("Estimation completed. " + msg)
                evt = self.show_modal_summary("Estimation completed", msg)
                evt.wait()

        except Exception as e:
            self.log_and_console(f"\nError: {e}\n")
        finally:
            pass

if __name__ == "__main__":
    app = ctk.CTk()
    gui = PaymentCalculatorGUI(app)
    app.mainloop()
