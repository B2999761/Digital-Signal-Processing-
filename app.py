import tkinter as pd
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim
import os


class WatermarkRemover:
    @staticmethod
    def process_image(img_bgr):
        """
        CURRENT ALGORITHM: Dynamic Blind Masking (Classical)
        * REPLACE this function content with your Deep Learning Model later *
        """
        original_h, original_w = img_bgr.shape[:2]
        proc_size = (512, 512)
        
        img_resized = cv2.resize(img_bgr, proc_size)
        
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        mask = cv2.adaptiveThreshold(enhanced, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 5)
        
        kernel_small = np.ones((2,2), np.uint8)
        mask = cv2.erode(mask, kernel_small, iterations=1)
        kernel_large = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel_large, iterations=2)
        
        result = cv2.inpaint(img_resized, mask, 3, cv2.INPAINT_TELEA)
        
        result_final = cv2.resize(result, (original_w, original_h))
        return result_final

    @staticmethod
    def calculate_metrics(img_gt, img_pred):
        # Ensure sizes match
        if img_gt.shape != img_pred.shape:
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))
            
        # PSNR
        psnr_val = cv2.PSNR(img_gt, img_pred)
        
        # SSIM (Needs Grayscale)
        gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray_gt, gray_pred)
        
        return psnr_val, ssim_val

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("EE6001 Project: Blind Watermark Removal Demo")
        self.root.geometry("1200x650")
        
        self.path_wm = None
        self.path_gt = None
        self.img_wm_cv = None
        self.img_gt_cv = None
        self.img_out_cv = None
        
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 10), padding=6)
        style.configure("TLabel", font=("Helvetica", 11))
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))

        control_frame = ttk.Frame(root, padding="10")
        control_frame.pack(side="top", fill="x")
        
        ttk.Label(control_frame, text="Control Panel:", style="Header.TLabel").pack(side="left", padx=10)
        
        self.btn_load_wm = ttk.Button(control_frame, text="1. Load Watermarked Image", command=self.load_wm_image)
        self.btn_load_wm.pack(side="left", padx=5)
        
        self.btn_load_gt = ttk.Button(control_frame, text="2. Load Ground Truth", command=self.load_gt_image)
        self.btn_load_gt.pack(side="left", padx=5)
        
        self.btn_run = ttk.Button(control_frame, text="▶ Run Processing", command=self.run_processing, state="disabled")
        self.btn_run.pack(side="left", padx=20)

        self.lbl_status = ttk.Label(root, text="Status: Waiting for input...", foreground="blue")
        self.lbl_status.pack(side="top", pady=5)

        display_frame = ttk.Frame(root)
        display_frame.pack(expand=True, fill="both", padx=20, pady=10)
        
        self.panel_wm = self.create_image_panel(display_frame, "Input (Watermarked)")
        self.panel_wm.pack(side="left", expand=True, fill="both", padx=5)
        
        self.panel_out = self.create_image_panel(display_frame, "Model Output")
        self.panel_out.pack(side="left", expand=True, fill="both", padx=5)
        
        self.panel_gt = self.create_image_panel(display_frame, "Ground Truth")
        self.panel_gt.pack(side="left", expand=True, fill="both", padx=5)
        
        metrics_frame = ttk.LabelFrame(root, text="Performance Metrics", padding="15")
        metrics_frame.pack(side="bottom", fill="x", padx=20, pady=20)
        
        self.lbl_psnr = ttk.Label(metrics_frame, text="PSNR: -- dB", font=("Consolas", 14, "bold"))
        self.lbl_psnr.pack(side="left", padx=50)
        
        self.lbl_ssim = ttk.Label(metrics_frame, text="SSIM: --", font=("Consolas", 14, "bold"))
        self.lbl_ssim.pack(side="left", padx=50)

    def create_image_panel(self, parent, title):
        frame = ttk.Frame(parent, borderwidth=2, relief="groove")
        ttk.Label(frame, text=title, font=("Helvetica", 12, "bold")).pack(pady=5)
        lbl_img = ttk.Label(frame, text="No Image")
        lbl_img.pack(expand=True)
        frame.lbl_ref = lbl_img 
        return frame

    def load_wm_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.path_wm = path
            self.img_wm_cv = cv2.imread(path)
            self.show_image(self.img_wm_cv, self.panel_wm.lbl_ref)
            self.lbl_status.config(text="Watermarked image loaded.")
            self.check_ready()

    def load_gt_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
        if path:
            self.path_gt = path
            self.img_gt_cv = cv2.imread(path)
            self.show_image(self.img_gt_cv, self.panel_gt.lbl_ref)
            self.lbl_status.config(text="Ground Truth loaded.")
            self.check_ready()

    def check_ready(self):
        if self.path_wm and self.path_gt:
            self.btn_run.config(state="normal")
            self.lbl_status.config(text="Ready to process.", foreground="green")

    def run_processing(self):
        if self.img_wm_cv is None: return
        
        self.lbl_status.config(text="Processing... Please wait...", foreground="orange")
        self.root.update() # Force UI update
        
        self.img_out_cv = WatermarkRemover.process_image(self.img_wm_cv)
        
        self.show_image(self.img_out_cv, self.panel_out.lbl_ref)
        
        psnr, ssim_val = WatermarkRemover.calculate_metrics(self.img_gt_cv, self.img_out_cv)
        
        self.lbl_psnr.config(text=f"PSNR: {psnr:.2f} dB")
        self.lbl_ssim.config(text=f"SSIM: {ssim_val:.4f}")
        
        self.lbl_status.config(text="Processing Complete.", foreground="black")

    def show_image(self, cv_img, label_widget):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb_img)
        
        display_h = 400
        aspect_ratio = im_pil.width / im_pil.height
        display_w = int(display_h * aspect_ratio)
        im_resized = im_pil.resize((display_w, display_h))
        
        imgtk = ImageTk.PhotoImage(image=im_resized)
        label_widget.config(image=imgtk, text="")
        label_widget.image = imgtk # Keep reference to avoid garbage collection

if __name__ == "__main__":
    root = pd.Tk()
    app = App(root)
    root.mainloop()