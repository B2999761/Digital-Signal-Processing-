import tkinter as tk  # Fixed import alias from 'pd' to 'tk'
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.metrics import structural_similarity as ssim
import os
import threading  # NEW: Required to prevent GUI freezing

# ===== PyTorch imports =====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet34, ResNet34_Weights

# ==========================================
# 1. MODEL ARCHITECTURE (Must match training)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        # Use valid weights for newer torchvision versions
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = nn.MaxPool2d(2, 2)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64 + 64, 64)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

        self.register_buffer("enc_mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("enc_std", torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        x_enc = (x - self.enc_mean) / self.enc_std
        x0_in = self.input_conv(x)
        x0 = self.layer0(x_enc)
        p0 = self.pool(x0)
        x1 = self.layer1(p0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.up4(x4)
        x3_rs = F.interpolate(x3, size=d4.shape[2:], mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, x3_rs], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x2_rs = F.interpolate(x2, size=d3.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, x2_rs], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x1_rs = F.interpolate(x1, size=d2.shape[2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, x1_rs], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x0_up = F.interpolate(x0_in, size=d1.shape[2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, x0_up], dim=1)
        d1 = self.dec1(d1)

        d1_up = F.interpolate(d1, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = self.final_conv(d1_up)
        out = torch.sigmoid(out)
        return out

# ==========================================
# 2. LOGIC CONTROLLER
# ==========================================
class WatermarkRemover:
    METHODS = ["Classical (CLAHE + Inpaint)", "ResUNet (Deep Model)"]
    current_method = METHODS[0]
    
    # Check that this filename matches exactly what you have in your folder
    MODEL_WEIGHTS_PATH = "best_resnet_unet_watermark.pth" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_tensor = T.ToTensor()
    model = None

    @staticmethod
    def set_method(method_name: str):
        if method_name in WatermarkRemover.METHODS:
            WatermarkRemover.current_method = method_name

    @staticmethod
    def load_resunet_model():
        if WatermarkRemover.model is not None:
            return True

        if not os.path.exists(WatermarkRemover.MODEL_WEIGHTS_PATH):
            return "FileNotFound" # Return string code for error

        try:
            # Initialize model
            model = ResNetUNet(n_classes=3).to(WatermarkRemover.device)
            
            # Load weights
            state_dict = torch.load(
                WatermarkRemover.MODEL_WEIGHTS_PATH, 
                map_location=WatermarkRemover.device
            )
            model.load_state_dict(state_dict)
            model.eval()
            WatermarkRemover.model = model
            print(f"Model loaded on {WatermarkRemover.device}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return str(e)

    @staticmethod
    def process_image(img_bgr):
        if WatermarkRemover.current_method == "Classical (CLAHE + Inpaint)":
            return WatermarkRemover.process_classical(img_bgr)
        elif WatermarkRemover.current_method == "ResUNet (Deep Model)":
            # Attempt load if not loaded
            status = WatermarkRemover.load_resunet_model()
            if status is not True:
                return None # Failed to load
            return WatermarkRemover.process_resunet(img_bgr)
        return WatermarkRemover.process_classical(img_bgr)

    @staticmethod
    def process_classical(img_bgr):
        original_h, original_w = img_bgr.shape[:2]
        proc_size = (512, 512)
        img_resized = cv2.resize(img_bgr, proc_size)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        mask = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 15, 5)
        kernel_small = np.ones((2,2), np.uint8)
        mask = cv2.erode(mask, kernel_small, iterations=1)
        kernel_large = np.ones((3,3), np.uint8)
        mask = cv2.dilate(mask, kernel_large, iterations=2)
        result = cv2.inpaint(img_resized, mask, 3, cv2.INPAINT_TELEA)
        result_final = cv2.resize(result, (original_w, original_h))
        return result_final

    @staticmethod
    def process_resunet(img_bgr):
        if WatermarkRemover.model is None: return None
        original_h, original_w = img_bgr.shape[:2]
        
        # Preprocessing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        TARGET_H, TARGET_W = 352, 528
        pil_resized = pil_img.resize((TARGET_W, TARGET_H), Image.BICUBIC)
        
        x = WatermarkRemover.to_tensor(pil_resized).unsqueeze(0).to(WatermarkRemover.device)

        # Inference
        with torch.no_grad():
            y_pred = WatermarkRemover.model(x)

        # Postprocessing
        y_pred = y_pred.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        y_pred = (y_pred * 255).astype(np.uint8)
        
        # Resize to original
        y_pred_resized = cv2.resize(y_pred, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
        out_bgr = cv2.cvtColor(y_pred_resized, cv2.COLOR_RGB2BGR)
        return out_bgr

    @staticmethod
    def calculate_metrics(img_gt, img_pred):
        if img_gt.shape != img_pred.shape:
            img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))
        psnr_val = cv2.PSNR(img_gt, img_pred)
        gray_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)
        gray_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
        ssim_val = ssim(gray_gt, gray_pred)
        return psnr_val, ssim_val

# ==========================================
# 3. GUI APPLICATION
# ==========================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("EE6001 Project: Blind Watermark Removal Demo")
        self.root.geometry("1200x700")
        
        self.path_wm = None
        self.path_gt = None
        self.img_wm_cv = None
        self.img_gt_cv = None
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.configure("Header.TLabel", font=("Helvetica", 14, "bold"))

        # --- Control Frame ---
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side="top", fill="x")
        
        ttk.Label(control_frame, text="Control Panel:", style="Header.TLabel").pack(side="left", padx=10)
        
        self.btn_load_wm = ttk.Button(control_frame, text="1. Load Input", command=self.load_wm_image)
        self.btn_load_wm.pack(side="left", padx=5)
        
        self.btn_load_gt = ttk.Button(control_frame, text="2. Load Truth", command=self.load_gt_image)
        self.btn_load_gt.pack(side="left", padx=5)
        
        # Method Selection
        ttk.Label(control_frame, text="Method:", padding=(15,0)).pack(side="left")
        self.method_var = tk.StringVar(value=WatermarkRemover.METHODS[0])
        self.method_combo = ttk.Combobox(control_frame, textvariable=self.method_var, 
                                         values=WatermarkRemover.METHODS, state="readonly", width=25)
        self.method_combo.pack(side="left", padx=5)
        self.method_combo.bind("<<ComboboxSelected>>", self.on_method_change)

        self.btn_run = ttk.Button(control_frame, text="▶ Run Processing", command=self.start_processing_thread, state="disabled")
        self.btn_run.pack(side="left", padx=20)

        # --- Status Bar ---
        self.lbl_status = ttk.Label(self.root, text="Status: Waiting for input...", foreground="blue")
        self.lbl_status.pack(side="top", pady=5)
        
        # --- Progress Bar (Indeterminate) ---
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="indeterminate")
        # Pack it but hide it initially; we will pack it when running

        # --- Display Frame ---
        display_frame = ttk.Frame(self.root)
        display_frame.pack(expand=True, fill="both", padx=20, pady=10)
        
        self.panel_wm = self.create_image_panel(display_frame, "Input (Watermarked)")
        self.panel_wm.pack(side="left", expand=True, fill="both", padx=5)
        
        self.panel_out = self.create_image_panel(display_frame, "Model Output")
        self.panel_out.pack(side="left", expand=True, fill="both", padx=5)
        
        self.panel_gt = self.create_image_panel(display_frame, "Ground Truth")
        self.panel_gt.pack(side="left", expand=True, fill="both", padx=5)
        
        # --- Metrics ---
        metrics_frame = ttk.LabelFrame(self.root, text="Performance Metrics", padding="15")
        metrics_frame.pack(side="bottom", fill="x", padx=20, pady=20)
        
        self.lbl_psnr = ttk.Label(metrics_frame, text="PSNR: -- dB", font=("Consolas", 14, "bold"))
        self.lbl_psnr.pack(side="left", padx=50)
        
        self.lbl_ssim = ttk.Label(metrics_frame, text="SSIM: --", font=("Consolas", 14, "bold"))
        self.lbl_ssim.pack(side="left", padx=50)

    def on_method_change(self, event=None):
        method_name = self.method_var.get()
        WatermarkRemover.set_method(method_name)
        self.lbl_status.config(text=f"Method selected: {method_name}", foreground="blue")

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
        # We allow running without Ground Truth (metrics will just be skipped)
        if self.path_wm:
            self.btn_run.config(state="normal")
            self.lbl_status.config(text="Ready to process.", foreground="green")

    # ===== THREADING LOGIC START =====
    def start_processing_thread(self):
        """Starts the processing in a background thread to keep GUI responsive."""
        if self.img_wm_cv is None: return
        
        # Disable button and show progress
        self.btn_run.config(state="disabled")
        self.progress.pack(pady=5)
        self.progress.start(10) # Move progress bar
        self.lbl_status.config(text="Processing... (Loading Model & Inferring)", foreground="orange")
        
        # Run the heavy task in a separate thread
        threading.Thread(target=self.run_processing, daemon=True).start()

    def run_processing(self):
        """Runs inside the background thread."""
        try:
            # 1. Run inference
            img_out_cv = WatermarkRemover.process_image(self.img_wm_cv)
            
            # 2. Schedule UI update back on the main thread
            self.root.after(0, lambda: self.finish_processing(img_out_cv))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed:\n{e}"))
            self.root.after(0, lambda: self.reset_ui_state())

    def finish_processing(self, img_out_cv):
        """Called by main thread when processing is done."""
        self.progress.stop()
        self.progress.pack_forget() # Hide progress bar
        self.btn_run.config(state="normal")

        if img_out_cv is None:
            messagebox.showerror("Error", "Model returned None. Check if .pth file exists.")
            self.lbl_status.config(text="Error: Check console/popup.", foreground="red")
            return

        self.show_image(img_out_cv, self.panel_out.lbl_ref)
        
        # Calculate Metrics if GT is available
        if self.img_gt_cv is not None:
            psnr, ssim_val = WatermarkRemover.calculate_metrics(self.img_gt_cv, img_out_cv)
            self.lbl_psnr.config(text=f"PSNR: {psnr:.2f} dB")
            self.lbl_ssim.config(text=f"SSIM: {ssim_val:.4f}")
        else:
            self.lbl_psnr.config(text="PSNR: --")
            self.lbl_ssim.config(text="SSIM: --")
            
        self.lbl_status.config(text="Processing Complete.", foreground="black")

    def reset_ui_state(self):
        self.progress.stop()
        self.progress.pack_forget()
        self.btn_run.config(state="normal")
        self.lbl_status.config(text="Error occurred.", foreground="red")
    # ===== THREADING LOGIC END =====

    def show_image(self, cv_img, label_widget):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb_img)
        
        display_h = 350
        aspect_ratio = im_pil.width / im_pil.height
        display_w = int(display_h * aspect_ratio)
        im_resized = im_pil.resize((display_w, display_h))
        
        imgtk = ImageTk.PhotoImage(image=im_resized)
        label_widget.config(image=imgtk, text="")
        label_widget.image = imgtk 

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()