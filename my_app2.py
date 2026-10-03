import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np

ctk.set_appearance_mode("dark")

window = ctk.CTk()
window.title("Image DSP Studio")
window.geometry("1500x780")
window.configure(fg_color="#141420")

color_img = None
gray_array = None
result_array = None

BG = "#141420"
CARD = "#1e1e30"
ACCENT = "#a78bfa"
ACCENT_2 = "#34d399"
ACCENT_3 = "#fbbf24"
TEXT = "#e4e4f0"
MUTED = "#8888a0"

PANEL_SIZE = 340

# ---------- DSP functions ----------

def convolve2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return out

def gaussian_kernel(size, sigma):
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    g = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return g / g.sum()

def normalize(a):
    a = np.abs(a)
    if a.max() > 0:
        a = a / a.max() * 255
    return a

def median_filter(img, size):
    pad = size // 2
    padded = np.pad(img, pad, mode="reflect")
    out = np.zeros_like(img)
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            window = padded[i:i + size, j:j + size]
            out[i, j] = np.median(window)
    return out

def non_max_suppression(mag, angle):
    h, w = mag.shape
    out = np.zeros_like(mag)
    angle = angle % 180
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            q, r = 255, 255
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = mag[i, j + 1], mag[i, j - 1]
            elif 22.5 <= a < 67.5:
                q, r = mag[i + 1, j - 1], mag[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q, r = mag[i + 1, j], mag[i - 1, j]
            elif 112.5 <= a < 157.5:
                q, r = mag[i - 1, j - 1], mag[i + 1, j + 1]
            if mag[i, j] >= q and mag[i, j] >= r:
                out[i, j] = mag[i, j]
    return out

def hysteresis(img, low, high):
    strong = 255
    weak = 75
    result = np.zeros_like(img)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    h, w = result.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if result[i, j] == weak:
                if strong in result[i - 1:i + 2, j - 1:j + 2]:
                    result[i, j] = strong
                else:
                    result[i, j] = 0
    return result

def canny_edge_detect(img, sigma):
    blurred = convolve2d(img, gaussian_kernel(5, sigma))
    gx = convolve2d(blurred, SOBEL_X)
    gy = convolve2d(blurred, SOBEL_X.T)
    mag = np.hypot(gx, gy)
    mag = mag / mag.max() * 255
    angle = np.degrees(np.arctan2(gy, gx))
    thin = non_max_suppression(mag, angle)
    result = hysteresis(thin, low=20, high=40)
    return result

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
SHARPEN = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)

def apply_filter(name, img, size, sigma):
    if name == "Original":
        return img
    if name == "Box blur":
        k = np.ones((size, size)) / (size * size)
        return convolve2d(img, k)
    if name == "Gaussian blur":
        k = gaussian_kernel(size, sigma)
        return convolve2d(img, k)
    if name == "Median":
        return median_filter(img, size)
    if name == "Sharpen":
        return convolve2d(img, SHARPEN)
    if name == "Sobel edges":
        gx = convolve2d(img, SOBEL_X)
        gy = convolve2d(img, SOBEL_X.T)
        return normalize(np.hypot(gx, gy))
    if name == "Prewitt edges":
        gx = convolve2d(img, PREWITT_X)
        gy = convolve2d(img, PREWITT_X.T)
        return normalize(np.hypot(gx, gy))
    if name == "Laplacian edges":
        return normalize(convolve2d(img, LAPLACIAN))
    if name == "Canny edges":
        return canny_edge_detect(img, sigma)
    return img

# ---------- App logic ----------

def open_image():
    global color_img, gray_array
    path = filedialog.askopenfilename()
    if path == "":
        return
    img = Image.open(path).convert("RGB")
    img.thumbnail((PANEL_SIZE, PANEL_SIZE))
    color_img = img
    gray_img = img.convert("L")
    gray_array = np.array(gray_img, dtype=np.float64)
    show(color_label, color_img)
    show(original_label, gray_array)
    update_result()

def update_result(event=None):
    global result_array
    if gray_array is None:
        return
    name = filter_choice.get()
    size = int(size_slider.get())
    if size % 2 == 0:
        size += 1
    sigma = sigma_slider.get()
    result_array = np.clip(apply_filter(name, gray_array, size, sigma), 0, 255)
    show(result_label, result_array)

def show(label_widget, content):
    if isinstance(content, Image.Image):
        img = content
    else:
        img = Image.fromarray(content.astype(np.uint8))
    ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
    label_widget.configure(image=ctk_img, text="")
    label_widget.image = ctk_img

def save_result():
    if result_array is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".png")
    if path:
        img = Image.fromarray(result_array.astype(np.uint8))
        img.save(path)

# ---------- Header card ----------

header_card = ctk.CTkFrame(window, fg_color=CARD, corner_radius=16)
header_card.pack(fill="x", padx=25, pady=(25, 15))

left_head = ctk.CTkFrame(header_card, fg_color="transparent")
left_head.pack(side="left", padx=20, pady=18)

ctk.CTkLabel(left_head, text="Image DSP Studio", font=ctk.CTkFont(size=20, weight="bold"),
             text_color=TEXT).pack(anchor="w")
ctk.CTkLabel(left_head, text="Spatial filtering & edge detection playground",
             font=ctk.CTkFont(size=12), text_color=MUTED).pack(anchor="w")

right_head = ctk.CTkFrame(header_card, fg_color="transparent")
right_head.pack(side="right", padx=20, pady=18)

open_button = ctk.CTkButton(right_head, text="Open Image", command=open_image,
                             corner_radius=10, height=38, width=130,
                             fg_color=ACCENT, hover_color="#8f6ff0",
                             text_color="#141420", font=ctk.CTkFont(weight="bold"))
open_button.pack(side="left", padx=5)

save_button = ctk.CTkButton(right_head, text="Save Result", command=save_result,
                             corner_radius=10, height=38, width=130,
                             fg_color="transparent", border_width=1, border_color=MUTED,
                             hover_color="#2a2a40", text_color=TEXT)
save_button.pack(side="left", padx=5)

# ---------- Controls card ----------

controls_card = ctk.CTkFrame(window, fg_color=CARD, corner_radius=16)
controls_card.pack(fill="x", padx=25, pady=(0, 15))

controls_inner = ctk.CTkFrame(controls_card, fg_color="transparent")
controls_inner.pack(padx=20, pady=18, fill="x")

ctk.CTkLabel(controls_inner, text="FILTER", font=ctk.CTkFont(size=11, weight="bold"),
             text_color=ACCENT_2).grid(row=0, column=0, sticky="w")
filter_choice = ctk.StringVar(value="Original")
filter_names = ["Original", "Box blur", "Gaussian blur", "Median", "Sharpen",
                 "Sobel edges", "Prewitt edges", "Laplacian edges", "Canny edges"]
dropdown = ctk.CTkOptionMenu(controls_inner, variable=filter_choice, values=filter_names,
                              command=update_result, corner_radius=10, width=170,
                              fg_color="#2a2a40", button_color=ACCENT,
                              button_hover_color="#8f6ff0")
dropdown.grid(row=1, column=0, sticky="w", pady=(5, 0))

ctk.CTkLabel(controls_inner, text="KERNEL SIZE", font=ctk.CTkFont(size=11, weight="bold"),
             text_color=ACCENT_2).grid(row=0, column=1, sticky="w", padx=(40, 0))
size_slider = ctk.CTkSlider(controls_inner, from_=3, to=15, number_of_steps=6,
                             command=update_result, width=160,
                             progress_color=ACCENT, button_color=ACCENT,
                             button_hover_color="#8f6ff0")
size_slider.set(5)
size_slider.grid(row=1, column=1, sticky="w", padx=(40, 0), pady=(5, 0))

ctk.CTkLabel(controls_inner, text="SIGMA", font=ctk.CTkFont(size=11, weight="bold"),
             text_color=ACCENT_2).grid(row=0, column=2, sticky="w", padx=(40, 0))
sigma_slider = ctk.CTkSlider(controls_inner, from_=0.5, to=5, number_of_steps=45,
                              command=update_result, width=160,
                              progress_color=ACCENT, button_color=ACCENT,
                              button_hover_color="#8f6ff0")
sigma_slider.set(1.5)
sigma_slider.grid(row=1, column=2, sticky="w", padx=(40, 0), pady=(5, 0))

# ---------- Image cards ----------

image_row = ctk.CTkFrame(window, fg_color="transparent")
image_row.pack(padx=25, pady=(0, 25), expand=True, fill="both")

def image_card(parent, title_text, dot_color):
    card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=16)
    top = ctk.CTkFrame(card, fg_color="transparent")
    top.pack(fill="x", padx=18, pady=(15, 5))
    ctk.CTkLabel(top, text="●", text_color=dot_color, font=ctk.CTkFont(size=14)).pack(side="left")
    ctk.CTkLabel(top, text=title_text, font=ctk.CTkFont(size=13, weight="bold"),
                 text_color=TEXT).pack(side="left", padx=(6, 0))
    img_label = ctk.CTkLabel(card, text="No image loaded", text_color=MUTED,
                              width=PANEL_SIZE, height=PANEL_SIZE)
    img_label.pack(padx=18, pady=(5, 18))
    return card, img_label

color_card, color_label = image_card(image_row, "Color", ACCENT_3)
color_card.pack(side="left", expand=True, fill="both", padx=(0, 10))

original_card, original_label = image_card(image_row, "Grayscale", ACCENT_2)
original_card.pack(side="left", expand=True, fill="both", padx=10)

result_card, result_label = image_card(image_row, "Result", ACCENT)
result_card.pack(side="left", expand=True, fill="both", padx=(10, 0))

window.mainloop()
