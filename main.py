import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing and Geometric Transformations")
        self.root.configure(bg="lightgray")

        # Frame for control buttons
        button_frame = tk.Frame(self.root, bg="lightgray")
        button_frame.pack(side="left", fill="y")

        # Buttons for each functionality
        tk.Button(button_frame, text="Load Image", command=self.load_image).pack(pady=5)
        tk.Button(button_frame, text="Convert to Grayscale", command=self.convert_to_grayscale).pack(pady=5)
        tk.Button(button_frame, text="Spatial Filters", command=self.open_filter_window).pack(pady=5)
        tk.Button(button_frame, text="Contrast Adjustment", command=self.adjust_contrast).pack(pady=5)
        tk.Button(button_frame, text="Morphological Operations", command=self.morphological_operations).pack(pady=5)
        tk.Button(button_frame, text="Segmentation and Contours", command=self.segment_and_find_contours).pack(pady=5)
        tk.Button(button_frame, text="Custom Filters", command=self.open_custom_filter_window).pack(pady=5)
        tk.Button(button_frame, text="Display Histogram", command=self.display_histogram).pack(pady=5)
        tk.Button(button_frame, text="Equalize Histogram", command=self.equalize_histogram).pack(pady=5)
        tk.Button(button_frame, text="Geometric Transformations", command=self.geometric_transformations).pack(pady=5)

        # Canvas for image display
        self.original_canvas = tk.Canvas(self.root, width=400, height=400, bg="white")
        self.original_canvas.pack(side="left", padx=10, pady=10)
        self.processed_canvas = tk.Canvas(self.root, width=400, height=400, bg="white")
        self.processed_canvas.pack(side="left", padx=10, pady=10)

        # Label for image information
        self.info_label = tk.Label(self.root, text="", bg="lightgray")
        self.info_label.pack(pady=5)

        self.original_image = None
        self.processed_image = None
        self.pontos = []
        self.num_pontos = 0

    def load_image(self):
        """Load an image from the file system."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = self.resize_image(self.original_image, 0.5)  # Resize the image to 50%
            self.display_image(self.original_image, self.original_canvas)
            self.display_image_info(self.original_image)

    def display_image(self, image, canvas):
        """Display the image on the Tkinter Canvas."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk  # Store the reference to avoid garbage collection

    def display_image_info(self, image):
        """Display basic image information."""
        height, width, channels = image.shape
        info_text = f"Dimensions: {width}x{height}, Channels: {channels}"
        self.info_label.config(text=info_text)

    def convert_to_grayscale(self):
        """Convert the image to grayscale."""
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = gray_image
            self.display_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), self.processed_canvas)

    def open_filter_window(self):
        """Open a new window to apply spatial filters."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Spatial Filters")
            window.configure(bg="lightgray")

            def apply_filter(filter_type):
                kernel_size = int(kernel_size_slider.get())
                if filter_type == "mean":
                    filtered_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
                elif filter_type == "gaussian":
                    filtered_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
                elif filter_type == "median":
                    filtered_image = cv2.medianBlur(self.original_image, kernel_size)
                elif filter_type == "laplacian":
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    filtered_image = cv2.Laplacian(gray_image, cv2.CV_64F)
                    filtered_image = cv2.convertScaleAbs(filtered_image)
                elif filter_type == "sobel":
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
                    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
                    filtered_image = cv2.convertScaleAbs(cv2.sqrt(sobelx ** 2 + sobely ** 2))
                self.processed_image = filtered_image
                self.display_image(filtered_image if len(filtered_image.shape) == 3 else cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR), self.processed_canvas)

            kernel_size_slider = tk.Scale(window, from_=3, to=15, orient=tk.HORIZONTAL, label="Kernel Size", bg="lightgray")
            kernel_size_slider.set(3)
            kernel_size_slider.pack(pady=5)

            tk.Button(window, text="Mean Filter", command=lambda: apply_filter('mean')).pack(pady=5)
            tk.Button(window, text="Gaussian Filter", command=lambda: apply_filter('gaussian')).pack(pady=5)
            tk.Button(window, text="Median Filter", command=lambda: apply_filter('median')).pack(pady=5)
            tk.Button(window, text="Laplacian Filter", command=lambda: apply_filter('laplacian')).pack(pady=5)
            tk.Button(window, text="Sobel Filter", command=lambda: apply_filter('sobel')).pack(pady=5)

    def adjust_contrast(self):
        """Adjust brightness and contrast using sliders."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Brightness/Contrast Adjustment")
            window.configure(bg="lightgray")

            def update(_):
                alpha = contrast_slider.get() / 100
                beta = brightness_slider.get() - 50
                adjusted_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)
                self.display_image(adjusted_image, self.processed_canvas)

            contrast_slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, label="Contrast", command=update, bg="lightgray")
            contrast_slider.set(50)
            contrast_slider.pack(pady=5)

            brightness_slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, label="Brightness", command=update, bg="lightgray")
            brightness_slider.set(50)
            brightness_slider.pack(pady=5)

    def resize_image(self, image, scale):
        """Resize the image to a smaller version."""
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized_image

    def apply_morphological_operation(self, operation, kernel_size):
        """Apply morphological operations like erosion, dilation, etc."""
        if self.original_image is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            if operation == 'erosion':
                processed_image = cv2.erode(self.original_image, kernel)
            elif operation == 'dilation':
                processed_image = cv2.dilate(self.original_image, kernel)
            elif operation == 'opening':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
            elif operation == 'gradient':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_GRADIENT, kernel)
            self.display_image(processed_image, self.processed_canvas)

    def morphological_operations(self):
        """Open a new window to apply morphological operations."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Morphological Operations")
            window.configure(bg="lightgray")

            def apply(operation):
                kernel_size = int(kernel_size_slider.get())
                self.apply_morphological_operation(operation, kernel_size)

            kernel_size_slider = tk.Scale(window, from_=3, to=15, orient=tk.HORIZONTAL, label="Kernel Size", bg="lightgray")
            kernel_size_slider.set(3)
            kernel_size_slider.pack(pady=5)

            tk.Button(window, text="Erosion", command=lambda: apply('erosion')).pack(pady=5)
            tk.Button(window, text="Dilation", command=lambda: apply('dilation')).pack(pady=5)
            tk.Button(window, text="Opening", command=lambda: apply('opening')).pack(pady=5)
            tk.Button(window, text="Closing", command=lambda: apply('closing')).pack(pady=5)
            tk.Button(window, text="Gradient", command=lambda: apply('gradient')).pack(pady=5)

    def segment_and_find_contours(self):
        """Segment the image and find contours."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Segmentation and Contours")
            window.configure(bg="lightgray")

            def apply_segmentation():
                threshold_value = threshold_slider.get()
                gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_image = self.original_image.copy()
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
                self.display_image(contour_image, self.processed_canvas)

            threshold_slider = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold", bg="lightgray")
            threshold_slider.set(128)
            threshold_slider.pack(pady=5)

            tk.Button(window, text="Apply", command=apply_segmentation).pack(pady=5)

    def open_custom_filter_window(self):
        """Open a new window to apply a custom filter."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Custom Filter")
            window.configure(bg="lightgray")

            def apply_filter():
                kernel = np.array([
                    [int(entry_00.get()), int(entry_01.get()), int(entry_02.get())],
                    [int(entry_10.get()), int(entry_11.get()), int(entry_12.get())],
                    [int(entry_20.get()), int(entry_21.get()), int(entry_22.get())]
                ], dtype=np.float32)
                custom_filtered_image = cv2.filter2D(self.original_image, -1, kernel)
                self.display_image(custom_filtered_image, self.processed_canvas)

            entries = []
            for i in range(3):
                row = []
                for j in range(3):
                    entry = tk.Entry(window, width=5)
                    entry.grid(row=i, column=j, padx=5, pady=5)
                    row.append(entry)
                entries.append(row)

            entry_00, entry_01, entry_02 = entries[0]
            entry_10, entry_11, entry_12 = entries[1]
            entry_20, entry_21, entry_22 = entries[2]

            tk.Button(window, text="Apply Filter", command=apply_filter).grid(row=3, columnspan=3, pady=10)

    def display_histogram(self):
        """Display the histogram of the image."""
        if self.processed_image is not None:
            # If the image is grayscale, show a single histogram
            if len(self.processed_image.shape) == 2:
                histogram = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
                plt.plot(histogram, color='black')
                plt.xlim([0, 256])
            else:
                # Display histogram for each color channel
                for i, color in enumerate(['b', 'g', 'r']):
                    histogram = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                    plt.plot(histogram, color=color)
                    plt.xlim([0, 256])
            plt.show()

    def equalize_histogram(self):
        """Equalize the histogram of the image."""
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:  # Grayscale
                equalized_image = cv2.equalizeHist(self.processed_image)
            else:  # Color image, convert to YCrCb and equalize the Y channel
                ycrcb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            self.display_image(equalized_image, self.processed_canvas)

    def desenhar_pontos(self, img, pontos, cor=(0, 255, 0), espessura=2):
        for i, ponto in enumerate(pontos):
            cv2.circle(img, (int(ponto[0]), int(ponto[1])), 5, cor, -1)
            if i > 0:
                cv2.line(img, (int(pontos[i-1][0]), int(pontos[i-1][1])), (int(ponto[0]), int(ponto[1])), cor, espessura)
        if len(pontos) > 1:
            cv2.line(img, (int(pontos[-1][0]), int(pontos[-1][1])), (int(pontos[0][0]), int(pontos[0][1])), cor, espessura)

    def transladar(self, pontos, tx, ty):
        matriz_translacao = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
        pontos_transladados = matriz_translacao @ pontos_homogeneos.T
        return pontos_transladados[:2].T

    def rotacionar(self, pontos, angulo, ponto_ancoragem):
        rad = np.deg2rad(angulo)
        matriz_rotacao = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        pontos_homogeneos = np.hstack([pontos - ponto_ancoragem, np.ones((pontos.shape[0], 1))])
        pontos_rotacionados = matriz_rotacao @ pontos_homogeneos.T
        return (pontos_rotacionados[:2].T + ponto_ancoragem)

    def escalonar(self, pontos, sx, sy):
        matriz_escalonamento = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
        pontos_escalonados = matriz_escalonamento @ pontos_homogeneos.T
        return pontos_escalonados[:2].T

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pontos) < self.num_pontos:
            self.pontos.append([x, y])

    def geometric_transformations(self):
        self.num_pontos = int(input("Digite o número de pontos: "))
        self.pontos = []

        cv2.namedWindow("Transformacoes Geometricas")
        cv2.setMouseCallback("Transformacoes Geometricas", self.mouse_callback)

        while len(self.pontos) < self.num_pontos:
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            self.desenhar_pontos(img, self.pontos)
            cv2.imshow("Transformacoes Geometricas", img)
            cv2.waitKey(1)

        while True:
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            self.desenhar_pontos(img, self.pontos)
            cv2.imshow("Transformacoes Geometricas", img)
            cv2.waitKey(1)
            print("Escolha a transformação geométrica:")
            print("1. Translação")
            print("2. Rotação")
            print("3. Escalonamento")
            print("4. Resetar")
            print("5. Sair")
            escolha = int(input("Digite o número da opção desejada: "))
            if escolha == 1:
                tx = float(input("Digite o valor de deslocamento em x: "))
                ty = float(input("Digite o valor de deslocamento em y: "))
                self.pontos = self.transladar(self.pontos, tx, ty)
            elif escolha == 2:
                angulo = float(input("Digite o ângulo de rotação em graus: "))
                ponto_ancoragem = np.mean(self.pontos, axis=0)  # Usar o centroide como ponto de ancoragem
                self.pontos = self.rotacionar(self.pontos, angulo, ponto_ancoragem)
            elif escolha == 3:
                sx = float(input("Digite o fator de escala em x: "))
                sy = float(input("Digite o fator de escala em y: "))
                self.pontos = self.escalonar(self.pontos, sx, sy)
            elif escolha == 4:
                self.pontos = []
                self.num_pontos = int(input("Digite o número de pontos: "))
                while len(self.pontos) < self.num_pontos:
                   import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing and Geometric Transformations")
        self.root.configure(bg="lightgray")

        # Frame for control buttons
        button_frame = tk.Frame(self.root, bg="lightgray")
        button_frame.pack(side="left", fill="y")

        # Buttons for each functionality
        tk.Button(button_frame, text="Load Image", command=self.load_image).pack(pady=5)
        tk.Button(button_frame, text="Convert to Grayscale", command=self.convert_to_grayscale).pack(pady=5)
        tk.Button(button_frame, text="Spatial Filters", command=self.open_filter_window).pack(pady=5)
        tk.Button(button_frame, text="Contrast Adjustment", command=self.adjust_contrast).pack(pady=5)
        tk.Button(button_frame, text="Morphological Operations", command=self.morphological_operations).pack(pady=5)
        tk.Button(button_frame, text="Segmentation and Contours", command=self.segment_and_find_contours).pack(pady=5)
        tk.Button(button_frame, text="Custom Filters", command=self.open_custom_filter_window).pack(pady=5)
        tk.Button(button_frame, text="Display Histogram", command=self.display_histogram).pack(pady=5)
        tk.Button(button_frame, text="Equalize Histogram", command=self.equalize_histogram).pack(pady=5)
        tk.Button(button_frame, text="Geometric Transformations", command=self.geometric_transformations).pack(pady=5)

        # Canvas for image display
        self.original_canvas = tk.Canvas(self.root, width=400, height=400, bg="white")
        self.original_canvas.pack(side="left", padx=10, pady=10)
        self.processed_canvas = tk.Canvas(self.root, width=400, height=400, bg="white")
        self.processed_canvas.pack(side="left", padx=10, pady=10)

        # Label for image information
        self.info_label = tk.Label(self.root, text="", bg="lightgray")
        self.info_label.pack(pady=5)

        self.original_image = None
        self.processed_image = None
        self.pontos = []
        self.num_pontos = 0

    def load_image(self):
        """Load an image from the file system."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = self.resize_image(self.original_image, 0.5)  # Resize the image to 50%
            self.display_image(self.original_image, self.original_canvas)
            self.display_image_info(self.original_image)

    def display_image(self, image, canvas):
        """Display the image on the Tkinter Canvas."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk  # Store the reference to avoid garbage collection

    def display_image_info(self, image):
        """Display basic image information."""
        height, width, channels = image.shape
        info_text = f"Dimensions: {width}x{height}, Channels: {channels}"
        self.info_label.config(text=info_text)

    def convert_to_grayscale(self):
        """Convert the image to grayscale."""
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = gray_image
            self.display_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), self.processed_canvas)

    def open_filter_window(self):
        """Open a new window to apply spatial filters."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Spatial Filters")
            window.configure(bg="lightgray")

            def apply_filter(filter_type):
                kernel_size = int(kernel_size_slider.get())
                if filter_type == "mean":
                    filtered_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
                elif filter_type == "gaussian":
                    filtered_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
                elif filter_type == "median":
                    filtered_image = cv2.medianBlur(self.original_image, kernel_size)
                elif filter_type == "laplacian":
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    filtered_image = cv2.Laplacian(gray_image, cv2.CV_64F)
                    filtered_image = cv2.convertScaleAbs(filtered_image)
                elif filter_type == "sobel":
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
                    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
                    filtered_image = cv2.convertScaleAbs(cv2.sqrt(sobelx ** 2 + sobely ** 2))
                self.processed_image = filtered_image
                self.display_image(filtered_image if len(filtered_image.shape) == 3 else cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR), self.processed_canvas)

            kernel_size_slider = tk.Scale(window, from_=3, to=15, orient=tk.HORIZONTAL, label="Kernel Size", bg="lightgray")
            kernel_size_slider.set(3)
            kernel_size_slider.pack(pady=5)

            tk.Button(window, text="Mean Filter", command=lambda: apply_filter('mean')).pack(pady=5)
            tk.Button(window, text="Gaussian Filter", command=lambda: apply_filter('gaussian')).pack(pady=5)
            tk.Button(window, text="Median Filter", command=lambda: apply_filter('median')).pack(pady=5)
            tk.Button(window, text="Laplacian Filter", command=lambda: apply_filter('laplacian')).pack(pady=5)
            tk.Button(window, text="Sobel Filter", command=lambda: apply_filter('sobel')).pack(pady=5)

    def adjust_contrast(self):
        """Adjust brightness and contrast using sliders."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Brightness/Contrast Adjustment")
            window.configure(bg="lightgray")

            def update(_):
                alpha = contrast_slider.get() / 100
                beta = brightness_slider.get() - 50
                adjusted_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)
                self.display_image(adjusted_image, self.processed_canvas)

            contrast_slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, label="Contrast", command=update, bg="lightgray")
            contrast_slider.set(50)
            contrast_slider.pack(pady=5)

            brightness_slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, label="Brightness", command=update, bg="lightgray")
            brightness_slider.set(50)
            brightness_slider.pack(pady=5)

    def resize_image(self, image, scale):
        """Resize the image to a smaller version."""
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized_image

    def apply_morphological_operation(self, operation, kernel_size):
        """Apply morphological operations like erosion, dilation, etc."""
        if self.original_image is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            if operation == 'erosion':
                processed_image = cv2.erode(self.original_image, kernel)
            elif operation == 'dilation':
                processed_image = cv2.dilate(self.original_image, kernel)
            elif operation == 'opening':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
            elif operation == 'gradient':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_GRADIENT, kernel)
            self.display_image(processed_image, self.processed_canvas)

    def morphological_operations(self):
        """Open a new window to apply morphological operations."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Morphological Operations")
            window.configure(bg="lightgray")

            def apply(operation):
                kernel_size = int(kernel_size_slider.get())
                self.apply_morphological_operation(operation, kernel_size)

            kernel_size_slider = tk.Scale(window, from_=3, to=15, orient=tk.HORIZONTAL, label="Kernel Size", bg="lightgray")
            kernel_size_slider.set(3)
            kernel_size_slider.pack(pady=5)

            tk.Button(window, text="Erosion", command=lambda: apply('erosion')).pack(pady=5)
            tk.Button(window, text="Dilation", command=lambda: apply('dilation')).pack(pady=5)
            tk.Button(window, text="Opening", command=lambda: apply('opening')).pack(pady=5)
            tk.Button(window, text="Closing", command=lambda: apply('closing')).pack(pady=5)
            tk.Button(window, text="Gradient", command=lambda: apply('gradient')).pack(pady=5)

    def segment_and_find_contours(self):
        """Segment the image and find contours."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Segmentation and Contours")
            window.configure(bg="lightgray")

            def apply_segmentation():
                threshold_value = threshold_slider.get()
                gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_image = self.original_image.copy()
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
                self.display_image(contour_image, self.processed_canvas)

            threshold_slider = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold", bg="lightgray")
            threshold_slider.set(128)
            threshold_slider.pack(pady=5)

            tk.Button(window, text="Apply", command=apply_segmentation).pack(pady=5)

    def open_custom_filter_window(self):
        """Open a new window to apply a custom filter."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Custom Filter")
            window.configure(bg="lightgray")

            def apply_filter():
                kernel = np.array([
                    [int(entry_00.get()), int(entry_01.get()), int(entry_02.get())],
                    [int(entry_10.get()), int(entry_11.get()), int(entry_12.get())],
                    [int(entry_20.get()), int(entry_21.get()), int(entry_22.get())]
                ], dtype=np.float32)
                custom_filtered_image = cv2.filter2D(self.original_image, -1, kernel)
                self.display_image(custom_filtered_image, self.processed_canvas)

            entries = []
            for i in range(3):
                row = []
                for j in range(3):
                    entry = tk.Entry(window, width=5)
                    entry.grid(row=i, column=j, padx=5, pady=5)
                    row.append(entry)
                entries.append(row)

            entry_00, entry_01, entry_02 = entries[0]
            entry_10, entry_11, entry_12 = entries[1]
            entry_20, entry_21, entry_22 = entries[2]

            tk.Button(window, text="Apply Filter", command=apply_filter).grid(row=3, columnspan=3, pady=10)

    def display_histogram(self):
        """Display the histogram of the image."""
        if self.processed_image is not None:
            # If the image is grayscale, show a single histogram
            if len(self.processed_image.shape) == 2:
                histogram = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
                plt.plot(histogram, color='black')
                plt.xlim([0, 256])
            else:
                # Display histogram for each color channel
                for i, color in enumerate(['b', 'g', 'r']):
                    histogram = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                    plt.plot(histogram, color=color)
                    plt.xlim([0, 256])
            plt.show()

    def equalize_histogram(self):
        """Equalize the histogram of the image."""
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:  # Grayscale
                equalized_image = cv2.equalizeHist(self.processed_image)
            else:  # Color image, convert to YCrCb and equalize the Y channel
                ycrcb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            self.display_image(equalized_image, self.processed_canvas)

    def desenhar_pontos(self, img, pontos, cor=(0, 255, 0), espessura=2):
        for i, ponto in enumerate(pontos):
            cv2.circle(img, (int(ponto[0]), int(ponto[1])), 5, cor, -1)
            if i > 0:
                cv2.line(img, (int(pontos[i-1][0]), int(pontos[i-1][1])), (int(ponto[0]), int(ponto[1])), cor, espessura)
        if len(pontos) > 1:
            cv2.line(img, (int(pontos[-1][0]), int(pontos[-1][1])), (int(pontos[0][0]), int(pontos[0][1])), cor, espessura)

    def transladar(self, pontos, tx, ty):
        matriz_translacao = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
        pontos_transladados = matriz_translacao @ pontos_homogeneos.T
        return pontos_transladados[:2].T

    def rotacionar(self, pontos, angulo, ponto_ancoragem):
        rad = np.deg2rad(angulo)
        matriz_rotacao = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        pontos_homogeneos = np.hstack([pontos - ponto_ancoragem, np.ones((pontos.shape[0], 1))])
        pontos_rotacionados = matriz_rotacao @ pontos_homogeneos.T
        return (pontos_rotacionados[:2].T + ponto_ancoragem)

    def escalonar(self, pontos, sx, sy):
        matriz_escalonamento = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
        pontos_homogeneos = np.hstack([pontos, np.ones((pontos.shape[0], 1))])
        pontos_escalonados = matriz_escalonamento @ pontos_homogeneos.T
        return pontos_escalonados[:2].T

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.pontos) < self.num_pontos:
            self.pontos.append([x, y])

    def geometric_transformations(self):
        self.num_pontos = int(input("Digite o número de pontos: "))
        self.pontos = []

        cv2.namedWindow("Transformacoes Geometricas")
        cv2.setMouseCallback("Transformacoes Geometricas", self.mouse_callback)

        while len(self.pontos) < self.num_pontos:
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            self.desenhar_pontos(img, self.pontos)
            cv2.imshow("Transformacoes Geometricas", img)
            cv2.waitKey(1)

        while True:
            img = np.zeros((500, 500, 3), dtype=np.uint8)
            self.desenhar_pontos(img, self.pontos)
            cv2.imshow("Transformacoes Geometricas", img)
            cv2.waitKey(1)
            print("Escolha a transformação geométrica:")
            print("1. Translação")
            print("2. Rotação")
            print("3. Escalonamento")
            print("4. Resetar")
            print("5. Sair")
            escolha = int(input("Digite o número da opção desejada: "))
            if escolha == 1:
                tx = float(input("Digite o valor de deslocamento em x: "))
                ty = float(input("Digite o valor de deslocamento em y: "))
                self.pontos = self.transladar(self.pontos, tx, ty)
            elif escolha == 2:
                angulo = float(input("Digite o ângulo de rotação em graus: "))
                ponto_ancoragem = np.mean(self.pontos, axis=0)  # Usar o centroide como ponto de ancoragem
                self.pontos = self.rotacionar(self.pontos, angulo, ponto_ancoragem)
            elif escolha == 3:
                sx = float(input("Digite o fator de escala em x: "))
                sy = float(input("Digite o fator de escala em y: "))
                self.pontos = self.escalonar(self.pontos, sx, sy)
            elif escolha == 4:
                self.pontos = []
                self.num_pontos = int(input("Digite o número de pontos: "))
                while len(self.pontos) < self.num_pontos:
                    img = np.zeros((500, 500, 3), dtype=np.uint8)
                    self.desenhar_pontos(img, self.pontos)
                    cv2.imshow("Transformacoes Geometricas", img)
                    cv2.waitKey(1)
            elif escolha == 5:
                cv2.destroyAllWindows()
                return
            else:
                print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()