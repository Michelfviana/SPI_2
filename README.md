# SPI

![Project Image](src/Screenshot%20from%202024-09-20%2015-26-27.png)

A Python-based software for image processing using a graphical interface built with Tkinter. This application allows users to load images, convert them to grayscale, apply filters, adjust brightness and contrast, and perform geometric transformations. It uses OpenCV for image manipulation and PIL (Pillow) for displaying images. The main features include:

- Loading images from the filesystem and displaying them.
- Converting images to grayscale.
- Applying spatial filters, such as Mean, Gaussian, Median, Laplacian, and Sobel.
- Adjusting brightness and contrast using interactive sliders in a separate window.
- Applying custom filters.
- Performing morphological operations like erosion, dilation, opening, closing, and gradient.
- Segmenting the image and finding contours.
- Displaying and equalizing histograms.
- Performing geometric transformations like translation, rotation, and scaling.

The user interface includes buttons for each function, a canvas to display the original and processed images, and a label to show basic information about the images.

## Requirements

Make sure you have Python 3.x installed. You can check your Python version with the command:

```bash
python --version
```

## Cloning the Repository

Clone this repository using the `git` command:

```bash
git clone https://github.com/Michelfviana/SPI_2.git
```

## Environment Setup

1. Navigate to the project directory:

   ```bash
   cd repository-name
   ```

2. Create a virtual environment to isolate the project dependencies:

   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows:**

     ```bash
     venv\Scripts\activate
     ```

   - **Linux/macOS:**

     ```bash
     source venv/bin/activate
     ```

4. Install the project dependencies listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

With the virtual environment activated and the dependencies installed, you can run the project:

```bash
python main.py
```
