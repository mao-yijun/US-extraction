import os
import cv2
import numpy as np
import pydicom
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QHBoxLayout, QTextEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer


# 超声波图像区域检测和保存
def detect_and_save_us_region(image, output_folder, base_filename, image_format="jpeg"):
    if image.dtype != np.uint8:
        image = cv2.convertScaleAbs(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    max_area = 0
    max_label = 0
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > max_area:
            max_area = area
            max_label = i

    x, y, w, h, area = stats[max_label]
    us_image = np.zeros_like(image)
    us_image[y:y + h, x:x + w] = image[y:y + h, x:x + w]

    # 转换为 BGR 再保存
    us_image_bgr = cv2.cvtColor(us_image, cv2.COLOR_RGB2BGR)
    us_image_path = os.path.join(output_folder, f"{base_filename}_us_image.{image_format}")
    cv2.imwrite(us_image_path, us_image_bgr)

    return us_image, us_image_path


class USImageExtractor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("US Image Extractor")
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()

        folder_layout = QVBoxLayout()
        font = QFont("Arial", 14)

        self.input_folder_label = QLabel("Input Folder: Not Selected")
        self.input_folder_label.setFont(font)
        self.output_folder_label = QLabel("Output Folder: Not Selected")
        self.output_folder_label.setFont(font)

        self.select_input_button = QPushButton("Select Input Folder")
        self.select_input_button.setFont(font)
        self.select_input_button.clicked.connect(self.select_input_folder)

        self.select_output_button = QPushButton("Select Output Folder")
        self.select_output_button.setFont(font)
        self.select_output_button.clicked.connect(self.select_output_folder)

        # 新增选择输出图像格式的下拉框
        self.format_label = QLabel("Output Image Format:")
        self.format_label.setFont(font)
        self.image_format_combo = QComboBox()
        self.image_format_combo.setFont(font)
        self.image_format_combo.addItems(["jpeg", "jpg", "png", "bmp", "tiff"])

        self.process_button = QPushButton("Process Images")
        self.process_button.setFont(font)
        self.process_button.clicked.connect(self.process_images)
        self.process_button.setEnabled(False)

        folder_layout.addWidget(self.input_folder_label)
        folder_layout.addWidget(self.select_input_button)
        folder_layout.addWidget(self.output_folder_label)
        folder_layout.addWidget(self.select_output_button)
        folder_layout.addWidget(self.format_label)
        folder_layout.addWidget(self.image_format_combo)
        folder_layout.addWidget(self.process_button)

        image_layout = QHBoxLayout()
        image_font = QFont("Arial", 18)

        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setFont(image_font)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setFixedSize(500, 500)

        self.us_image_label = QLabel("US Image")
        self.us_image_label.setFont(image_font)
        self.us_image_label.setAlignment(Qt.AlignCenter)
        self.us_image_label.setFixedSize(500, 500)

        image_layout.addWidget(self.original_image_label)
        image_layout.addWidget(self.us_image_label)

        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setFont(font)
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(200)
        log_layout.addWidget(self.log_output)

        main_layout.addLayout(folder_layout)
        main_layout.addLayout(image_layout)
        main_layout.addLayout(log_layout)

        self.setLayout(main_layout)

    def select_input_folder(self):
        image_files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.bmp *.jpg *.jpeg *.png *.tiff *.gif *.*)"
        )
        if image_files:
            self.selected_images = image_files
            self.input_folder = os.path.dirname(self.selected_images[0])
            self.input_folder_label.setText(f"Input Folder: {self.input_folder}")
            self.log_output.append(f"Selected Input Images: {', '.join(self.selected_images)}")
            self.check_folders_selected()
            # 选取图像后加载第一张原始图像进行展示
            first_image = self.load_image(self.selected_images[0])
            if first_image is not None:
                self.original_image_label.setPixmap(self.convert_to_pixmap(first_image))
            self.us_image_label.setText("US Image")

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_label.setText(f"Output Folder: {folder}")
            self.log_output.append(f"Selected Output Folder: {folder}")
            self.check_folders_selected()

    def check_folders_selected(self):
        if hasattr(self, 'input_folder') and hasattr(self, 'output_folder'):
            self.process_button.setEnabled(True)

    def process_images(self):
        if not hasattr(self, 'selected_images') or not self.selected_images:
            QMessageBox.warning(self, "No Images Selected", "Please select at least one image to process.")
            return

        image_format = self.image_format_combo.currentText()
        for image_path in self.selected_images:
            image = self.load_image(image_path)
            if image is not None:
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                us_image, us_image_path = detect_and_save_us_region(image, self.output_folder, base_filename, image_format)
                self.log_output.append(f"Processed: {image_path} -> {us_image_path}")
                self.display_images(image, us_image)
        # 显示处理完成的弹窗，1秒后自动关闭
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Processing Complete")
        msg_box.setText("All images processed.")
        msg_box.setStyleSheet("QLabel{min-width: 300px; font-size: 14pt;}")
        # msg_box.setStandardButtons(QMessageBox.NoButton)
        msg_box.show()
        QTimer.singleShot(1000, lambda: msg_box.close())

    def load_image(self, image_path):
        try:
            if '.' not in os.path.basename(image_path) or image_path.lower().endswith('.dcm'):
                dicom_image = pydicom.dcmread(image_path)
                image = dicom_image.pixel_array
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                if image.dtype != np.uint8:
                    image = cv2.convertScaleAbs(image)
            else:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            self.log_output.append(f"Error loading image {image_path}: {e}")
            return None

    def display_images(self, original_image, us_image):
        self.original_image_label.setPixmap(self.convert_to_pixmap(original_image))
        self.us_image_label.setPixmap(self.convert_to_pixmap(us_image))

    def convert_to_pixmap(self, image):
        height, width, channel = image.shape
        qimage = QImage(image.data, width, height, width * channel, QImage.Format_RGB888)
        return QPixmap.fromImage(qimage).scaled(500, 500, Qt.KeepAspectRatio)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    extractor = USImageExtractor()
    extractor.show()
    sys.exit(app.exec_())






