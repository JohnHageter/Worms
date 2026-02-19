from Module.ui.main import TimelapseApp
import sys
from PySide6.QtWidgets import QApplication
import cv2

app = QApplication(sys.argv)
win = TimelapseApp()
win.show()
sys.exit(app.exec())