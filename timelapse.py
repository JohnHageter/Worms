from Module.ui.Timelapse.app import TimelapseApp
import sys
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)
win = TimelapseApp()
win.show()
sys.exit(app.exec())

