from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QDockWidget,
    QWidget,
    QVBoxLayout,
)
from PySide6.QtCore import Qt
import sys


class TimelapseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planaria Tracker")
        self.resize(1400, 800)

        # --- Central camera preview ---
        preview = QLabel("Camera Preview")
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setStyleSheet("background: black; color: white;")
        self.setCentralWidget(preview)

        # --- Docks ---
        self.camera_dock = self.make_dock("Camera Configuration")
        self.save_dock = self.make_dock("Save Settings")
        self.tracking_dock = self.make_dock("Tracking Configuration")
        self.console_dock = self.make_dock("Console")

        # --- Dock placement ---
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.camera_dock)
        self.tabifyDockWidget(self.camera_dock, self.save_dock)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.tracking_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console_dock)

        # Start with some hidden
        self.tracking_dock.hide()
        self.console_dock.hide()

        # --- Menu to toggle docks ---
        self.create_view_menu()

    def make_dock(self, title: str) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.addWidget(QLabel(title))
        layout.addStretch()

        dock.setWidget(content)
        return dock

    def create_view_menu(self):
        view_menu = self.menuBar().addMenu("View")

        for dock in [
            self.camera_dock,
            self.save_dock,
            self.tracking_dock,
            self.console_dock,
        ]:
            view_menu.addAction(dock.toggleViewAction())

