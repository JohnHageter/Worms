from PySide6.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QPushButton,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, QDateTime
from Module.ui.utils.logger import logger

class ConsoleDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Console", parent)

        self.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )

        # ---------- Container ----------
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ---------- Log view ----------
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        layout.addWidget(self.console)

        # ---------- Controls ----------
        controls = QHBoxLayout()
        controls.addStretch()

        self.clear_btn = QPushButton("Clear")
        controls.addWidget(self.clear_btn)

        layout.addLayout(controls)

        self.setWidget(container)

        self.setStyleSheet(
            """
            QTextEdit {
                background-color: #111;
                color: #ddd;
                font-family: Consolas;
                font-size: 11px;
            }
            """
        )

        self.clear_btn.clicked.connect(self.console.clear)
        logger.log_signal.connect(self.log)

    def log(self, message: str, level: str = "INFO"):
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.console.append(f"[{timestamp}] [{level}] {message}")
