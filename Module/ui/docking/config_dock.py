from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QGroupBox, QHBoxLayout
from PySide6.QtCore import Qt


class ConfigDock(QDockWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
            | Qt.DockWidgetArea.TopDockWidgetArea
            | Qt.DockWidgetArea.BottomDockWidgetArea
        )

        self._container = QWidget(self)
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(6)

        self._layout.addStretch(1)

        self.setWidget(self._container)

    def add_panel(self, panel: QWidget):
        self._layout.insertWidget(self._layout.count() - 1, panel)


class ConfigPanel(QGroupBox):
    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(4)

    def add_row(self, row: QHBoxLayout):
        self._layout.addLayout(row)

    def add_widget(self, widget: QWidget):
        self._layout.addWidget(widget)
