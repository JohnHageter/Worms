DARK_STYLESHEET = """
QWidget {
    background-color: #121212;
    color: #E0E0E0;
}

QPushButton {
    background-color: #1E1E1E;
    border: 1px solid #333;
    padding: 5px;
    border-radius: 4px;
}

QPushButton:hover:!disabled {
    background-color: #2A2A2A;
    color: #FFFFFF;
}

QPushButton:pressed {
    background-color: #333;
}

QPushButton:disabled {
    background-color: #444;
    color: #888;
    border: 1px solid #666
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1E1E1E;
    border: 1px solid #333;
}

QSlider::groove:horizontal {
    background: #333;
    height: 6px;
}

QSlider::handle:horizontal {
    background: #888;
    width: 12px;
}
"""
