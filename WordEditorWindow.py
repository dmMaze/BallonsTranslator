import sys
from qtpy.QtWidgets import (
    QApplication, QDialog, QLabel, QLineEdit, QPushButton, QTextEdit,
    QGroupBox, QListWidget, QHBoxLayout, QVBoxLayout, QSizePolicy
)
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt


class WordEditorWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Word Editor Window")
        self.setMinimumSize(800, 600)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Upper part: Display an image.
        self.image_label = QLabel(self)
        # Replace "path/to/your/image.png" with the path to your image file.
        pixmap = QPixmap("path/to/your/image.png")
        if pixmap.isNull():
            # If the image is not found, show a placeholder text.
            self.image_label.setText("Image not found")
            self.image_label.setAlignment(Qt.AlignCenter)
            # Optionally, give it a fixed height.
            self.image_label.setMinimumHeight(150)
        else:
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
            self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.image_label)

        # 2. Second row: A long block of text on the left and a button to the right.
        text_row_layout = QHBoxLayout()
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \n"
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n"
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
        )
        self.text_button = QPushButton("Action", self)
        # Give the text area more space than the button.
        text_row_layout.addWidget(self.text_edit, stretch=3)
        text_row_layout.addWidget(self.text_button, stretch=1)
        main_layout.addLayout(text_row_layout)

        # 3. Third row: Two groups side by side.
        groups_layout = QHBoxLayout()

        # 3.a Left group: "Word not on"
        self.group_word_not_on = QGroupBox("Word not on", self)
        word_group_layout = QVBoxLayout(self.group_word_not_on)
        # Word editing field at the top.
        self.word_line_edit = QLineEdit(self.group_word_not_on)
        self.word_line_edit.setPlaceholderText("Edit word here...")
        word_group_layout.addWidget(self.word_line_edit)
        # Row of buttons.
        word_buttons_layout = QHBoxLayout()
        self.change_all_btn = QPushButton("Change All", self.group_word_not_on)
        self.change_btn = QPushButton("Change", self.group_word_not_on)
        self.skip_btn = QPushButton("Skip", self.group_word_not_on)
        self.skip_all_btn = QPushButton("Skip All", self.group_word_not_on)
        self.add_btn = QPushButton("Add (case)", self.group_word_not_on)  # "Add a name taking into account the case"
        word_buttons_layout.addWidget(self.change_all_btn)
        word_buttons_layout.addWidget(self.change_btn)
        word_buttons_layout.addWidget(self.skip_btn)
        word_buttons_layout.addWidget(self.skip_all_btn)
        word_buttons_layout.addWidget(self.add_btn)
        word_group_layout.addLayout(word_buttons_layout)
        groups_layout.addWidget(self.group_word_not_on)

        # 3.b Right group: "Suggested word options"
        self.group_suggested = QGroupBox("Suggested word options", self)
        suggested_layout = QVBoxLayout(self.group_suggested)
        # List field for suggested words.
        self.suggestions_list = QListWidget(self.group_suggested)
        # For demonstration, add some list items.
        self.suggestions_list.addItems(["Suggestion 1", "Suggestion 2", "Suggestion 3"])
        suggested_layout.addWidget(self.suggestions_list)
        # Row of buttons: Apply and Remember.
        suggestions_buttons_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply", self.group_suggested)
        self.remember_btn = QPushButton("Remember", self.group_suggested)
        suggestions_buttons_layout.addWidget(self.apply_btn)
        suggestions_buttons_layout.addWidget(self.remember_btn)
        suggested_layout.addLayout(suggestions_buttons_layout)
        groups_layout.addWidget(self.group_suggested)

        main_layout.addLayout(groups_layout)

        # 4. Bottom: Abort button.
        self.abort_btn = QPushButton("Abort", self)
        main_layout.addWidget(self.abort_btn, alignment=Qt.AlignRight)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WordEditorWindow()
    window.show()
    sys.exit(app.exec_())