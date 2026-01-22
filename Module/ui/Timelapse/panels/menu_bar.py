import imgui

@staticmethod
def draw_main_menu(app) -> None:
    if imgui.begin_main_menu_bar():  # returns True if the menu bar is open
        if imgui.begin_menu("File", True):
            clicked_quit, _ = imgui.menu_item("Quit", "Ctrl+Q", False, True)
            if clicked_quit:
                app.running = False
            imgui.end_menu()

        if imgui.begin_menu("Configuration", True):
            clicked_cam, _ = imgui.menu_item("Camera Settings")
            clicked_save, _ = imgui.menu_item("Save Config")
            clicked_load, _ = imgui.menu_item("Load Config")

            if clicked_cam:
                print("Open camera settings panel")
            if clicked_save:
                print("Saving configuration...")
            if clicked_load:
                print("Loading configuration...")
            imgui.end_menu()

        if imgui.begin_menu("About", True):
            clicked_info, _ = imgui.menu_item("Info")
            if clicked_info:
                print("Planaria Tracker v1.0")
            imgui.end_menu()

        imgui.end_main_menu_bar()  # only called if begin_main_menu_bar returned True
