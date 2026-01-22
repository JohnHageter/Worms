import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
import OpenGL.GL as GL
import glfw


class TimelapseApp:
    def __init__(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW.")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        width, height = mode.size.width, mode.size.height

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)  
        glfw.window_hint(glfw.DECORATED, glfw.TRUE)  
        glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.FALSE)

        self.window = glfw.create_window(
            int(width*.8), int(height*.8), "Planaria Tracker", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create window.")

        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

        imgui.create_context()

        io = imgui.get_io()
        otf = "resources/Ra-Mono.otf"
        self.ramono_18 = io.fonts.add_font_from_file_ttf(otf, 18)
        self.ramono_16 = io.fonts.add_font_from_file_ttf(otf, 16)

        self.impl = GlfwRenderer(self.window)

        # State variables
        self.running = True
        self.camera_preview = False
        self.camera_fps: float = 10.0
        self.is_acquiring = False
        self.last_frame_time = None
        self.selected_camera: str = "IDS Camera"
        self.camera_types: list[str] = ["IDS Camera", "OpenCV Camera"]

        self.console_logs: list[str] = []

    def run(self) -> None:
        while self.running and not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.impl.process_inputs()

            imgui.new_frame()

            imgui.push_font(self.ramono_18)
            self.draw_main_menu()
            imgui.pop_font()

            imgui.push_font(self.ramono_16)
            self.draw_console()
            self.draw_camera_controls()
            imgui.pop_font()
            # self.draw_camera_controls()
            # self.draw_acquisition_controls()
            # self.draw_status_window()
            # self.draw_quit_panel()

            imgui.render()

            GL.glClearColor(.2,.2,.2,0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)

            self.impl.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)

        self.close()

    def close(self) -> None:
        self.impl.shutdown()
        glfw.terminate()

    def draw_quit_panel(self) -> None:
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(150, 80)
        imgui.begin("Controls", False, imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_BACKGROUND)

        if imgui.button("Quit"):
            self.running = False

        imgui.end()

    def draw_main_menu(self) -> None:
        if imgui.begin_main_menu_bar(): 
            if imgui.begin_menu("File", True):
                clicked_quit, _ = imgui.menu_item("Quit", "Ctrl+Q", False, True)
                if clicked_quit:
                    self.running = False
                imgui.end_menu()

            if imgui.begin_menu("Configuration", True):
                clicked_cam, _ = imgui.menu_item("Camera Settings")
                clicked_save, _ = imgui.menu_item("Save Config")
                clicked_load, _ = imgui.menu_item("Load Config")

                if clicked_cam:
                    self.log("Open camera settings panel")
                if clicked_save:
                    self.log("Saving configuration...")
                if clicked_load:
                    self.log("Loading configuration...")
                imgui.end_menu()

            if imgui.begin_menu("About", True):
                clicked_info, _ = imgui.menu_item("Info")
                if clicked_info:
                    self.log("Planaria Tracker v1.0")
                imgui.end_menu()

            imgui.end_main_menu_bar() 

    def draw_console(self) -> None:
        window_width, window_height = glfw.get_window_size(self.window)
        console_height = 200

        imgui.set_next_window_position(0, window_height - console_height)
        imgui.set_next_window_size(window_width, console_height)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 0)
        imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 1)

        flags = (
            imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_COLLAPSE
        )

        imgui.begin("Console###console", flags=flags)
        imgui.begin_child(
            "ScrollingRegion", window_width - 20, console_height - 20, border=True
        )
        for line in self.console_logs[-1000:]:  
            imgui.text(line)

        if len(self.console_logs) > 0:
            imgui.set_scroll_here_y(1.0)

        imgui.end_child()
        imgui.end()

        imgui.pop_style_var(2)

    def log(self, msg: str) -> None:
        self.console_logs.append(msg)

    def draw_camera_controls(self) -> None:
        imgui.begin("Camera configuration", imgui.WINDOW_NO_MOVE )
        if not hasattr(self, "camera_types") or not self.camera_types:
            self.camera_types = ["IDS Camera", "OpenCV Camera"]

        if not hasattr(self, "selected_camera"):
            self.selected_camera = self.camera_types[0]

        current_index = self.camera_types.index(self.selected_camera)
        changed, new_index = imgui.combo("Camera type", current_index, self.camera_types)

        if changed:
            self.selected_camera = self.camera_types[new_index]
            self.log(f"Camera type set to: {self.selected_camera}")
            
        imgui.end()
