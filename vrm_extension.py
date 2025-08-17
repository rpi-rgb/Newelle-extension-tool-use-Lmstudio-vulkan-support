from .extensions import NewelleExtension
from urllib.parse import urlencode, urljoin
from http.server import HTTPServer, SimpleHTTPRequestHandler
from .handlers.avatar import AvatarHandler
from .handlers.tts import TTSHandler
from .handlers import HandlerDescription, ExtraSettings
import threading
import os
import subprocess
import json
from pydub import AudioSegment
from livepng import LivePNG
from gi.repository import Gtk, WebKit, GLib
from time import sleep
from .utility.strings import rgb_to_hex

class VRMAvatarExtension(NewelleExtension):
    id = "vrmavatar"
    name="VRM Avatar"

    def get_avatar_handlers(self) -> list[dict]:
        return [HandlerDescription("vrm", "VRM Avatar", "3D Avatars in VRM format", VRMHandler)]


class VRMHandler(AvatarHandler):
    key = "vrm"
    _wait_js : threading.Event
    _wait_js2 : threading.Event
    _expressions_raw : list[str]
    _motions_raw : list[str]
    def __init__(self, settings, path: str):
        super().__init__(settings, path)
        self._expressions_raw = []
        self._motions_raw = []
        self._wait_js = threading.Event()
        self._wait_js2 = threading.Event()
        self.webview_path = os.path.join(path, "avatars", "vrm", "web")
        self.models_dir = os.path.join(self.webview_path, "models")
        self.webview = None
        self.httpd = None

    def get_available_models(self):
        file_list = []
        # This can fail if the directory does not exist, e.g. before installation
        if not os.path.isdir(self.models_dir):
            return []
        for root, _, files in os.walk(self.models_dir):
            for file in files:
                if file.endswith('.vrm'):
                    file_name = file.rstrip('.vrm')
                    relative_path = os.path.relpath(os.path.join(root, file), self.models_dir)
                    file_list.append((file_name, relative_path))
        return file_list

    def model_updated(self):
        self.settings_update()

    def get_model(self):
        m = self.get_setting("model", False)
        return "models/model.vrm" if m is None else m

    def get_extra_settings(self) -> list:
        widget = Gtk.Box()
        try:
            color = widget.get_style_context().lookup_color('window_bg_color')[1]
            default = rgb_to_hex(color.red, color.green, color.blue)
        except Exception:
            # Fallback color if GTK context is not available
            default = "#FFFFFF"

        return [
            {
                "key": "model",
                "title": _("VRM Model"),
                "description": _("VRM Model to use"),
                "type": "combo",
                "values": self.get_available_models(),
                "default": "models/model.vrm",
                "folder": os.path.abspath(self.models_dir),
                "refresh": lambda x: self.settings_update(),
                "update_settings": True
            },
            {
             "key": "fps",
                "title": _("Lipsync Framerate"),
                "description": _("Maximum amount of frames to generate for lipsync"),
                "type": "range",
                "min": 5,
                "max": 30,
                "default": 10.0,
                "round-digits": 0
            },
            {
                "key": "background-color",
                "title": _("Background Color"),
                "description": _("Background color of the avatar"),
                "type": "entry",
                "default": default,
            },
        ]

    def is_installed(self) -> bool:
        return os.path.isdir(self.webview_path)

    def install(self):
        """
        Clones the VRM-Web-Viewer repository during installation.
        Handles errors if git is not installed or if the cloning fails.
        """
        try:
            print("Cloning VRM-Web-Viewer repository...")
            subprocess.check_output(
                ["git", "clone", "https://github.com/NyarchLinux/VRM-Web-Viewer", self.webview_path],
                stderr=subprocess.STDOUT
            )
            print("Repository cloned successfully.")
        except FileNotFoundError:
            print("Error: git command not found. Please install git and try again.")
            # Optionally, notify the user through the UI
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.output.decode('utf-8')}")
            print("Please check your internet connection and that the repository URL is accessible.")

    def __start_webserver(self):
        """
        Starts a local HTTP server in a background thread to serve the VRM web viewer files.
        """
        folder_path = self.webview_path
        class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
            def translate_path(self, path):
                path = super().translate_path(path)
                return os.path.join(folder_path, os.path.relpath(path, os.getcwd()))

        # Use a context manager to ensure the server is properly closed
        with HTTPServer(('127.0.0.1', 0), CustomHTTPRequestHandler) as httpd:
            self.httpd = httpd
            model = self.get_setting("model")
            background_color = self.get_setting("background-color")
            scale = int(self.get_setting("scale", False, 100))/100
            q = urlencode({"model": "models/" + model, "bg": background_color, "scale": scale})
            url = urljoin("http://localhost:" + str(httpd.server_address[1]), f"?{q}")

            # Load the URL in the main GTK thread
            GLib.idle_add(self.webview.load_uri, url)

            print(f"HTTP server started at {url}")
            httpd.serve_forever()
            print("HTTP server stopped.")

    def create_gtk_widget(self) -> Gtk.Widget:
        """
        Creates the WebView widget for the avatar.
        Manages the lifecycle of the server and webview to prevent resource leaks.
        """
        # If a server is already running from a previous instance, shut it down.
        if self.httpd:
            threading.Thread(target=self.httpd.shutdown).start()
            self.httpd = None

        self.webview = WebKit.WebView()
        self.webview.connect("destroy", self.destroy)
        self.webview.connect("load-changed", self._on_load_changed)

        # Start the web server in a background thread.
        # It will be shut down when the widget is destroyed.
        server_thread = threading.Thread(target=self.__start_webserver)
        server_thread.daemon = True # Allows the main program to exit even if this thread is running
        server_thread.start()

        self.webview.set_hexpand(True)
        self.webview.set_vexpand(True)
        settings = self.webview.get_settings()
        settings.set_enable_webaudio(True)
        settings.set_media_playback_requires_user_gesture(False)
        self.webview.set_is_muted(False)
        self.webview.set_settings(settings)
        return self.webview

    def _on_load_changed(self, webview, load_event):
        """
        Signal handler for when the WebView's content is loaded.
        """
        if load_event == WebKit.LoadEvent.FINISHED:
            print("WebView content loaded. Fetching initial avatar data.")
            # Fetching data can be slow, so run it in a background thread.
            threading.Thread(target=self._get_initial_data).start()

    def _get_initial_data(self):
        """
        Fetches expressions and motions from the loaded VRM model.
        """
        self.get_expressions()
        self.get_motions()

    def destroy(self, widget=None):
        """
        Called when the webview widget is destroyed by GTK.
        Ensures the HTTP server is shut down.
        """
        if self.httpd:
            threading.Thread(target=self.httpd.shutdown).start()
            self.httpd = None
        self.webview = None
        print("VRMHandler destroyed.")

    def _fetch_from_js(self, js_script, event, result_callback):
        """
        Helper function to robustly fetch data from JavaScript.
        Calls a JS function and waits for a callback.
        """
        if self.webview is None:
            return False

        event.clear()
        # Ensure the UI call happens on the main GTK thread
        GLib.idle_add(self.webview.evaluate_javascript, js_script, -1, None, result_callback, None)

        if not event.wait(5):  # 5-second timeout
            print(f"Error: Timeout waiting for JS function '{js_script}'")
            return False
        return True

    def wait_emotions(self, _, result):
        try:
            value = self.webview.evaluate_javascript_finish(result)
            if value:
                self._expressions_raw = json.loads(value.to_string())
        except Exception as e:
            print(f"Error processing emotions from JS: {e}")
        finally:
            self._wait_js.set()

    def wait_motions(self, _, result):
        try:
            value = self.webview.evaluate_javascript_finish(result)
            if value:
                self._motions_raw = json.loads(value.to_string())
        except Exception as e:
            print(f"Error processing motions from JS: {e}")
        finally:
            self._wait_js2.set()

    def get_expressions_raw(self, allow_webview=True):
        if len(self._expressions_raw) > 0:
            return self._expressions_raw
        if self.webview is None or not allow_webview:
            return self.get_setting(self.get_model() + " expressions", False) or []

        self._expressions_raw = []
        if self._fetch_from_js("get_expressions_json()", self._wait_js, self.wait_emotions):
            self.set_setting(self.get_model() + " expressions", self._expressions_raw)

        return self._expressions_raw

    def get_motions_raw(self, allow_webview=True):
        if len(self._motions_raw) > 0:
            return self._motions_raw
        if self.webview is None or not allow_webview:
            return self.get_setting(self.get_model() + " motions", False) or []

        self._motions_raw = []
        if self._fetch_from_js("get_motions_json()", self._wait_js2, self.wait_motions):
             self.set_setting(self.get_model() + " motions", self._motions_raw)

        return self._motions_raw

    def convert_motion(self, motion: str):
        # This logic can be simplified if we assume names are consistent
        raw_motions = self.get_motions_raw()
        if motion in raw_motions:
            return motion
        # Fallback or mapping logic can be added here if needed
        return None

    def convert_expression(self, expression: str):
        raw_expressions = self.get_expressions_raw()
        if expression in raw_expressions:
            return expression
        return None

    def get_expressions(self) -> list[str]:
        return self.get_expressions_raw()

    def get_motions(self) -> list[str]:
        return self.get_motions_raw()

    def do_motion(self, motion : str):
        motion = self.convert_motion(motion)
        if motion and self.webview:
            script = f"do_motion('{json.dumps(motion)}')"
            GLib.idle_add(self.webview.evaluate_javascript, script, -1, None, None, None)

    def set_expression(self, expression : str):
        exp = self.convert_expression(expression)
        if exp and self.webview:
            script = f"set_expression('{json.dumps(exp)}')"
            GLib.idle_add(self.webview.evaluate_javascript, script, -1, None, None, None)

    def speak(self, path: str, tts: TTSHandler, frame_rate: int):
        # Run the entire speak logic in a background thread to avoid blocking the UI
        thread = threading.Thread(target=self._speak_thread, args=(path, tts, frame_rate))
        thread.daemon = True
        thread.start()

    def _speak_thread(self, path: str, tts: TTSHandler, frame_rate: int):
        try:
            tts.stop()
            audio = AudioSegment.from_file(path)
            sample_rate = audio.frame_rate
            audio_data = audio.get_array_of_samples()
            amplitudes = LivePNG.calculate_amplitudes(sample_rate, audio_data, frame_rate=frame_rate)

            # The animation and sound playing should also be managed carefully
            t1 = threading.Thread(target=self._start_animation, args=(amplitudes, frame_rate))
            t2 = threading.Thread(target=tts.playsound, args=(path, ))
            t1.daemon = True
            t2.daemon = True
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        except Exception as e:
            print(f"Error in speak thread: {e}")

    def _start_animation(self, amplitudes: list[float], frame_rate=10):
        try:
            max_amplitude = max(amplitudes) if amplitudes else 1.0
            for amplitude in amplitudes:
                if self.stop_request:
                    self.set_mouth(0)
                    return
                self.set_mouth(amplitude/max_amplitude)
                sleep(1/frame_rate)
        except Exception as e:
            print(f"Error in lipsync animation: {e}")

    def set_mouth(self, value):
        if self.webview:
            script = f"set_mouth_y({value})"
            GLib.idle_add(self.webview.evaluate_javascript, script, -1, None, None, None)
