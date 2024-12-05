import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import requests
import threading
import socket
import time
import gestureDetection
import xml.etree.ElementTree as ET
import json
import os
import sys
from PIL import Image, ImageTk
import concurrent.futures
import webbrowser
import logging
from tkinter import filedialog
from tooltip import Tooltip
from cv2_enumerate_cameras import enumerate_cameras

class VideoDisplayPanel(tk.Label):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = None

    def update_image(self, frame):
        self.config(text='')
        display_width = 640
        display_height = 360
        frame_resized = cv2.resize(frame, (display_width, display_height))

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.config(image=img)
        self.image = img

    def clear_image(self):
        self.config(image='', text='')
        self.image = None

    def show_load(self):
        self.config(image='', text="Loading...", font=("Helvetica", 24), anchor="center")
        self.image = None

    def show_idle(self):
        self.config(image='', text="No video feed active.", font=("Helvetica", 24), anchor="center")
        self.image = None

class VideoFeed:
    def __init__(self, camera_index=0, frame_callback=None):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.frame_callback = frame_callback
        self.thread = None
        self.cap_lock = threading.Lock()

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_MSMF)
        if not self.cap.isOpened():
            print(f"Failed to open camera {self.camera_index}.")
            return False

        # Set camera resolution to 1920x1080
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()  # Wait for the thread to finish
            self.thread = None
        with self.cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        print("VideoFeed stopped and camera released.")
        if self.frame_callback:
            self.frame_callback = None

    def _update(self):
        while self.running:
            with self.cap_lock:
                if not self.cap or not self.cap.isOpened():
                    break
                try:
                    ret, frame = self.cap.read()
                except Exception as e:
                    print(f"Exception during cap.read(): {e}")
                    break  # Exit the loop if an exception occurs
            if not ret:
                print("Failed to capture a frame.")
                continue

            flipped_frame = cv2.flip(frame, 1)

            if self.frame_callback and self.running:
                self.frame_callback(flipped_frame)

class RokuRemote:
    def __init__(self):
        self.roku_devices = []
        self.selected_roku_device = None
        self.roku_ip = None
        self.roku_mac = None
        self.hand_threads = {}
        self.hand_threads_lock = threading.Lock()
        self.gesture_commands_lock = threading.Lock()
        self.debug_mode = True
        self.skeleton_view = True
        self.auto_start = False
        self.selected_camera_index = 0
        self.video_feed = None
        self.video_active = False
        self.gesture_state = None
        self.last_action = None

        # Initialize Mediapipe Hand Detector
        self.hands = mp.solutions.hands.Hands(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.2,
            max_num_hands=1
        )
        self.mp_hands_lock = threading.Lock()

        self.root = None
        self.left_frame = None
        self.right_frame = None
        self.video_display_panel = None
        self.control_buttons = []

        # Settings and logging directories
        self.path_file_path = self.get_path_file_path()
        self.settings_directory = self.load_settings_directory()
        self.log_file_path = None
        self.config_file_path = None

        self.load_config()
        self.setup_logging()

        self.hand_absent_times = {}
        self.hand_absent_timeout = 0.5

    def get_app_data_dir(self):
        """Returns the path to the application's default data directory, platform-specific."""
        app_name = 'HandiRokuRemote'

        if sys.platform.startswith('win'):
            # On Windows, use %APPDATA%
            appdata = os.getenv('APPDATA')
            if appdata:
                return os.path.join(appdata, app_name)
            else:
                # Fallback to user's home directory
                return os.path.join(os.path.expanduser('~'), app_name)
        elif sys.platform == 'darwin':
            # On macOS, use ~/Library/Application Support/
            return os.path.join(os.path.expanduser('~/Library/Application Support'), app_name)
        else:
            # On Linux and other Unix-like systems, use ~/.local/share/
            return os.path.join(os.path.expanduser('~/.local/share'), app_name)

    def get_path_file_path(self):
        """Returns the path to the 'roku_remote_path.txt' file."""
        app_data_dir = self.get_app_data_dir()
        if not os.path.exists(app_data_dir):
            os.makedirs(app_data_dir)
        return os.path.join(app_data_dir, 'roku_remote_path.txt')

    def load_settings_directory(self):
        """Load the settings directory from 'roku_remote_path.txt' or set default."""
        path_file = self.get_path_file_path()
        if os.path.exists(path_file):
            with open(path_file, 'r') as f:
                settings_dir = f.read().strip()
                if os.path.exists(settings_dir):
                    return settings_dir
        # If the path file doesn't exist or the directory doesn't exist, use default
        default_settings_dir = self.get_app_data_dir()
        if not os.path.exists(default_settings_dir):
            os.makedirs(default_settings_dir)
        # Write the default path to the path file
        with open(path_file, 'w') as f:
            f.write(default_settings_dir)
        return default_settings_dir

    def save_settings_directory(self):
        """Save the settings directory to 'roku_remote_path.txt'."""
        path_file = self.get_path_file_path()
        with open(path_file, 'w') as f:
            f.write(self.settings_directory)

    def setup_logging(self):
        """Set up logging based on the settings directory."""
        if not os.path.exists(self.settings_directory):
            os.makedirs(self.settings_directory)

        self.log_file_path = os.path.join(self.settings_directory, 'roku_remote.log')

        # Remove all handlers associated with the root logger object
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=self.log_file_path,
            level=logging.DEBUG if self.debug_mode else logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='a'
        )
        logging.debug("Logging initialized.")

    def load_config(self):
        """Load the settings from a configuration file."""
        # First, check if settings directory exists
        self.config_file_path = os.path.join(self.settings_directory, 'roku_config.json')

        if os.path.exists(self.config_file_path):
            with open(self.config_file_path, 'r') as f:
                data = json.load(f)
                self.roku_ip = data.get('roku_ip')
                self.roku_mac = data.get('roku_mac')
                self.selected_camera_index = data.get('selected_camera_index', 0)
                self.auto_start = data.get('auto_start', False)
                self.debug_mode = data.get('debug_mode', False)
                self.skeleton_view = data.get('skeleton_view', False)
                self.known_devices = data.get('known_devices', [])
                self.last_selected_device_ip = data.get('last_selected_device_ip')
                print(f"Loaded configuration from file: IP={self.roku_ip}, MAC={self.roku_mac}, "
                      f"Camera Index={self.selected_camera_index}, Auto Start={self.auto_start}, "
                      f"Settings Directory={self.settings_directory}")
        else:
            self.known_devices = []
            self.last_selected_device_ip = None

    def save_config(self):
        """Save the settings to a configuration file."""
        data = {
            'roku_ip': self.roku_ip,
            'roku_mac': self.roku_mac,
            'selected_camera_index': self.selected_camera_index,
            'auto_start': self.auto_start,
            'debug_mode': self.debug_mode,
            'skeleton_view': self.skeleton_view,
            'known_devices': self.known_devices,
            'last_selected_device_ip': self.selected_roku_device['ip'] if self.selected_roku_device else None
        }

        if not os.path.exists(self.settings_directory):
            os.makedirs(self.settings_directory)

        with open(self.config_file_path, 'w') as f:
            json.dump(data, f)
        print(f"Saved configuration to file: IP={self.roku_ip}, MAC={self.roku_mac}, "
              f"Camera Index={self.selected_camera_index}, Auto Start={self.auto_start}, "
              f"Settings Directory={self.settings_directory}")
        logging.debug(f"Configuration saved to {self.config_file_path}")

    def change_settings_directory(self):
        # Not used for now
        """Allows user to select a new settings directory."""
        new_directory = filedialog.askdirectory(initialdir=self.settings_directory, title="Select Settings Directory")
        if new_directory:
            old_settings_directory = self.settings_directory
            self.settings_directory = new_directory

            # Move existing config and log files to new directory
            old_config_file = os.path.join(old_settings_directory, 'roku_config.json')
            old_log_file = os.path.join(old_settings_directory, 'roku_remote.log')

            new_config_file = os.path.join(self.settings_directory, 'roku_config.json')
            new_log_file = os.path.join(self.settings_directory, 'roku_remote.log')

            if not os.path.exists(self.settings_directory):
                os.makedirs(self.settings_directory)

            if os.path.exists(old_config_file):
                os.rename(old_config_file, new_config_file)
                logging.debug(f"Moved config file from {old_config_file} to {new_config_file}")
            if os.path.exists(old_log_file):
                os.rename(old_log_file, new_log_file)
                logging.debug(f"Moved log file from {old_log_file} to {new_log_file}")

            self.config_file_path = new_config_file
            self.save_settings_directory()  # Update the path file
            self.setup_logging()  # Re-setup logging with new directory
            self.settings_directory_entry.delete(0, tk.END)
            self.settings_directory_entry.insert(0, self.settings_directory)
            print(f"Settings directory changed to: {self.settings_directory}")
            logging.debug(f"Settings directory changed to: {self.settings_directory}")
            self.save_config()  # Save the new settings

    def attempt_connect_known_devices(self):
        """Attempt to connect to the last selected device or known devices from the config."""
        self.roku_devices = []
        if hasattr(self, 'last_selected_device_ip') and self.last_selected_device_ip:
            print(f"Attempting to connect to the last selected device at {self.last_selected_device_ip}...")
            logging.debug(f"Attempting to connect to last selected device at {self.last_selected_device_ip}")
            device_info = self.get_roku_device_info(self.last_selected_device_ip)
            if device_info:
                self.selected_roku_device = device_info
                self.roku_ip = device_info['ip']
                self.roku_mac = device_info.get('wifi_mac') or device_info.get('ethernet_mac')
                self.add_to_known_devices(device_info)
                self.roku_devices.append(device_info)
                print(f"Connected to last selected Roku device at {self.roku_ip}")
                logging.debug(f"Connected to last selected Roku device at {self.roku_ip}")
                self.save_config()
            else:
                print(f"Last selected device at {self.last_selected_device_ip} is not available.")
                logging.debug(f"Last selected device at {self.last_selected_device_ip} is not available.")

        # Attempt to connect to known devices
        if self.known_devices:
            print("Attempting to connect to known devices...")
            logging.debug("Attempting to connect to known devices...")
            for device_info in self.known_devices:
                ip = device_info['ip']
                device_info = self.get_roku_device_info(ip)
                if device_info:
                    if device_info not in self.roku_devices:
                        self.roku_devices.append(device_info)
                    print(f"Found known Roku device at {ip}")
                    logging.debug(f"Found known Roku device at {ip}")
            if self.roku_devices:
                if not self.selected_roku_device:
                    self.selected_roku_device = self.roku_devices[0]
                    self.roku_ip = self.selected_roku_device['ip']
                    self.roku_mac = self.selected_roku_device.get('wifi_mac') or self.selected_roku_device.get('ethernet_mac')
                    self.save_config()
            else:
                print("Known devices not found, starting network scan.")
                logging.debug("Known devices not found, starting network scan.")
                self.refresh_device_list()
        else:
            self.refresh_device_list()

        self.update_device_combo()

    def discover_roku_devices(self, subnet='192.168.1', start=1, end=254):
        """Scan the local network for all Roku devices."""
        print(f"Scanning network {subnet}.0/{end - start + 1} for Roku devices...")
        logging.debug(f"Scanning network {subnet}.0/{end - start + 1} for Roku devices...")
        devices = []
        ips = [f'{subnet}.{i}' for i in range(start, end + 1)]

        def scan_ip(ip):
            url = f'http://{ip}:8060/query/device-info'
            try:
                response = requests.get(url, timeout=0.25)
                if response.status_code == 200 and 'roku' in response.text.lower():
                    device_info = self.get_roku_device_info(ip)
                    if device_info:
                        device_name = device_info['user_device_name'] or device_info['friendly_device_name'] or device_info['default_device_name']
                        print(f"Found Roku device at {ip} with name '{device_name}'")
                        logging.debug(f"Found Roku device at {ip} with name '{device_name}'")
                        return device_info
            except requests.exceptions.RequestException as e:
                pass
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(scan_ip, ip) for ip in ips]
            for future in concurrent.futures.as_completed(futures):
                device_info = future.result()
                if device_info:
                    devices.append(device_info)
        if not devices:
            print("Failed to find Roku devices via network scan.")
            logging.debug("Failed to find Roku devices via network scan.")
        return devices

    def get_roku_device_info(self, ip_address):
        """Retrieve device info from the Roku device."""
        url = f'http://{ip_address}:8060/query/device-info'
        try:
            response = requests.get(url, timeout=0.5)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                device_info = {
                    'ip': ip_address,
                    'serial_number': root.findtext('serial-number'),
                    'device_id': root.findtext('device-id'),
                    'vendor_name': root.findtext('vendor-name'),
                    'model_name': root.findtext('model-name'),
                    'friendly_device_name': root.findtext('friendly-device-name'),
                    'friendly_model_name': root.findtext('friendly-model-name'),
                    'default_device_name': root.findtext('default-device-name'),
                    'user_device_name': root.findtext('user-device-name'),
                    'user_device_location': root.findtext('user-device-location'),
                    'wifi_mac': root.findtext('wifi-mac'),
                    'ethernet_mac': root.findtext('ethernet-mac'),
                }
                return device_info
        except requests.exceptions.RequestException:
            pass
        return None

    def wake_on_lan(self):
        """Send a Wake-on-LAN magic packet to the Roku device."""
        if not self.roku_mac:
            print("MAC address not known. Cannot send Wake-on-LAN packet.")
            logging.debug("MAC address not known. Cannot send Wake-on-LAN packet.")
            return
        mac_address = self.roku_mac.replace(':', '').replace('-', '').replace('.', '')
        if len(mac_address) != 12:
            print("Invalid MAC address format.")
            logging.debug("Invalid MAC address format.")
            return
        magic_packet = bytes.fromhex('FF' * 6 + mac_address * 16)
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.sendto(magic_packet, ('<broadcast>', 9))
            sock.close()
            print("Sent Wake-on-LAN packet.")
            logging.debug("Sent Wake-on-LAN packet.")
        except Exception as e:
            print(f"Failed to send Wake-on-LAN packet: {e}")
            logging.debug(f"Failed to send Wake-on-LAN packet: {e}")

    def send_command(self, command):
        """
        Send a command to the Roku device via HTTP POST request.
        """
        if not self.roku_ip:
            print("No Roku device IP address known.")
            logging.debug("No Roku device IP address known.")
            if self.roku_mac:
                print("Attempting to wake up Roku device via Wake-on-LAN...")
                logging.debug("Attempting to wake up Roku device via Wake-on-LAN...")
                self.wake_on_lan()
                time.sleep(1)
                self.roku_devices = self.discover_roku_devices()
                for device in self.roku_devices:
                    device_mac = device.get('wifi_mac') or device.get('ethernet_mac')
                    if device_mac and device_mac.lower() == self.roku_mac.lower():
                        self.selected_roku_device = device
                        self.roku_ip = device['ip']
                        print(f"Reconnected to Roku device at {self.roku_ip}")
                        logging.debug(f"Reconnected to Roku device at {self.roku_ip}")
                        self.save_config()
                        break
                if not self.roku_ip:
                    print("Unable to discover Roku device after Wake-on-LAN.")
                    logging.debug("Unable to discover Roku device after Wake-on-LAN.")
                    return
            else:
                print("Cannot wake up Roku device because MAC address is unknown.")
                logging.debug("Cannot wake up Roku device because MAC address is unknown.")
                return

        url = f"http://{self.roku_ip}:8060/keypress/{command}"
        try:
            response = requests.post(url, timeout=0.25)
            if response.status_code == 200:
                print(f"Sent command '{command}' to Roku at {self.roku_ip}")
                logging.debug(f"Sent command '{command}' to Roku at {self.roku_ip}")
            elif response.status_code == 202:
                print(f"Command '{command}' accepted by Roku at {self.roku_ip}. Waiting for transition...")
                logging.debug(f"Command '{command}' accepted by Roku at {self.roku_ip}. Waiting for transition...")
                time.sleep(1)  # Allow Roku to complete power transition
            elif response.status_code == 403:
                print(f"Command '{command}' is not supported on this device.")
                logging.debug(f"Command '{command}' is not supported on this device.")
            else:
                print(f"Failed to send command '{command}' to Roku at {self.roku_ip}, status code: {response.status_code}")
                logging.debug(f"Failed to send command '{command}' to Roku at {self.roku_ip}, status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending command '{command}' to Roku at {self.roku_ip}: {e}")
            logging.debug(f"Error sending command '{command}' to Roku at {self.roku_ip}: {e}")

    def start_video_feed(self):
        """Starts the video feed using the selected camera index."""
        if self.video_feed:
            self.stop_video_feed()

        self.video_display_panel.show_load()
        self.root.update_idletasks()
        threading.Thread(target=self.initialize_video_feed, daemon=True).start()

    def initialize_video_feed(self):
        """Initializes the video feed after the GUI has updated."""
        try:
            camera_selection = self.camera_combo.get()
            camera_index = int(camera_selection.split("(")[-1].rstrip(")"))
            print(f"Attempting to start video feed with camera index {camera_index}")
            logging.debug(f"Attempting to start video feed with camera index {camera_index}")
            self.video_feed = VideoFeed(camera_index=camera_index, frame_callback=self.process_frame)
            success = self.video_feed.start()
            if not success:
                raise Exception("Failed to start video feed.")
            print("Video feed started.")
            logging.debug("Video feed started.")
            self.video_active = True
            self.root.after(0, lambda: self.start_button.config(text="Stop Video Feed", command=self.stop_video_feed))
            self.root.after(0, lambda: self.camera_combo.config(state='disabled'))
            self.root.after(0, self.update_start_button_state)
            self.root.after(0, self.root.geometry, "")
        except Exception as e:
            print(f"Error starting video feed: {e}")
            logging.debug(f"Error starting video feed: {e}")
            self.video_active = False
            self.video_feed = None
            self.root.after(0, self.video_display_panel.show_idle)
            self.root.after(0, lambda: self.start_button.config(text="Start Video Feed", command=self.start_video_feed))
            self.root.after(0, lambda: self.camera_combo.config(state='normal'))
            self.root.after(0, self.update_start_button_state())

    def stop_video_feed(self):
        """Stops the video feed."""
        if self.video_active:
            self.gesture_state = None
            self.last_action = None
            self.video_active = False
            if self.video_feed:
                try:
                    self.video_feed.stop()
                    self.video_feed = None
                    print("Camera feed stopped.")
                    logging.debug("Camera feed stopped.")
                except Exception as e:
                    print(f"Error stopping video feed: {e}")
                    logging.debug(f"Error stopping video feed: {e}")

            # Stop all HandClassifier threads
            threads_to_join = []
            with self.hand_threads_lock:
                for hand_thread in self.hand_threads.values():
                    hand_thread.stop()
                    threads_to_join.append(hand_thread)
                self.hand_threads.clear()
                self.hand_absent_times.clear()
            # Now, outside the lock, join the threads
            for hand_thread in threads_to_join:
                hand_thread.join(timeout=1)  # Wait for the thread to finish

            print("All HandClassifier threads stopped.")
            logging.debug("All HandClassifier threads stopped.")

            self.video_display_panel.show_idle()
            self.start_button.config(text="Start Video Feed", command=self.start_video_feed)
            self.camera_combo.config(state='normal')
            self.update_start_button_state()
            self.root.after(0, self.root.geometry, "")

    def process_frame(self, frame):
        """Handles incoming frames and schedules GUI updates in the main thread."""
        if not self.video_active:
            return
        if frame is None:
            print("Received empty frame.")
            logging.debug("Received empty frame.")
            return
        self.root.after_idle(self._process_frame_in_main_thread, frame)

    def _process_frame_in_main_thread(self, frame):
        """Processes the frame and updates the GUI in the main thread."""
        if not self.video_active:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape
        with self.mp_hands_lock:
            result = self.hands.process(frame_rgb)
        hand_landmarks_list = result.multi_hand_landmarks
        handedness_list = result.multi_handedness

        current_hand_ids = set()
        debug_strings = []

        if hand_landmarks_list and handedness_list:
            for idx, (hand_landmarks, handedness) in enumerate(zip(hand_landmarks_list, handedness_list)):
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                current_hand_ids.add(hand_label)

                # Draw hand landmarks on the frame
                if self.skeleton_view:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # Update or create a thread for this hand
                with self.hand_threads_lock:
                    if hand_label in self.hand_threads:
                        # Update landmarks
                        self.hand_threads[hand_label].update_landmarks(hand_landmarks, hand_label)
                        # Reset absent time since hand is detected
                        self.hand_absent_times[hand_label] = None
                    else:
                        # Start new HandClassifier thread
                        hand_thread = gestureDetection.HandClassifier(
                            hand_id=hand_label,
                            initial_landmarks=hand_landmarks,
                            command_callback=self.gesture_command_callback,
                            hand_label=hand_label
                        )
                        hand_thread.start()
                        self.hand_threads[hand_label] = hand_thread
                        self.hand_absent_times[hand_label] = None  # Initialize absent time

                # If debug mode is on, collect debug information
                if self.debug_mode:
                    with self.hand_threads_lock:
                        hand_thread = self.hand_threads.get(hand_label)
                    if hand_thread:
                        debug_string = hand_thread.debug_string
                        debug_strings.append(debug_string)

                        # Overlay the debug information on the frame
                        x, y = 10, 30 + idx * 80
                        for i, line in enumerate(debug_string.split('\n')):
                            cv2.putText(frame, line, (x, y + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                        # Draw the red line between thumb point and selected other point
                        if hand_thread.line_points:
                            thumb_id, other_id = hand_thread.line_points
                            thumb_landmark = hand_landmarks.landmark[thumb_id]
                            other_landmark = hand_landmarks.landmark[other_id]

                            thumb_x = int(thumb_landmark.x * image_width)
                            thumb_y = int(thumb_landmark.y * image_height)
                            other_x = int(other_landmark.x * image_width)
                            other_y = int(other_landmark.y * image_height)

                            # Draw red line
                            cv2.line(frame, (thumb_x, thumb_y), (other_x, other_y), (0, 0, 255), 2)

        else:
            # If no hands are detected, set gesture state to None
            self.gesture_state = None
            self.last_action = None

        # Display gesture state in the top-right corner
        gesture_text = self.gesture_state if self.gesture_state else "Idle"
        font_scale = 1.8  # Scale for gesture state text
        thickness = 4  # Thickness for better visibility

        # Get text size for gesture state
        text_size = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = max(10, image_width - text_size[0] - 10)  # Align gesture state to the right
        text_y = min(image_height - 10, text_size[1] + 30)  # Top padding

        # Render the gesture state text
        cv2.putText(frame, gesture_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        # Display last action below the gesture state
        if self.last_action:
            last_action_text = f"{self.last_action}"
            last_action_font_scale = 1.2  # Twice as large as before
            last_action_thickness = 3  # Thicker for better visibility

            # Get text size for last action
            last_action_text_size = cv2.getTextSize(last_action_text, cv2.FONT_HERSHEY_SIMPLEX, last_action_font_scale, last_action_thickness)[0]
            # Align last action text to the right, below the gesture state text
            last_action_text_x = max(10, image_width - last_action_text_size[0] - 10)
            last_action_text_y = text_y + text_size[1] + 20  # Add padding below the gesture text

            # Render the last action text in green, aligned to the right
            cv2.putText(frame, last_action_text, (last_action_text_x, last_action_text_y), cv2.FONT_HERSHEY_SIMPLEX, last_action_font_scale, (0, 255, 0), last_action_thickness)


        # Remove threads for hands that are no longer detected, after a grace period
        with self.hand_threads_lock:
            all_hand_ids = set(self.hand_threads.keys())
            lost_hands = all_hand_ids - current_hand_ids
            for hand_id in lost_hands:
                if self.hand_absent_times.get(hand_id) is None:
                    self.hand_absent_times[hand_id] = time.time()
                else:
                    elapsed_time = time.time() - self.hand_absent_times[hand_id]
                    if elapsed_time > self.hand_absent_timeout:
                        self.gesture_state = None
                        self.last_action = None
                        self.hand_threads[hand_id].stop()
                        del self.hand_threads[hand_id]
                        del self.hand_absent_times[hand_id]

            for hand_id in current_hand_ids:
                if hand_id in self.hand_absent_times:
                    self.hand_absent_times[hand_id] = None

        if self.video_display_panel:
            self.video_display_panel.update_image(frame)

    def gesture_command_callback(self, gesture, debug_string, direction, gesture_state):
        if not self.video_active:
            return
        self.root.after(0, self.handle_gesture, gesture, gesture_state)

    def handle_gesture(self, gesture, gesture_state):
        if not self.video_active:
            return  # Do not handle gestures if video feed is stopped
        try:
            with self.mp_hands_lock:
                self.gesture_state = gesture_state

            # Map gestures to Roku commands
            gesture_to_command = {
                'Play': 'Play',
                'Select': 'Select',
                'Up': 'Up',
                'Down': 'Down',
                'Left': 'Left',
                'Right': 'Right',
                "Power": "Power",
                "VolumeUp": "VolumeUp",
                "VolumeDown": "VolumeDown",
                "Back": "Back",
                "Home": "Home",
                "VolumeMute": "VolumeMute",
                "Rev": 'Rev',
                "Fwd": 'Fwd',
            }
            if gesture in gesture_to_command:
                command = gesture_to_command[gesture]
                # Start a new thread to send the command
                threading.Thread(target=self.send_command, args=(command,), daemon=True).start()
                self.last_action = command
                logging.debug(f"Gesture '{gesture}' recognized, command '{command}' sent.")
            if gesture == 'Fist':
                if self.last_action != 'Fist':
                    self.last_action = 'Fist'
            if gesture == 'End Tracking':
                if self.last_action != 'End Tracking':
                    self.last_action = 'End Tracking'
            if self.gesture_state == "Active" and self.last_action == "End Tracking":
                self.last_action = None
            
        except Exception as e:
            print(f"Error in handle_gesture: {e}")
            logging.debug(f"Error in handle_gesture: {e}")

    def toggle_debug_mode(self):
        """Toggle the debug mode."""
        self.debug_mode = self.debug_var.get()
        self.setup_logging()  # Re-setup logging with new debug mode
        self.save_config()
        print(f"Debug mode set to {self.debug_mode}")
        logging.debug(f"Debug mode set to {self.debug_mode}")

    def toggle_skeleton_view(self):
        """Toggle the debug mode."""
        self.skeleton_view = self.skeleton_var.get()
        self.setup_logging()  # Re-setup logging with new debug mode
        self.save_config()
        print(f"Skeleton view set to {self.skeleton_view}")
        logging.debug(f"Skeleton view set to {self.skeleton_view}")

    def toggle_auto_start(self):
        """Toggle the auto start setting."""
        self.auto_start = self.auto_start_var.get()
        self.save_config()
        print(f"Auto start set to {self.auto_start}")
        logging.debug(f"Auto start set to {self.auto_start}")

    def on_camera_selected(self, event):
        """Update selected camera index when camera selection changes."""
        selected_camera = self.camera_combo.get()
        if "(" in selected_camera and selected_camera.endswith(")"):
            self.selected_camera_index = int(selected_camera.split("(")[-1].rstrip(")"))
            self.save_config()
            print(f"Selected camera index set to {self.selected_camera_index}")
            logging.debug(f"Selected camera index set to {self.selected_camera_index}")
        self.update_start_button_state()

    def on_device_selected(self, event):
        """Handle device selection from dropdown."""
        selected_device_name = self.device_combo.get()
        for device in self.roku_devices:
            device_name = device['user_device_name'] or device['friendly_device_name'] or device['default_device_name']
            device_location = device.get('user_device_location', '')
            if device_location:
                device_name += f" - {device_location}"
            display_name = f"{device_name} ({device['ip']})"
            if display_name == selected_device_name:
                self.selected_roku_device = device
                self.roku_ip = device['ip']
                self.roku_mac = device.get('wifi_mac') or device.get('ethernet_mac')
                self.add_to_known_devices(device)
                print(f"Selected Roku device: {device_name} at {self.roku_ip}")
                logging.debug(f"Selected Roku device: {device_name} at {self.roku_ip}")
                self.save_config()
                break
        self.update_control_buttons_state()
        self.update_start_button_state()

    def add_to_known_devices(self, device):
        """Add or update a device in the known devices list."""
        for known_device in self.known_devices:
            if known_device['ip'] == device['ip']:
                known_device.update(device)
                break
        else:
            self.known_devices.append(device)
        self.save_config()

    def on_connect_manual_ip(self):
        """Handle manual IP input and connect button."""
        manual_ip = self.manual_ip_entry.get().strip()
        if manual_ip:
            device_info = self.get_roku_device_info(manual_ip)
            if device_info:
                self.selected_roku_device = device_info
                self.roku_ip = device_info['ip']
                self.roku_mac = device_info.get('wifi_mac') or device_info.get('ethernet_mac')
                self.add_to_known_devices(device_info)
                self.roku_devices.append(device_info)
                print(f"Connected to Roku device: {device_info['user_device_name']} at {self.roku_ip}")
                logging.debug(f"Connected to Roku device: {device_info['user_device_name']} at {self.roku_ip}")
                self.update_device_combo()
                self.manual_ip_entry.delete(0, tk.END)
            else:
                print(f"Failed to connect to Roku at {manual_ip}.")
                logging.debug(f"Failed to connect to Roku at {manual_ip}.")
        else:
            print("Please enter a valid IP address.")
            logging.debug("No IP address entered in manual IP field.")

    def refresh_device_list(self):
        """Refresh the list of Roku devices."""
        def _discover_and_update():
            self.roku_devices = self.discover_roku_devices()
            for device in self.roku_devices:
                self.add_to_known_devices(device)
            self.root.after(0, self.update_device_combo)
        threading.Thread(target=_discover_and_update).start()
        self.device_combo['values'] = ["Loading devices..."]
        self.device_combo.set("Loading devices...")
        logging.debug("Started thread to refresh device list.")

    def update_device_combo(self):
        device_names = []
        for device in self.roku_devices:
            device_name = device['user_device_name'] or device['friendly_device_name'] or device['default_device_name']
            device_location = device.get('user_device_location', '')
            if device_location:
                device_name += f" - {device_location}"
            display_name = f"{device_name} ({device['ip']})"
            device_names.append(display_name)
        if device_names:
            self.device_combo['values'] = device_names
            self.device_combo.config(width=max(len(name) for name in device_names))
            if self.selected_roku_device:
                selected_name = self.selected_roku_device['user_device_name'] or self.selected_roku_device['friendly_device_name'] or self.selected_roku_device['default_device_name']
                selected_location = self.selected_roku_device.get('user_device_location', '')
                if selected_location:
                    selected_name += f" - {selected_location}"
                selected_display_name = f"{selected_name} ({self.selected_roku_device['ip']})"
                if selected_display_name in device_names:
                    self.device_combo.set(selected_display_name)
                else:
                    self.device_combo.current(0)
                    self.on_device_selected(None)
            else:
                self.device_combo.current(0)
                self.on_device_selected(None)
        else:
            self.device_combo['values'] = []
            self.device_combo.set("No devices found")
            logging.debug("No devices found after updating device combo.")
        self.update_control_buttons_state()
        self.update_start_button_state()

    def refresh_camera_list(self):
        """Refreshes the camera list by detecting available cameras using enumerate_cameras."""
        print("Refreshing camera list")
        logging.debug("Refreshing camera list")
        available_cameras = []
        try:
            for camera_info in enumerate_cameras(cv2.CAP_MSMF):
                available_cameras.append(f"{camera_info.name} ({camera_info.index})")

            if available_cameras:
                self.camera_combo['values'] = available_cameras
                for idx, cam in enumerate(available_cameras):
                    if f"({self.selected_camera_index})" in cam:
                        self.camera_combo.current(idx)
                        break
                else:
                    self.camera_combo.current(0)
                self.on_camera_selected(None)
            else:
                self.camera_combo['values'] = ["No cameras found"]
                self.camera_combo.set("No cameras found")
                logging.debug("No cameras found during camera refresh.")
        except Exception as e:
            print(f"Error while refreshing camera list: {e}")
            logging.debug(f"Error while refreshing camera list: {e}")
            self.camera_combo['values'] = ["Error detecting cameras"]
            self.camera_combo.set("Error detecting cameras")
        self.update_start_button_state()

    def update_start_button_state(self):
        """Enable or disable the start video feed button based on selections."""
        if hasattr(self, 'start_button'):  # Ensure the button is created
            if self.video_active:
                self.start_button.config(state='normal')
            elif self.device_combo.get() and self.camera_combo.get():
                self.start_button.config(state='normal')
            else:
                self.start_button.config(state='disabled')

    def update_control_buttons_state(self):
        """Enable or disable control buttons based on device selection."""
        if hasattr(self, 'control_buttons'):  # Ensure control buttons are created
            if self.selected_roku_device:
                for btn in self.control_buttons:
                    btn.config(state='normal')
            else:
                for btn in self.control_buttons:
                    btn.config(state='disabled')

    def setup_gui(self):
        """Setup the GUI for starting and stopping the video feed and controlling Roku."""
        self.root = tk.Tk()
        self.root.title("HandiRokuRemote")
        self.root.geometry("800x720")
        self.root.resizable(True, True)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, "images", "icon.ico")
        if os.path.exists(icon_path):
            self.root.iconbitmap(icon_path)

        # Main frame to hold left and right frames
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True)

        # Left frame for controls
        self.left_frame = tk.Frame(main_frame, width=200)
        self.left_frame.pack(side='left', fill='y', padx=10, pady=10)

        # Right frame for video feed
        self.right_frame = tk.Frame(main_frame)
        self.right_frame.pack(side='right', fill='both', expand=True)

        # Start/Stop Video Button
        self.start_button = tk.Button(self.left_frame, text="Start Video Feed", command=self.start_video_feed, width=20, height=2)
        self.start_button.pack(pady=10)
        self.start_button.config(state='disabled')  # Initially disabled

        # Auto Start and Debug Mode Checkboxes on the same row
        checkbox_frame = tk.Frame(self.left_frame)
        checkbox_frame.pack(pady=5)

        # Instructions Button
        instructions_button = tk.Button(
            checkbox_frame, text="Instructions", command=lambda: webbrowser.open("https://github.com/BBelk/HandiRokuRemote?tab=readme-ov-file#installation-and-interface-overview")
        )
        instructions_button.grid(row=0, column=0, padx=10)

        # Auto Start Checkbox
        self.auto_start_var = tk.BooleanVar(value=self.auto_start)
        auto_start_checkbox = tk.Checkbutton(
            checkbox_frame, text="Auto Start", variable=self.auto_start_var, command=self.toggle_auto_start)
        auto_start_checkbox.grid(row=1, column=0, padx=10)

        # Debug Mode Checkbox
        self.debug_var = tk.BooleanVar(value=self.debug_mode)
        debug_checkbox = tk.Checkbutton(
            checkbox_frame, text="Debug Mode", variable=self.debug_var, command=self.toggle_debug_mode)
        debug_checkbox.grid(row=0, column=1, padx=10)

        # Skeleton View Checkbox
        self.skeleton_var = tk.BooleanVar(value=self.skeleton_view)
        skeleton_checkbox = tk.Checkbutton(
            checkbox_frame, text="Skeleton View", variable=self.skeleton_var, command=self.toggle_skeleton_view)
        skeleton_checkbox.grid(row=1, column=1, padx=10)

        
        # Device Selection Frame
        device_frame = tk.LabelFrame(self.left_frame, text="Device Selection")
        device_frame.pack(fill='x', pady=2)

        # Device dropdown and manual IP entry
        self.device_combo = ttk.Combobox(device_frame, state="readonly", width=40)  # Set width for device dropdown
        self.device_combo.pack(pady=5)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_selected)

        # Manual IP Entry
        manual_frame = tk.Frame(device_frame)
        manual_frame.pack(fill='x', pady=2)
        manual_ip_label = tk.Label(manual_frame, text="Manual IP:")
        manual_ip_label.pack(side='left', padx=5)
        self.manual_ip_entry = tk.Entry(manual_frame)
        self.manual_ip_entry.pack(side='left', fill='x', expand=True, padx=5)
        connect_button = tk.Button(manual_frame, text="Connect", command=self.on_connect_manual_ip)
        connect_button.pack(side='left', padx=5)

        # Refresh Devices Button
        refresh_devices_button = tk.Button(device_frame, text="Refresh Devices", command=self.refresh_device_list)
        refresh_devices_button.pack(pady=2)

        # Camera Selection Frame
        camera_frame = tk.LabelFrame(self.left_frame, text="Camera Selection")
        camera_frame.pack(fill='x', pady=2)

        # Camera selection Combobox
        self.camera_combo = ttk.Combobox(camera_frame, state="readonly", width=40)  # Match width to device dropdown
        self.camera_combo.pack(pady=5)
        self.refresh_camera_list()  # Populate the camera list
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Button to refresh camera list
        refresh_cameras_button = tk.Button(camera_frame, text="Refresh Cameras", command=self.refresh_camera_list)
        refresh_cameras_button.pack(pady=2)

        settings_frame = tk.LabelFrame(self.left_frame, text="Settings Directory")
        settings_frame.pack(fill='x', pady=2)

        # Read-only Text widget to display the directory with wrapping and highlighting support
        self.settings_directory_display = tk.Text(
            settings_frame,
            height=2,  # Adjust height as needed
            wrap='word',  # Wrap text within the widget
            state='normal',  # Allow changes to content
            bg=self.root.cget('bg'),  # Match the background color of the root
            relief='flat',
            width=15,  # Set a fixed width to prevent expansion
            padx=5,
            pady=5
        )
        self.settings_directory_display.insert('1.0', self.settings_directory)  # Insert the directory text
        self.settings_directory_display.config(state='disabled')  # Make it read-only
        self.settings_directory_display.pack(fill='x', padx=5, pady=5)



        def resource_path(relative_path):
            """ Get absolute path to resource, works for dev and for PyInstaller """
            base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
            return os.path.join(base_path, relative_path)

        # Define the base path for the tooltip images
        base_image_path = resource_path(os.path.join('images', 'tooltip-images'))

        # Roku Control Buttons
        control_frame = tk.Frame(self.left_frame)
        control_frame.pack(pady=10, fill='x')

        # Add spacing columns to align content equidistantly
        control_frame.grid_columnconfigure(0, weight=1)  # Spacer column on the left
        control_frame.grid_columnconfigure(3, weight=1)  # Spacer column on the right

        # Create a frame specifically for centering the buttons
        center_frame = tk.Frame(control_frame)
        center_frame.grid(row=0, column=0, columnspan=3, pady=5)  # Span across 3 columns to center

        # Activate/Deactivate button
        activate_button = tk.Button(
            center_frame, text="🤞 Activate/Deactivate", state="disabled", width=17, height=1
        )
        activate_button.grid(row=0, column=0, padx=10)  # Add padding for spacing
        Tooltip(activate_button, image_path=os.path.join(base_image_path, "fingers-crossed.jpg"))

        # Middle button (currently commented out)
        # power_button = tk.Button(
        #     center_frame, text="⚡ Power", command=lambda: self.send_command('Power'), width=10, height=1, state='disabled'
        # )
        # power_button.grid(row=0, column=1, padx=10)  # Add padding for spacing
        # Tooltip(power_button, image_path=os.path.join(base_image_path, "horns.jpg"))

        # Fist button
        fist_button = tk.Button(
            center_frame, text="✊ Between Gestures", state="disabled", width=17, height=1
        )
        fist_button.grid(row=0, column=1, padx=10)  # Add padding for spacing
        Tooltip(fist_button, image_path=os.path.join(base_image_path, "fist.jpg"))


        # deactivate_button = tk.Button(
        #     control_frame, text="Deactivate", state="disabled", width=10, height=1
        # )
        # deactivate_button.grid(row=0, column=2, pady=5)  # Right of the Power button
        # Tooltip(deactivate_button, image_path=os.path.join(base_image_path, "fingers-deactivate.jpg"))

        # Back and Home Row
        back_button = tk.Button(control_frame, text="⟲ Back", command=lambda: self.send_command('Back'), width=10, height=1, state='disabled')
        Tooltip(back_button, image_path=os.path.join(base_image_path, "thumbs-out.jpg"), wide=True)

        power_button = tk.Button(
            control_frame, text="⚡ Power", command=lambda: self.send_command('Power'), width=10, height=1, state='disabled'
        )
        Tooltip(power_button, image_path=os.path.join(base_image_path, "horns.jpg"))

        home_button = tk.Button(control_frame, text="⌂ Home", command=lambda: self.send_command('Home'), width=10, height=1, state='disabled')
        Tooltip(home_button, image_path=os.path.join(base_image_path, "home.jpg"))

        back_button.grid(row=1, column=0, pady=5, padx=5)
        power_button.grid(row=1, column=1, pady=5, padx=5)
        home_button.grid(row=1, column=2, pady=5, padx=5)
        self.control_buttons.extend([back_button, power_button, home_button])

        # Navigation and Volume Controls
        nav_frame = tk.Frame(control_frame)
        nav_frame.grid(row=2, column=1, pady=5, padx=10)  # Navigation on the left

        up_button = tk.Button(nav_frame, text="↑", command=lambda: self.send_command('Up'), width=5, height=1, state='disabled')
        Tooltip(up_button, image_path=os.path.join(base_image_path, "i-up.jpg"))

        left_button = tk.Button(nav_frame, text="←", command=lambda: self.send_command('Left'), width=5, height=1, state='disabled')
        Tooltip(left_button, image_path=os.path.join(base_image_path, "i-left.jpg"))

        ok_button = tk.Button(nav_frame, text="OK", command=lambda: self.send_command('Select'), width=5, height=1, state='disabled')
        Tooltip(ok_button, image_path=os.path.join(base_image_path, "thumbs-in.jpg"), wide=True)

        right_button = tk.Button(nav_frame, text="→", command=lambda: self.send_command('Right'), width=5, height=1, state='disabled')
        Tooltip(right_button, image_path=os.path.join(base_image_path, "i-right.jpg"))

        down_button = tk.Button(nav_frame, text="↓", command=lambda: self.send_command('Down'), width=5, height=1, state='disabled')
        Tooltip(down_button, image_path=os.path.join(base_image_path, "i-down.jpg"))

        # Arrange Navigation Buttons in a Grid
        up_button.grid(row=0, column=1, pady=5)
        left_button.grid(row=1, column=0, pady=5)
        ok_button.grid(row=1, column=1, pady=5)
        right_button.grid(row=1, column=2, pady=5)
        down_button.grid(row=2, column=1, pady=5)

        self.control_buttons.extend([up_button, left_button, ok_button, right_button, down_button])

        volume_frame = tk.Frame(control_frame)
        volume_frame.grid(row=2, column=2, pady=5, padx=10)  # Volume controls on the right

        volume_down_button = tk.Button(volume_frame, text="🔉 Vol-", command=lambda: self.send_command('VolumeDown'), width=8, height=1, state='disabled')
        Tooltip(volume_down_button, image_path=os.path.join(base_image_path, "thumbs-down.jpg"))

        mute_button = tk.Button(volume_frame, text="🔇 Mute", command=lambda: self.send_command('VolumeMute'), width=8, height=1, state='disabled')
        Tooltip(mute_button, image_path=os.path.join(base_image_path, "fingers-mute.jpg"))

        volume_up_button = tk.Button(volume_frame, text="🔊 Vol+", command=lambda: self.send_command('VolumeUp'), width=8, height=1, state='disabled')
        Tooltip(volume_up_button, image_path=os.path.join(base_image_path, "thumbs-up.jpg"))

        # Arrange Volume Controls in a Grid
        volume_up_button.grid(row=0, column=0, pady=5)
        mute_button.grid(row=2, column=0, pady=5)
        volume_down_button.grid(row=1, column=0, pady=5)

        self.control_buttons.extend([volume_down_button, mute_button, volume_up_button])

        # Media Controls Row
        rw_button = tk.Button(control_frame, text="⏪ RW", command=lambda: self.send_command('Rev'), width=8, height=1, state='disabled')
        Tooltip(rw_button, image_path=os.path.join(base_image_path, "i-m-left.jpg"))

        play_pause_button = tk.Button(control_frame, text="⏯ Play/Pause", command=lambda: self.send_command('Play'), width=16, height=1, state='disabled')
        Tooltip(play_pause_button, image_path=os.path.join(base_image_path, "i-m-up.jpg"))

        ff_button = tk.Button(control_frame, text="⏩ FF", command=lambda: self.send_command('Fwd'), width=8, height=1, state='disabled')
        Tooltip(ff_button, image_path=os.path.join(base_image_path, "i-m-right.jpg"))

        rw_button.grid(row=3, column=0, pady=5)
        play_pause_button.grid(row=3, column=1, pady=5)
        ff_button.grid(row=3, column=2, pady=5)
        self.control_buttons.extend([rw_button, play_pause_button, ff_button])


        # Video Display Panel
        self.video_display_panel = VideoDisplayPanel(self.right_frame)
        self.video_display_panel.pack(fill='both', expand=True)
        self.video_display_panel.show_idle()  # Show idle message initially

        # Attempt to connect to known devices after GUI is set up
        self.attempt_connect_known_devices()

        self.root.after(0, self.root.geometry, "")

        # Check if auto_start is enabled and start video feed
        if self.auto_start:
            if self.roku_ip:
                print(f"Auto-starting video feed with camera index {self.selected_camera_index}")
                logging.debug(f"Auto-starting video feed with camera index {self.selected_camera_index}")
                self.start_video_feed()
            else:
                print("Auto-start is enabled, but no Roku device found. Waiting to connect.")
                logging.debug("Auto-start is enabled, but no Roku device found. Waiting to connect.")

        self.root.mainloop()

if __name__ == '__main__':
    roku_remote = RokuRemote()
    if roku_remote.roku_ip:
        print(f"Connected to Roku at {roku_remote.roku_ip}")
        logging.debug(f"Connected to Roku at {roku_remote.roku_ip}")
    else:
        print("No Roku device found on the network.")
        logging.debug("No Roku device found on the network.")
    roku_remote.setup_gui()
