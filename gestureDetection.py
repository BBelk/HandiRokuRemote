import threading
import mediapipe as mp
import time
import math
import logging

class HandClassifier(threading.Thread):
    def __init__(self, hand_id, initial_landmarks, command_callback, hand_label):
        super().__init__(daemon=True)
        self.hand_id = hand_id
        self.landmarks = initial_landmarks
        self.landmarks_lock = threading.Lock()
        self.running = True
        self.command_callback = command_callback
        self.hand_label = hand_label  # 'Left' or 'Right'

        # Gesture detection variables
        self.last_gesture = None
        self.state = 'Idle'  # Initial state is 'Idle'
        self.last_fist_time = None  # Time when fist was entered
        self.delay_after_fist = 0.3  # Delay in seconds

        self.r_gesture_detected = False  # Flag to prevent multiple transitions on continuous 'R' gesture
        self.r_gesture_start_time = None  # Start time when 'R' gesture is first detected
        self.min_r_duration = 0.3  # Minimum duration for 'R' gesture

        self.fist_made_in_active = False  # Flag to ensure fist is made after entering 'Active'

        # Variables to store current gesture, hand angle, and extended fingers
        self.current_gesture = None
        self.current_direction = None  # To store the current direction
        self.extended_fingers = None  # This will be a tuple of booleans (t, i, m, r, p)
        self.base_distance = None  # Base distance will be set upon detecting R gesture
        self.debug_string = ""
        self.thumb_distance = None
        self.angle = 0
        self.angle2 = 0

        # Variables to store points for drawing
        self.line_points = None  # Tuple of (thumb_point_index, selected_other_point_index)

        # Variables for gesture delay after fist
        self.gesture_start_time = None  # Start time when fingers are extended after fist
        self.min_gesture_duration = 0.3  # Minimum duration to hold the gesture before accepting it
        self.debug_r_string = ""
        self.end_tracking_time = time.time()

    def update_landmarks(self, new_landmarks, hand_label):
        with self.landmarks_lock:
            self.landmarks = new_landmarks
            self.hand_label = hand_label

    def stop(self):
        self.running = False

    def run(self):
        try:
            while self.running:
                with self.landmarks_lock:
                    landmarks = self.landmarks

                if landmarks:
                    # Process landmarks to detect gestures
                    gesture = self.detect_gesture(landmarks, self.hand_label)
                    if not self.running:
                        break  # Exit immediately if stopped
                    if gesture and gesture != self.last_gesture:
                        self.last_gesture = gesture
                        if self.current_direction is None:
                            self.current_direction = "None"
                        if gesture is not None:
                            # Build the debug string
                            self.debug_string = (
                                f"T:{self.extended_fingers[0]} I:{self.extended_fingers[1]} "
                                f"M:{self.extended_fingers[2]} R:{self.extended_fingers[3]} "
                                f"P:{self.extended_fingers[4]}\nThumb distance: {self.thumb_distance}\nAngle:{self.angle}"
                                f"Base Distance: {self.base_distance}"
                                # \nGesture: {self.current_gesture or 'None'}\n"
                                # f"Direction: {self.current_direction or 'None'}\nState: {self.state}"
                            )
                            # Call the callback function with the detected gesture
                            self.command_callback(gesture, self.debug_string, self.current_direction, self.state)
                    elif gesture is None:
                        self.last_gesture = None
                        if self.extended_fingers is not None:
                            self.debug_string = (
                                f"T:{self.extended_fingers[0]} I:{self.extended_fingers[1]} "
                                f"M:{self.extended_fingers[2]} R:{self.extended_fingers[3]} "
                                f"P:{self.extended_fingers[4]}\nThumb distance: {self.thumb_distance}\nAngle:{self.angle}"
                                f"Base Distance: {self.base_distance}"
                                # \nGesture: {self.current_gesture or 'None'}\n"
                                # f"Direction: {self.current_direction or 'None'}\nState: {self.state}"
                            )
                            self.command_callback(None, self.debug_string, self.current_direction, self.state)
                else:
                    self.last_gesture = None
                    # Do not reset the state immediately; handled in main code
                    self.r_gesture_detected = False  # Reset the R gesture flag when hand is lost
                    self.fist_made_in_active = False  # Reset fist flag
                    self.base_distance = None  # Reset base_distance when hand is lost
                    self.current_direction = None  # Reset direction
                    self.r_gesture_start_time = None  # Reset R gesture start time
                    self.gesture_start_time = None  # Reset gesture start time

                time.sleep(0.01)
        except Exception as e:
            print(f"Exception in HandClassifier thread: {e}")
            logging.debug(f"Exception in HandClassifier thread: {e}")

    def get_extended_fingers(self, landmarks):
        """
        Determines which fingers are extended based on angles and distances.
        Returns booleans for thumb, index, middle, ring, pinky fingers.
        Also stores the thumb point and the selected other point for drawing.
        """
        try:
            if self.base_distance is None:
                # Cannot determine finger extensions without base_distance
                self.line_points = None  # Reset line points
                return False, False, False, False, False

            import math  # Ensure math module is imported

            mp_hands = mp.solutions.hands

            wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # List of landmark IDs for all fingers (index, middle, ring, pinky)
            finger_tip_ids = [
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP
            ]
            finger_dip_ids = [
                mp_hands.HandLandmark.INDEX_FINGER_DIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                mp_hands.HandLandmark.RING_FINGER_DIP,
                mp_hands.HandLandmark.PINKY_DIP
            ]
            finger_pip_ids = [
                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp_hands.HandLandmark.RING_FINGER_PIP,
                mp_hands.HandLandmark.PINKY_PIP
            ]
            finger_mcp_ids = [
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.PINKY_MCP
            ]

            # Helper function to calculate the angle between three points
            def calculate_angle(a, b, c):
                """
                Calculates the angle at point 'b' given three points a, b, and c.
                """
                ba = [a.x - b.x, a.y - b.y, a.z - b.z]
                bc = [c.x - b.x, c.y - b.y, c.z - b.z]

                # Calculate the dot product and magnitudes
                dot_product = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
                magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
                magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)

                if magnitude_ba * magnitude_bc == 0:
                    return 0.0

                # Calculate the angle in radians and then convert to degrees
                angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
                angle_deg = math.degrees(angle_rad)
                return angle_deg

            # Check extension for each finger
            extended = []
            for tip_id, dip_id, pip_id, mcp_id in zip(finger_tip_ids, finger_dip_ids, finger_pip_ids, finger_mcp_ids):
                # Retrieve landmark points
                mcp = landmarks.landmark[mcp_id]
                pip = landmarks.landmark[pip_id]
                dip = landmarks.landmark[dip_id]
                tip = landmarks.landmark[tip_id]

                # Calculate angles at PIP and DIP joints
                angle_pip = calculate_angle(mcp, pip, dip)
                angle_dip = calculate_angle(pip, dip, tip)

                # Determine finger extension based on angles
                if angle_pip > 140 and angle_dip > 140:
                    angle_extended = True
                else:
                    angle_extended = False

                # Distance-based method (kept for robustness)
                distance = self.calculate_distance(tip, mcp)
                wrist_distance = self.calculate_distance(tip, wrist)
                knuckle_distance = self.calculate_distance(mcp, wrist)
                dip_distance = self.calculate_distance(dip, pip)

                if (wrist_distance < knuckle_distance) or (dip_distance < 0.35 * self.base_distance):
                    distance_extended = False
                elif distance > 0.5 * self.base_distance:
                    distance_extended = True
                else:
                    distance_extended = False

                # Combine angle and distance methods, angle method overrules
                if angle_extended and not (wrist_distance < knuckle_distance):
                    extended.append(True)
                else:
                    extended.append(distance_extended)

            # Thumb detection (unchanged)
            thumb_points = [
                (mp_hands.HandLandmark.THUMB_TIP, landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]),
                (mp_hands.HandLandmark.THUMB_IP, landmarks.landmark[mp_hands.HandLandmark.THUMB_IP])
            ]

            candidate_points = [
                (mp_hands.HandLandmark.INDEX_FINGER_MCP, landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]),
                (mp_hands.HandLandmark.INDEX_FINGER_DIP, landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]),
                (mp_hands.HandLandmark.INDEX_FINGER_PIP, landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]),
                (mp_hands.HandLandmark.MIDDLE_FINGER_DIP, landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]),
                (mp_hands.HandLandmark.MIDDLE_FINGER_MCP, landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]),
                (mp_hands.HandLandmark.RING_FINGER_MCP, landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP])
            ]

            min_distance = None
            selected_thumb_point = None
            selected_other_point = None

            for thumb_id, thumb_point in thumb_points:
                for other_id, other_point in candidate_points:
                    distance = self.calculate_distance(thumb_point, other_point)
                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        selected_thumb_point = thumb_id
                        selected_other_point = other_id

            self.thumb_distance = min_distance
            thumb_extended = self.thumb_distance > 0.3 * self.base_distance
            self.line_points = (selected_thumb_point, selected_other_point)

            thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended = [thumb_extended] + extended
            return thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended

        except Exception as e:
            print(f"Error in get_extended_fingers: {e}")
            logging.debug(f"Error in get_extended_fingers: {e}")
            return False, False, False, False, False

    def calculate_distance(self, point1, point2):
        """
        Calculates the Euclidean distance between two landmarks.
        """
        dx = point1.x - point2.x
        dy = point1.y - point2.y
        dz = point1.z - point2.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def detect_r_gesture(self, landmarks, hand_label):
        """
        Detects the 'R' gesture (sign language 'R') where index and middle fingers are crossed,
        only when the palm is facing the camera and the fingers are extended.
        Works for both left and right hands.
        """
        if self.end_tracking_time > time.time() - 1:
            return
        mp_hands = mp.solutions.hands
        self.debug_r_string = ""

        # Helper function to calculate the angle between three points
        def calculate_angle(a, b, c):
            """
            Calculates the angle at point 'b' given three points a, b, and c.
            """
            ba = [a.x - b.x, a.y - b.y, a.z - b.z]
            bc = [c.x - b.x, c.y - b.y, c.z - b.z]

            # Calculate the dot product and magnitudes
            dot_product = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
            magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
            magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)

            if magnitude_ba * magnitude_bc == 0:
                return 0.0

            # Calculate the angle in radians and then convert to degrees
            angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
            angle_deg = math.degrees(angle_rad)
            return angle_deg

        # Helper function to check if two line segments in 2D intersect
        def lines_intersect(p1, p2, p3, p4):
            """
            Checks if line segment p1-p2 intersects with line segment p3-p4.
            Points are expected to be in 2D.
            """
            def ccw(a, b, c):
                return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)

            return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

        wrist = landmarks.landmark[mp_hands.HandLandmark.WRIST]
        index_mcp = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_mcp = landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

        # Vectors along the palm
        v1 = [
            index_mcp.x - wrist.x,
            index_mcp.y - wrist.y,
            index_mcp.z - wrist.z
        ]
        v2 = [
            pinky_mcp.x - wrist.x,
            pinky_mcp.y - wrist.y,
            pinky_mcp.z - wrist.z
        ]

        # Normal vector (cross product)
        normal = [
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        ]
        # Adjust for left hand
        if hand_label == 'Left':
            normal = [-n for n in normal]

        # Determine if palm is facing the camera
        palm_facing_camera = normal[2] > 0
        angle_wrist = calculate_angle(wrist, index_mcp, index_tip)

        self.debug_r_string += f"n:{normal[2]} pfc:{palm_facing_camera}\n wrist_angle:{angle_wrist}"

        if not palm_facing_camera or angle_wrist < 150:
            return False

        # Check if index and middle fingers are extended using angles
        finger_ids = {
            'index': [
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.INDEX_FINGER_DIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ],
            'middle': [
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP
            ]
        }

        fingers_extended = {}
        for finger_name, ids in finger_ids.items():
            mcp = landmarks.landmark[ids[0]]
            pip = landmarks.landmark[ids[1]]
            dip = landmarks.landmark[ids[2]]
            tip = landmarks.landmark[ids[3]]

            angle_pip = calculate_angle(mcp, pip, dip)
            angle_dip = calculate_angle(pip, dip, tip)
            self.debug_r_string += f"\n{finger_name} pip: {angle_pip} dip: {angle_dip}"

            if angle_pip > 160 and angle_dip > 160:
                fingers_extended[finger_name] = True
            else:
                fingers_extended[finger_name] = False

        if not (fingers_extended['index'] and fingers_extended['middle']):
            return False

        # Now that fingers are extended, calculate temp_base_distance
        temp_base_distance = self.calculate_distance(index_tip, wrist)
        middle_mcp = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        middle_to_wrist_distance = self.calculate_distance(middle_mcp, wrist)

        self.debug_r_string += f"\ntbd: {temp_base_distance} mtwd: {middle_to_wrist_distance}"
        if temp_base_distance < middle_to_wrist_distance:
            return False

        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        tip_distance = self.calculate_distance(index_tip, middle_tip)

        fingers_close = tip_distance < 0.08 * temp_base_distance
        self.debug_r_string += f"\ntip_distance: {tip_distance}"

        if fingers_close:
            # Additional check: Do the lines from index PIP to TIP and middle PIP to TIP intersect?
            index_pip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_pip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

            # Convert 3D landmarks to 2D by ignoring the Z coordinate
            index_line_start = index_pip
            index_line_end = index_tip
            middle_line_start = middle_pip
            middle_line_end = middle_tip

            # Check if the lines intersect
            intersect = lines_intersect(
                index_line_start, index_line_end,
                middle_line_start, middle_line_end
            )

            if intersect:
                self.base_distance = self.calculate_distance(index_tip, middle_mcp)
                return True
            else:
                return False
        else:
            return False


    def detect_gesture(self, landmarks, hand_label):
        current_time = time.time()

        # Check for 'R' gesture in Idle state
        if self.state == "Idle":
            is_r_gesture = self.detect_r_gesture(landmarks, hand_label)
            if is_r_gesture:
                if self.r_gesture_start_time is None:
                    self.r_gesture_start_time = current_time
                    print(f"{self.hand_id}: 'R' gesture detected, starting timer...")
                    logging.debug(f"{self.hand_id}: 'R' gesture detected, starting timer...")
                elif current_time - self.r_gesture_start_time >= self.min_r_duration:
                    self.state = "Active"
                    self.r_gesture_detected = True
                    self.last_fist_time = None
                    self.fist_made_in_active = False
                    print(f"{self.hand_id}: Transition to 'Active' state (Gesture detection ON)")
                    logging.debug(f"{self.hand_id}: Transition to 'Active' state (Gesture detection ON)")
            else:
                self.r_gesture_start_time = None

            return None

        if self.state == "Active":
            self.r_gesture_start_time = None

            thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended = self.get_extended_fingers(landmarks)

            self.extended_fingers = (thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended)

            if not self.fist_made_in_active:
                if self.is_fist():
                    self.last_fist_time = current_time
                    self.fist_made_in_active = True
                    print(f"{self.hand_id}: Fist detected in 'Active' state")
                    logging.debug(f"{self.hand_id}: Fist detected in 'Active' state")
                    return 'Fist'
                return None

            if self.last_fist_time and (current_time - self.last_fist_time < self.delay_after_fist):
                return None

            if not self.is_fist():
                if self.gesture_start_time is None:
                    self.gesture_start_time = current_time
                elif current_time - self.gesture_start_time >= self.min_gesture_duration:
                    gesture = self.recognize_gesture(landmarks, hand_label)
                    if gesture:
                        print(f"{self.hand_id}: Detected gesture '{gesture}'")
                        logging.debug(f"{self.hand_id}: Detected gesture '{gesture}'")
                        self.last_fist_time = None
                        self.fist_made_in_active = False
                        self.gesture_start_time = None
                        return gesture
                    # if self.extended_fingers == [False, True, True, False, False]:
                    #     detect_r = self.detect_r_gesture(landmarks, hand_label)
                    # else:    
                    #     gesture = self.recognize_gesture(landmarks, hand_label)
                    #     if gesture and not detect_r:
                    #         print(f"{self.hand_id}: Detected gesture '{gesture}'")
                    #         logging.debug(f"{self.hand_id}: Detected gesture '{gesture}'")
                    #         self.last_fist_time = None
                    #         self.fist_made_in_active = False
                    #         self.gesture_start_time = None
                    #         return gesture
                    #     if detect_r:
                    #         self.return_to_idle()
            else:
                self.gesture_start_time = None

        return None

    def is_fist(self):
        """
        Determines if the hand is in a fist position.
        """
        # Ensure base_distance is set before checking finger extensions
        if self.base_distance is None or self.extended_fingers is None:
            return False

        thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended = self.extended_fingers
        return not (thumb_extended or index_extended or middle_extended or ring_extended or pinky_extended)

    def recognize_gesture(self, landmarks, hand_label):
        """
        Recognizes gestures after the 'Fist' state.
        Uses the hand direction and finger extensions.
        """
        # Ensure base_distance is set before recognizing gestures
        if self.base_distance is None:
            return None

        # Get booleans for each finger
        thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended = self.extended_fingers
        mp_hands = mp.solutions.hands

        gesture = None
        direction = None

        # Create a list of the extended fingers
        extended_fingers = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]

        if extended_fingers == [True, False, False, False, False]:  # Only thumb extended
            print("thumb")
            logging.debug("Only thumb extended")
            from_angle = landmarks.landmark[mp_hands.HandLandmark.WRIST]
            tip_point = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            direction = self.get_direction(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], tip_point, is_thumb=True)
            if direction == "Left":
                gesture = "Select"
                if hand_label == 'Left':
                    gesture = "Back"
            elif direction == "Right":
                gesture = "Back"
                if hand_label == 'Left':
                    gesture = "Select"
            elif direction == "Up":
                gesture = "VolumeUp"
            elif direction == "Down":
                gesture = "VolumeDown"

        elif extended_fingers[1:] == [True, False, False, False]:  # Index finger only, ignoring thumb
            tip_point = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            direction = self.get_direction(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], tip_point)
            gesture = direction

        elif extended_fingers[1:] == [True, True, False, False]:
            tip_point = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            direction = self.get_direction(landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP], tip_point)
            if direction == "Left":
                gesture = "Rev"
            if direction == "Right":
                gesture = "Fwd"
            if direction == "Up":
                if self.detect_r_gesture(landmarks, hand_label):
                    self.return_to_idle()
                    gesture = "End Tracking"
                    return gesture
                else:
                    gesture = "Play"

        elif extended_fingers[1:] == [True, True, True, False]:
            tip_point = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            mcp_point = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            direction = self.get_direction(mcp_point, tip_point)

            # if direction == "Up":
            #     self.state = "Idle"
            #     print(f"{self.hand_id}: Transition to 'Idle' state (Gesture detection OFF)")
            #     logging.debug(f"{self.hand_id}: Transition to 'Idle' state (Gesture detection OFF)")
            #     self.last_fist_time = None
            #     self.fist_made_in_active = False
            #     self.gesture_start_time = None
            #     return None
            if direction == "Down":
                # Mute volume
                gesture = "VolumeMute"
                return gesture
            else:
                return None

        elif extended_fingers == [False, True, False, False, True]:  # Index and pinky extended
            gesture = "Power"
            direction = None
        elif extended_fingers == [True, False, False, False, True]:  # Thumb and pinky
            gesture = "Home"
            direction = None

        else:
            gesture = None
            self.current_direction = None
            return None

        if direction:
            self.current_direction = direction  # Store direction for debug string
        else:
            self.current_direction = None

        return gesture
    
    def return_to_idle(self):
        self.state = "Idle"
        print(f"{self.hand_id}: Transition to 'Idle' state (Gesture detection OFF)")
        logging.debug(f"{self.hand_id}: Transition to 'Idle' state (Gesture detection OFF)")
        self.last_fist_time = None
        self.fist_made_in_active = False
        self.gesture_start_time = None
        # time.sleep(1)
        self.end_tracking_time = time.time()
        return None

    def get_direction(self, wrist_point, tip_point, is_thumb=False):
        """
        Determines direction (Up, Down, Left, Right) based on tip point relative to wrist.
        Uses normal 90-degree quadrants for thumb gestures and adjusted thresholds for others.
        """
        dx = tip_point.x - wrist_point.x
        dy = tip_point.y - wrist_point.y

        # Calculate angle in degrees and normalize it to [0, 360)
        angle = math.degrees(math.atan2(dy, dx))  # Calculate raw angle
        angle = (angle + 360) % 360  # Normalize angle to range [0, 360)

        print(f"Normalized angle: {angle}")
        # logging.debug(f"Normalized angle: {angle}")

        # Adjust angle based on hand label (if needed)
        if self.hand_label == 'Left':
            dx = -dx  # Flip the x-axis for left hand
            angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360  # Recalculate angle after flipping dx

        # Determine direction based on angle
        if is_thumb:
            # Use normal 90-degree quadrants for thumb
            if 45 <= angle < 135:
                return 'Down'
            elif 135 <= angle < 225:
                return 'Left'
            elif 225 <= angle < 315:
                return 'Up'
            else:
                # Covers angles from 315 to 360 and 0 to 45
                return 'Right'
        else:
            # Adjusted thresholds:
            # Up/Down reduced to 45 degrees each
            # Left/Right increased to 135 degrees each
            if 0 <= angle < 67.5 or 292.5 <= angle < 360:
                return 'Right'
            elif 67.5 <= angle < 112.5:
                return 'Down'
            elif 112.5 <= angle < 247.5:
                return 'Left'
            elif 247.5 <= angle < 292.5:
                return 'Up'
            else:
                return 'Unknown'


