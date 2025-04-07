import numpy as np
from pynput import keyboard
from flygym.examples.locomotion import PreprogrammedSteps, CPGNetwork
import threading

# Initialize CPG network
intrinsic_freqs = np.ones(6) * 12
intrinsic_amps = np.ones(6) * 1
phase_biases = np.pi * np.array(
    [
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
    ]
)
coupling_weights = (phase_biases > 0) * 10
convergence_coefs = np.ones(6) * 20

from cobarcontroller import CobarController

class KeyBoardController(CobarController):
    def __init__(self, timestep, seed, leg_step_time=0.025):

        self.timestep = timestep
        self.preprogrammed_steps = PreprogrammedSteps()
        self.intrinsic_freqs = intrinsic_freqs
        self.intrinsic_amps = intrinsic_amps
        self.phase_biases = phase_biases
        self.coupling_weights = coupling_weights
        self.convergence_coefs = convergence_coefs

        # Initialize CPG network
        self.cpg_network = CPGNetwork(
            timestep,
            intrinsic_freqs,
            intrinsic_amps,
            coupling_weights,
            phase_biases,
            convergence_coefs,
            np.random.rand(6),
            np.random.rand(6),
            seed,
        )

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)
        self.phase_increment = self.timestep / leg_step_time * 2 * np.pi

        self.tripod_map = {"LF": 0, "LM": 1, "LH": 0, "RF": 1, "RM": 0, "RH": 1}
        self.tripod_phases = np.zeros(2)

        self.goes_backward = False
        self.gain_right = 0.0
        self.gain_left = 0.0

        # Keyboard keys
        self.CPG_keys = ["w", "s", "a", "d"]

        # Shared lists to store key presses
        self.pressed_CPG_keys = []
        self.lock = threading.Lock()

        print("Starting key listener")
        # Start the keyboard listener thread
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.quit = False

    def on_press(self, key):
        key_str = (
            key.char if hasattr(key, "char") else str(key)
        )  # Gets the character of the key

        if key_str in self.CPG_keys:
            self.pressed_CPG_keys.append(key_str)
        if key_str == "Key.esc":  # Quit when esc is pressed
            self.listener.stop()
            self.quit = True

    def retrieve_keys(self):
        """Retrieve and clear all recorded key presses."""
        with self.lock:
            pCPG_keys = self.pressed_CPG_keys[:]
            self.pressed_CPG_keys.clear()

        return pCPG_keys

    def sort_keyboard_input(self, pCPG_keys):
        """Sorts the keys pressed and returns the last one."""
        keys = []
        if pCPG_keys:
            keys.append(max(set(pCPG_keys), key=pCPG_keys.count))

        return keys

    def set_CPGbias(self):

        # Retrieve all keys pressed since the last call
        keys = self.sort_keyboard_input(self.retrieve_keys())

        for key in keys:
            if key == "a":
                if self.goes_backward:
                    self.gain_right = -0.6
                    self.gain_left = -1.2
                else:
                    self.gain_left = 0.4
                    self.gain_right = 1.2
            elif key == "d":
                if self.goes_backward:
                    self.gain_left = -0.6
                    self.gain_right = -1.2
                else:
                    self.gain_right = 0.4
                    self.gain_left = 1.2
            elif key == "w":
                self.goes_backward = True
                self.gain_right = 1.0
                self.gain_left = 1.0
            elif key == "s":
                self.goes_backward = True
                self.gain_right = -1.0
                self.gain_left = -1.0

    def get_cpg_joint_angles(self):
        action = np.array([self.gain_left, self.gain_right])

        amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
        freqs = self.intrinsic_freqs.copy()
        freqs[:3] *= 1 if action[0] > 0 else -1
        freqs[3:] *= 1 if action[1] > 0 else -1
        self.cpg_network.intrinsic_amps = amps
        self.cpg_network.intrinsic_freqs = freqs

        self.cpg_network.step()

        joints_angles = []
        adhesion_onoff = []
        for i, leg in enumerate(self.preprogrammed_steps.legs):
            # get target angles from CPGs and apply correction
            my_joints_angles = self.preprogrammed_steps.get_joint_angles(
                leg,
                self.cpg_network.curr_phases[i],
                self.cpg_network.curr_magnitudes[i],
            )
            joints_angles.append(my_joints_angles)

            # get adhesion on/off signal
            my_adhesion_onoff = self.preprogrammed_steps.get_adhesion_onoff(
                leg, self.cpg_network.curr_phases[i]
            )

            # No adhesion in stumbling or retracted
            adhesion_onoff.append(my_adhesion_onoff)

        return {
            "joints": np.array(np.concatenate(joints_angles)),
            "adhesion": np.array(adhesion_onoff).astype(int),
        }

    def get_actions(self, obs):
        self.set_CPGbias()
        return self.get_cpg_joint_angles()

    def flush_keys(self):
        with self.lock:
            self.pressed_CPG_keys.clear()

    def reset(self, seed=None, init_phases=None, init_magnitudes=None, **kwargs):
        self.cpg_network.random_state = np.random.RandomState(seed)
        self.cpg_network.intrinsic_amps = self.intrinsic_amps
        self.cpg_network.intrinsic_freqs = self.intrinsic_freqs
        self.cpg_network.reset(init_phases, init_magnitudes)

        self.leg_phases = np.zeros(6)
        self.step_direction = np.zeros(6)
