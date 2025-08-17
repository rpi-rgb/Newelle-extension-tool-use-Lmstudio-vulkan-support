# -*- coding: utf-8 -*-
"""
Extension : Augmented Reality Avatar

This extension displays a 3D avatar in augmented reality, using the device's camera feed as a background.
"""

from .extensions import NewelleExtension
from .handlers.avatar import AvatarHandler
from .handlers import HandlerDescription, ExtraSettings
from .utility.pip import install_module, find_module

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib

try:
    from OpenGL import GL
except ImportError:
    pass # Will be handled by is_installed

import cv2
import numpy as np
import threading
import os
import pygltflib
import pyrr

class ARExtension(NewelleExtension):
    """
    Newelle extension to enable Augmented Reality avatars.
    """
    id = "ar_avatar"
    name = "Augmented Reality Avatar"

    def get_avatar_handlers(self) -> list[dict]:
        """
        Returns the AR avatar handler.
        """
        return [{
            "key": "ar",
            "title": "AR Avatar",
            "description": "3D Avatar in Augmented Reality",
            "class": ARHandler
        }]

class ARGLArea(Gtk.GLArea):
    """
    Custom Gtk.GLArea for rendering the AR scene.
    """

    BG_VERTEX_SHADER = """
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main()
        {
            gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    """

    BG_FRAGMENT_SHADER = """
        #version 330 core
        out vec4 FragColor;

        in vec2 TexCoord;

        uniform sampler2D background;

        void main()
        {
            FragColor = texture(background, TexCoord);
        }
    """

    MODEL_VERTEX_SHADER = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in ivec4 aJoints;
        layout (location = 2) in vec4 aWeights;

        uniform mat4 u_projection;
        uniform mat4 u_view;
        uniform mat4 u_model;

        const int MAX_JOINTS = 128;
        uniform mat4 u_joint_matrices[MAX_JOINTS];

        void main()
        {
            mat4 skin_matrix = aWeights.x * u_joint_matrices[aJoints.x] +
                               aWeights.y * u_joint_matrices[aJoints.y] +
                               aWeights.z * u_joint_matrices[aJoints.z] +
                               aWeights.w * u_joint_matrices[aJoints.w];

            gl_Position = u_projection * u_view * u_model * skin_matrix * vec4(aPos, 1.0);
        }
    """

    MODEL_FRAGMENT_SHADER = """
        #version 330 core
        out vec4 FragColor;

        void main()
        {
            FragColor = vec4(1.0, 0.5, 0.2, 1.0); // Orange color for now
        }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_required_version(3, 3)
        self.connect("realize", self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render", self._on_render)

        self.video_capture = None
        self.background_texture = None

        self.bg_shader_program = None
        self.bg_vao = None
        self.bg_vbo = None

        self.model_shader_program = None
        self.model_vao = None
        self.model_vbo = None
        self.model_vbo_joints = None
        self.model_vbo_weights = None
        self.model_ebo = None
        self.model_indices_count = 0
        self.model_indices_type = GL.GL_UNSIGNED_SHORT

        self.model_data = None
        self.animations = {}
        self.inverse_bind_matrices = []

        self.active_animation = None
        self.animation_start_time = 0
        self.joint_matrices = []

    def set_model(self, model):
        self.model_data = model
        self.make_current()
        self._process_model()
        print("Model received and processed in GLArea.")

    def play_animation(self, name):
        if name in self.animations:
            self.active_animation = self.animations[name]
            self.animation_start_time = GLib.get_monotonic_time() / 1000000.0 # a float of seconds
            print(f"Playing animation: {name}")
        else:
            print(f"Animation not found: {name}")

    def _update_animation(self):
        if not self.active_animation or not self.model_data:
            self.joint_matrices = [pyrr.matrix44.create_identity() for _ in range(128)]
            return

        current_time = (GLib.get_monotonic_time() / 1000000.0) - self.animation_start_time

        duration = 0
        for sampler in self.active_animation['samplers']:
            duration = max(duration, sampler['input'][-1])

        if duration > 0:
            current_time = current_time % duration

        node_transforms = [pyrr.matrix44.create_identity() for _ in self.model_data.nodes]

        for channel in self.active_animation['channels']:
            sampler = self.active_animation['samplers'][channel['sampler']]
            target_node_idx = channel['target_node']
            target_path = channel['target_path']

            for i in range(len(sampler['input']) - 1):
                if sampler['input'][i] <= current_time < sampler['input'][i+1]:
                    t = (current_time - sampler['input'][i]) / (sampler['input'][i+1] - sampler['input'][i])

                    if target_path == 'rotation':
                        q1 = pyrr.Quaternion(sampler['output'][i])
                        q2 = pyrr.Quaternion(sampler['output'][i+1])
                        q_interpolated = pyrr.Quaternion.slerp(q1, q2, t)
                        node_transforms[target_node_idx] = pyrr.matrix44.create_from_quaternion(q_interpolated)
                    break

        skin = self.model_data.skins[0]
        self.joint_matrices = [pyrr.matrix44.create_identity() for _ in skin.joints]

        for i, joint_index in enumerate(skin.joints):
            self.joint_matrices[i] = pyrr.matrix44.multiply(node_transforms[joint_index], self.inverse_bind_matrices[i])

    def _get_gl_type(self, component_type):
        if component_type == 5121: return GL.GL_UNSIGNED_BYTE
        if component_type == 5123: return GL.GL_UNSIGNED_SHORT
        if component_type == 5125: return GL.GL_UNSIGNED_INT
        return GL.GL_UNSIGNED_SHORT # Default

    def _process_model(self):
        if not self.model_data:
            return

        # Find skin
        skin = self.model_data.skins[0] # Assuming one skin

        # Get inverse bind matrices
        ibm_accessor = self.model_data.accessors[skin.inverseBindMatrices]
        ibm_buffer_view = self.model_data.bufferViews[ibm_accessor.bufferView]
        ibm_buffer = self.model_data.buffers[ibm_buffer_view.buffer]
        ibm_data = ibm_buffer.uri_as_bytes()[ibm_buffer_view.byteOffset:ibm_buffer_view.byteOffset + ibm_buffer_view.byteLength]
        self.inverse_bind_matrices = np.frombuffer(ibm_data, dtype=np.float32).reshape((-1, 4, 4))

        # For simplicity, we'll render the first mesh of the first scene.
        scene = self.model_data.scenes[self.model_data.scene]
        mesh_index = scene.nodes[0]
        node = self.model_data.nodes[mesh_index]
        mesh = self.model_data.meshes[node.mesh]
        primitive = mesh.primitives[0]

        # Get vertex data
        def get_data(attribute):
            accessor = self.model_data.accessors[attribute]
            buffer_view = self.model_data.bufferViews[accessor.bufferView]
            buffer = self.model_data.buffers[buffer_view.buffer]
            return buffer.uri_as_bytes()[buffer_view.byteOffset + accessor.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]

        pos_data = get_data(primitive.attributes.POSITION)
        joints_data = get_data(primitive.attributes.JOINTS_0)
        weights_data = get_data(primitive.attributes.WEIGHTS_0)

        indices_accessor = self.model_data.accessors[primitive.indices]
        indices_data = get_data(primitive.indices)
        self.model_indices_count = indices_accessor.count
        self.model_indices_type = self._get_gl_type(indices_accessor.componentType)

        self.model_vao = GL.glGenVertexArrays(1)
        self.model_vbo = GL.glGenBuffers(1)
        self.model_vbo_joints = GL.glGenBuffers(1)
        self.model_vbo_weights = GL.glGenBuffers(1)
        self.model_ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.model_vao)

        # Positions
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.model_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(pos_data), pos_data, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Joints
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.model_vbo_joints)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(joints_data), joints_data, GL.GL_STATIC_DRAW)
        GL.glVertexAttribIPointer(1, 4, GL.GL_UNSIGNED_SHORT, 0, None) # Assuming joints are u_short
        GL.glEnableVertexAttribArray(1)

        # Weights
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.model_vbo_weights)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, len(weights_data), weights_data, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(2, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(2)

        # Indices
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.model_ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, len(indices_data), indices_data, GL.GL_STATIC_DRAW)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

        # Parse animations
        for anim in self.model_data.animations:
            self.animations[anim.name] = {"channels": [], "samplers": []}
            for sampler in anim.samplers:
                input_accessor = self.model_data.accessors[sampler.input]
                output_accessor = self.model_data.accessors[sampler.output]

                input_data = np.frombuffer(get_data(sampler.input), dtype=np.float32)
                output_data = np.frombuffer(get_data(sampler.output), dtype=np.float32)

                if output_accessor.type == "VEC4":
                    output_data = output_data.reshape((-1, 4))
                elif output_accessor.type == "VEC3":
                    output_data = output_data.reshape((-1, 3))

                self.animations[anim.name]["samplers"].append({
                    "input": input_data,
                    "output": output_data,
                    "interpolation": sampler.interpolation
                })

            for channel in anim.channels:
                self.animations[anim.name]["channels"].append({
                    "sampler": channel.sampler,
                    "target_node": channel.target.node,
                    "target_path": channel.target.path
                })

    def _compile_shader(self, source, shader_type):
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            error = GL.glGetShaderInfoLog(shader).decode()
            print(f"Error compiling shader: {error}")
            return None
        return shader

    def _on_realize(self, area):
        """Called when the GLArea is realized."""
        area.make_current()
        try:
            # Initialize OpenGL
            GL.glClearColor(0.0, 0.0, 0.0, 1.0)
            GL.glEnable(GL.GL_DEPTH_TEST)

            # Compile background shaders
            vertex_shader = self._compile_shader(self.BG_VERTEX_SHADER, GL.GL_VERTEX_SHADER)
            fragment_shader = self._compile_shader(self.BG_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
            if not vertex_shader or not fragment_shader: return
            self.bg_shader_program = GL.glCreateProgram()
            GL.glAttachShader(self.bg_shader_program, vertex_shader)
            GL.glAttachShader(self.bg_shader_program, fragment_shader)
            GL.glLinkProgram(self.bg_shader_program)
            if not GL.glGetProgramiv(self.bg_shader_program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(self.bg_shader_program).decode()
                print(f"Error linking bg program: {error}")
                return
            GL.glDeleteShader(vertex_shader)
            GL.glDeleteShader(fragment_shader)

            # Compile model shaders
            vertex_shader = self._compile_shader(self.MODEL_VERTEX_SHADER, GL.GL_VERTEX_SHADER)
            fragment_shader = self._compile_shader(self.MODEL_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
            if not vertex_shader or not fragment_shader: return
            self.model_shader_program = GL.glCreateProgram()
            GL.glAttachShader(self.model_shader_program, vertex_shader)
            GL.glAttachShader(self.model_shader_program, fragment_shader)
            GL.glLinkProgram(self.model_shader_program)
            if not GL.glGetProgramiv(self.model_shader_program, GL.GL_LINK_STATUS):
                error = GL.glGetProgramInfoLog(self.model_shader_program).decode()
                print(f"Error linking model program: {error}")
                return
            GL.glDeleteShader(vertex_shader)
            GL.glDeleteShader(fragment_shader)

            # Create quad vertices
            quad_vertices = np.array([
                # positions   # texCoords
                -1.0,  1.0,  0.0, 1.0,
                -1.0, -1.0,  0.0, 0.0,
                 1.0, -1.0,  1.0, 0.0,

                -1.0,  1.0,  0.0, 1.0,
                 1.0, -1.0,  1.0, 0.0,
                 1.0,  1.0,  1.0, 1.0
            ], dtype=np.float32)

            self.bg_vao = GL.glGenVertexArrays(1)
            self.bg_vbo = GL.glGenBuffers(1)
            GL.glBindVertexArray(self.bg_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.bg_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, GL. ctypes.c_void_p(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 4 * 4, GL.ctypes.c_void_p(2 * 4))

            # Initialize camera
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                print("Error: Could not open video stream.")
                return

            # Create texture for camera feed
            self.background_texture = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.background_texture)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

            print("AR GLArea realized.")

        except GL.GLError as e:
            print(f"OpenGL Error on realize: {e}")
            return

    def _on_unrealize(self, area):
        """Called when the GLArea is unrealized."""
        if self.video_capture:
            self.video_capture.release()

        area.make_current()
        if self.background_texture:
            GL.glDeleteTextures(1, [self.background_texture])
        if self.bg_shader_program:
            GL.glDeleteProgram(self.bg_shader_program)
        if self.bg_vao:
            GL.glDeleteVertexArrays(1, [self.bg_vao])
        if self.bg_vbo:
            GL.glDeleteBuffers(1, [self.bg_vbo])

        if self.model_shader_program:
            GL.glDeleteProgram(self.model_shader_program)
        if self.model_vao:
            GL.glDeleteVertexArrays(1, [self.model_vao])
        if self.model_vbo:
            GL.glDeleteBuffers(1, [self.model_vbo])
        if self.model_ebo:
            GL.glDeleteBuffers(1, [self.model_ebo])

        print("AR GLArea unrealized.")

    def _on_render(self, area, ctx):
        """Called for each frame to render the scene."""
        if not self.video_capture or not self.video_capture.isOpened():
            return True

        # Capture frame-by-frame
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1) # Flip horizontally

            # Update background texture
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.background_texture)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, frame.shape[1], frame.shape[0], 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, frame)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        # Render the background texture
        GL.glUseProgram(self.bg_shader_program)
        GL.glBindVertexArray(self.bg_vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.background_texture)
        GL.glUniform1i(GL.glGetUniformLocation(self.bg_shader_program, "background"), 0)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        if self.model_data and self.model_vao:
            self._update_animation()

            GL.glUseProgram(self.model_shader_program)

            # Setup matrices
            width = self.get_allocated_width()
            height = self.get_allocated_height()

            projection = pyrr.matrix44.create_perspective_projection_matrix(45, width / height, 0.1, 100.0)
            view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
            model_matrix = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, -1, 0]))

            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.model_shader_program, "u_projection"), 1, GL.GL_FALSE, projection)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.model_shader_program, "u_view"), 1, GL.GL_FALSE, view)
            GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.model_shader_program, "u_model"), 1, GL.GL_FALSE, model_matrix)

            if self.joint_matrices:
                GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.model_shader_program, "u_joint_matrices"), len(self.joint_matrices), GL.GL_FALSE, self.joint_matrices)

            GL.glBindVertexArray(self.model_vao)
            GL.glDrawElements(GL.GL_TRIANGLES, self.model_indices_count, self.model_indices_type, None)

        # Force redraw
        self.queue_draw()
        return True


class ARHandler(AvatarHandler):
    """
    Avatar handler for Augmented Reality.
    """
    key = "ar"

    def __init__(self, settings, path: str):
        super().__init__(settings, path)
        self.widget = None
        self.model = None

    def get_extra_settings(self) -> list:
        """
        Returns extra settings for the AR avatar.
        """
        return [
            {
                "key": "model_path",
                "title": "3D Model Path",
                "description": "Path to the .glb/.gltf model file",
                "type": "entry",
                "default": "",
            }
        ]

    def is_installed(self) -> bool:
        """
        Checks if all dependencies are installed.
        """
        return all([
            find_module('OpenGL', self.pip_path),
            find_module('cv2', self.pip_path),
            find_module('numpy', self.pip_path),
            find_module('pygltflib', self.pip_path),
            find_module('pyrr', self.pip_path)
        ])

    def install(self):
        """
        Installs all necessary dependencies.
        """
        install_module('PyOpenGL', self.pip_path)
        install_module('opencv-python', self.pip_path)
        install_module('numpy', self.pip_path)
        install_module('pygltflib', self.pip_path)
        install_module('pyrr', self.pip_path)

    def _load_model(self):
        model_path = self.get_setting("model_path")
        if model_path and os.path.exists(model_path):
            try:
                self.model = pygltflib.GLTF2().load(model_path)
                print(f"Model {model_path} loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            self.model = None

    def create_gtk_widget(self) -> Gtk.Widget:
        """
        Creates the GTK widget for the AR avatar.
        """
        self._load_model()
        if not self.widget:
            self.widget = ARGLArea(hexpand=True, vexpand=True)

        if self.model:
            self.widget.set_model(self.model)

        return self.widget

    # TODO: Implement other AvatarHandler methods like speak, set_expression, do_motion
    def speak(self, path: str, tts, frame_rate: int):
        pass

    def set_expression(self, expression: str):
        pass

    def do_motion(self, motion: str):
        if self.widget:
            self.widget.play_animation(motion)
