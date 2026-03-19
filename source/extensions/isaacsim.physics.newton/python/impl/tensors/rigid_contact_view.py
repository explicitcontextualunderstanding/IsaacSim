# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rigid contact view for Newton physics tensor interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .backend import RigidContactSet
    from .frontends import NumpyFrontend, TorchFrontend, WarpFrontend

from .contact_kernels import (
    contact_data_kernel,
    contact_force_matrix_kernel,
    count_contacts_per_pair_kernel,
    net_contact_forces_kernel,
)
from .kernels import *
from .tensor_utils import zero_tensor

# Import tensor types from omni.physics.tensors for compatibility
try:
    from omni.physics.tensors import float32, uint8, uint32
except ImportError:
    float32 = wp.float32
    uint8 = wp.uint8
    uint32 = wp.uint32

copy_data = True


class NewtonRigidContactView:
    """View for contact sensors in Newton physics simulation.

    Provides tensor-based access to contact forces and contact data,
    matching PhysX IRigidContactView interface.

    Args:
        backend: RigidContactSet backend instance from NewtonSimView.
        frontend: Tensor framework frontend.
    """

    def __init__(self, backend: "RigidContactSet", frontend: "NumpyFrontend | TorchFrontend | WarpFrontend"):
        self._backend = backend
        self._frontend = frontend
        self._newton_stage = backend.newton_stage
        self._model = backend.model
        self._sim_timestamp = 0

        # Create tensors for caching contact forces
        self._net_forces, self._net_forces_desc = self._frontend.create_tensor((self.sensor_count, 3), float32)

        # Use body-to-sensor mapping from backend (already created with correct original body indices)
        self._body_sensor_map = self._backend.body_sensor_map

        # Create body-to-filter mapping
        self._body_filter_map = None
        self._create_filter_mappings()

        # Create additional tensors for detailed contact data (lazy init)
        self._force_matrix = None
        self._contact_forces_buffer = None
        self._contact_points_buffer = None
        self._contact_normals_buffer = None
        self._contact_separations_buffer = None
        self._contact_counts = None
        self._contact_start_indices = None

    def _create_filter_mappings(self):
        """Create filter mapping array for sensor-body-filter lookups."""
        if self._model is None:
            return

        # Create sensor x body -> filter index mapping (-1 if no filter)
        # This is a 2D map: [sensor_count, body_count] -> filter_idx
        num_bodies = self._model.body_count
        body_filter_map = np.full((self.sensor_count, num_bodies), -1, dtype=np.int32)

        # Populate filter mappings from backend's filter data
        # backend stores filter_paths_list: list of lists, one per sensor
        if hasattr(self._backend, "filter_paths") and self._backend.filter_paths:
            for sensor_idx in range(self.sensor_count):
                if sensor_idx < len(self._backend.filter_paths):
                    filter_paths_for_sensor = self._backend.filter_paths[sensor_idx]
                    for filter_idx, filter_path in enumerate(filter_paths_for_sensor):
                        # Find the body index for this filter path
                        try:
                            body_idx = list(self._model.body_label).index(filter_path)
                            body_filter_map[sensor_idx, body_idx] = filter_idx
                        except ValueError:
                            carb.log_warn(f"[NewtonRigidContactView] Filter body not found: {filter_path}")

        self._body_filter_map = wp.array(body_filter_map, dtype=wp.int32, device=self._model.device)

    @property
    def count(self) -> int:
        """Number of contact sensors in this view.

        Returns:
            Number of contact sensors.
        """
        return self._backend.count

    @property
    def sensor_count(self) -> int:
        """Number of contact sensors.

        Returns:
            Number of contact sensors.
        """
        return self._backend.sensor_count

    @property
    def filter_count(self) -> int:
        """Maximum number of filter bodies per sensor.

        Returns:
            Maximum number of filter bodies per sensor.
        """
        return self._backend.filter_count

    @property
    def sensor_names(self) -> list[str]:
        """List of sensor names.

        Returns:
            List of sensor names.
        """
        return self._backend.sensor_names

    @property
    def sensor_paths(self) -> list[str]:
        """List of USD paths for sensors.

        Returns:
            List of USD paths for sensors.
        """
        return self._backend.sensor_paths

    @property
    def filter_paths(self) -> list[list[str]]:
        """List of filter paths (list of lists, one list per sensor).

        Returns:
            List of filter paths with nested lists for each sensor.
        """
        return self._backend.filter_paths

    @property
    def filter_names(self) -> list[list[str]]:
        """List of filter names.

        Returns:
            List of filter names.
        """
        return self._backend.filter_names

    def update(self, dt: float):
        """Update simulation timestamp.

        Args:
            dt: Time delta.
        """
        self._sim_timestamp += dt

    @carb.profiler.profile
    def get_net_contact_forces(self, dt: float, copy: bool = copy_data):
        """Get net contact forces for each sensor.

        Args:
            dt: Time step for force calculation.
            copy: Whether to return a copy.

        Returns:
            A tensor of shape (sensor_count, 3) with net contact forces.
        """
        state = self._newton_stage.state_0
        model = self._model
        zero_tensor(self._net_forces)

        # Check if we have contact data in Newton
        contacts = self._newton_stage.contacts

        # If no contacts available or not populated, return zeros
        if contacts is None or not hasattr(contacts, "force") or not hasattr(contacts, "normal"):
            return self._net_forces

        # Get contact count - MuJoCo uses n_contacts, other solvers use rigid_contact_count
        if hasattr(contacts, "n_contacts"):
            contact_count = (
                contacts.n_contacts.numpy()[0] if hasattr(contacts.n_contacts, "numpy") else int(contacts.n_contacts)
            )
        elif hasattr(contacts, "rigid_contact_count"):
            contact_count = (
                contacts.rigid_contact_count.numpy()[0] if hasattr(contacts.rigid_contact_count, "numpy") else 0
            )
        else:
            contact_count = 0

        if contact_count == 0:
            # No contacts detected - return zeros (this is valid, not an error)
            return self._net_forces

        if model.rigid_contact_max > 0:

            wp.launch(
                kernel=net_contact_forces_kernel,
                dim=model.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count if hasattr(contacts, "rigid_contact_count") else contacts.n_contacts,
                    contacts.pair,  # MuJoCo uses pair instead of rigid_contact_shape0/1
                    contacts.normal,  # Pre-computed normal from solver
                    contacts.force,  # Pre-computed force magnitude from solver
                    model.shape_body,
                    self._body_sensor_map,
                ],
                outputs=[self._net_forces],
                device=model.device,
            )
        wp.synchronize_device(model.device)

        return self._net_forces

    def get_contact_force_matrix(self, dt: float, copy: bool = copy_data) -> wp.array:
        """Get contact force matrix (sensor x filter).

        Args:
            dt: Time step for force calculation.
            copy: Whether to return a copy.

        Returns:
            A tensor of shape (sensor_count, filter_count, 3).
        """
        # Lazy initialization
        if self._force_matrix is None:
            self._force_matrix, _ = self._frontend.create_tensor((self.sensor_count, self.filter_count, 3), float32)

        state = self._newton_stage.state_0
        model = self._model

        # Zero out force matrix (backend-agnostic)
        if hasattr(self._force_matrix, "fill_"):
            self._force_matrix.fill_(0.0)
        elif hasattr(self._force_matrix, "zero_"):
            self._force_matrix.zero_()
        else:
            self._force_matrix[:] = 0.0

        # Check if we have contact data
        contacts = self._newton_stage.contacts

        # If no contacts available or not populated, return zeros
        if contacts is None or not hasattr(contacts, "force") or not hasattr(contacts, "normal"):
            print("[get_contact_force_matrix] No contacts or missing force/normal fields")
            return self._force_matrix

        # Get contact count - MuJoCo uses n_contacts, other solvers use rigid_contact_count
        if hasattr(contacts, "n_contacts"):
            contact_count = (
                contacts.n_contacts.numpy()[0] if hasattr(contacts.n_contacts, "numpy") else int(contacts.n_contacts)
            )
        elif hasattr(contacts, "rigid_contact_count"):
            contact_count = (
                contacts.rigid_contact_count.numpy()[0] if hasattr(contacts.rigid_contact_count, "numpy") else 0
            )
        else:
            contact_count = 0

        if model.rigid_contact_max > 0 and contact_count > 0:
            wp.launch(
                kernel=contact_force_matrix_kernel,
                dim=model.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count if hasattr(contacts, "rigid_contact_count") else contacts.n_contacts,
                    contacts.pair,  # MuJoCo uses pair instead of rigid_contact_shape0/1
                    contacts.normal,  # Normal from solver (vec3)
                    contacts.force,  # Pre-computed scalar force magnitude from solver
                    model.shape_body,
                    self._body_sensor_map,
                    self._body_filter_map,
                    self.filter_count,
                ],
                outputs=[self._force_matrix],
                device=model.device,
            )
            wp.synchronize_device(model.device)

        return self._force_matrix

    def get_contact_data(self, dt: float, max_contact_data_count: int = 1000, copy: bool = copy_data):
        """Get detailed contact data (forces, points, normals, separations).

        Args:
            dt: Time step for force calculation.
            max_contact_data_count: Maximum number of contact points to store.
            copy: Whether to return copies.

        Returns:
            A tuple of (contact_forces, contact_points, contact_normals,
            contact_separations, contact_counts, contact_start_indices) where:

            - contact_forces: shape (max_contact_data_count, 1) - normal force magnitudes
            - contact_points: shape (max_contact_data_count, 3) - world space contact positions
            - contact_normals: shape (max_contact_data_count, 3) - contact normals
            - contact_separations: shape (max_contact_data_count, 1) - penetration depths
            - contact_counts: shape (sensor_count, filter_count) - number of contacts per pair
            - contact_start_indices: shape (sensor_count, filter_count) - buffer start indices
        """
        # Lazy initialization of buffers
        # Note: forces and separations are shape (N, 1) not (N,) to match PhysX API
        # Note: uint32 arrays are created as pure Warp arrays since PyTorch doesn't handle uint32 well
        if self._contact_forces_buffer is None or self._contact_forces_buffer.shape[0] != max_contact_data_count:
            self._contact_forces_buffer, _ = self._frontend.create_tensor((max_contact_data_count, 1), float32)
            self._contact_points_buffer, _ = self._frontend.create_tensor((max_contact_data_count, 3), float32)
            self._contact_normals_buffer, _ = self._frontend.create_tensor((max_contact_data_count, 3), float32)
            self._contact_separations_buffer, _ = self._frontend.create_tensor((max_contact_data_count, 1), float32)
            # Create uint32 arrays directly as Warp arrays (PyTorch doesn't support uint32 well)
            self._contact_counts = wp.zeros(
                (self.sensor_count, self.filter_count), dtype=wp.uint32, device=self._model.device
            )
            self._contact_start_indices = wp.zeros(
                (self.sensor_count, self.filter_count), dtype=wp.uint32, device=self._model.device
            )

        # Zero out buffers
        zero_tensor(self._contact_forces_buffer)
        zero_tensor(self._contact_points_buffer)
        zero_tensor(self._contact_normals_buffer)
        zero_tensor(self._contact_separations_buffer)
        self._contact_counts.zero_()
        self._contact_start_indices.zero_()

        state = self._newton_stage.state_0
        model = self._model

        # Check if we have contact data
        contacts = self._newton_stage.contacts

        # If no contacts available or not populated, return empty data
        if contacts is None or not hasattr(contacts, "force") or not hasattr(contacts, "normal"):
            return (
                self._contact_forces_buffer,
                self._contact_points_buffer,
                self._contact_normals_buffer,
                self._contact_separations_buffer,
                self._contact_counts,
                self._contact_start_indices,
            )

        if model.rigid_contact_max > 0:
            # Convert frontend tensors to Warp arrays for kernel launch
            forces_wp = self._to_warp_array(self._contact_forces_buffer)
            points_wp = self._to_warp_array(self._contact_points_buffer)
            normals_wp = self._to_warp_array(self._contact_normals_buffer)
            separations_wp = self._to_warp_array(self._contact_separations_buffer)
            # counts and indices are already pure Warp uint32 arrays
            counts_wp = self._contact_counts
            indices_wp = self._contact_start_indices

            # PASS 1: Count contacts per sensor-filter pair
            wp.launch(
                kernel=count_contacts_per_pair_kernel,
                dim=model.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count if hasattr(contacts, "rigid_contact_count") else contacts.n_contacts,
                    contacts.pair,  # MuJoCo uses pair instead of rigid_contact_shape0/1
                    model.shape_body,
                    self._body_sensor_map,
                    self._body_filter_map,
                ],
                outputs=[counts_wp],
                device=model.device,
            )
            wp.synchronize_device(model.device)

            # PASS 2: Exclusive scan to compute start indices
            counts_np = counts_wp.numpy().astype(np.uint32).reshape(-1)
            indices_np = np.zeros_like(counts_np, dtype=np.uint32)
            indices_np[1:] = np.cumsum(counts_np[:-1], dtype=np.uint32)

            # Copy back to device
            indices_wp_flat = wp.array(indices_np, dtype=wp.uint32, device=model.device)
            wp.copy(indices_wp, indices_wp_flat.reshape(indices_wp.shape))

            # Reset counts for second pass
            counts_wp.zero_()

            # PASS 3: Fill contact data with proper offsets
            wp.launch(
                kernel=contact_data_kernel,
                dim=model.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count if hasattr(contacts, "rigid_contact_count") else contacts.n_contacts,
                    contacts.pair,  # MuJoCo uses pair instead of rigid_contact_shape0/1
                    contacts.rigid_contact_point0 if hasattr(contacts, "rigid_contact_point0") else contacts.position,
                    contacts.rigid_contact_point1 if hasattr(contacts, "rigid_contact_point1") else contacts.position,
                    contacts.normal,  # Normal from solver (vec3)
                    contacts.force,  # Pre-computed scalar force magnitude from solver
                    (
                        contacts.rigid_contact_thickness0
                        if hasattr(contacts, "rigid_contact_thickness0")
                        else wp.zeros(model.rigid_contact_max, dtype=wp.float32, device=model.device)
                    ),
                    (
                        contacts.rigid_contact_thickness1
                        if hasattr(contacts, "rigid_contact_thickness1")
                        else wp.zeros(model.rigid_contact_max, dtype=wp.float32, device=model.device)
                    ),
                    model.shape_body,
                    state.body_q,
                    self._body_sensor_map,
                    self._body_filter_map,
                    max_contact_data_count,
                ],
                outputs=[
                    forces_wp,
                    points_wp,
                    normals_wp,
                    separations_wp,
                    counts_wp,
                    indices_wp,
                ],
                device=model.device,
            )
            wp.synchronize_device(model.device)

        # Convert uint32 Warp arrays to frontend tensors for compatibility with API layer
        # PyTorch doesn't support uint32 indexing, so convert to int32 for PyTorch
        try:
            import torch

            if isinstance(self._contact_forces_buffer, torch.Tensor):
                # Convert uint32 to int32 since PyTorch doesn't support uint32 indexing
                counts_frontend = wp.to_torch(self._contact_counts).to(torch.int32)
                indices_frontend = wp.to_torch(self._contact_start_indices).to(torch.int32)
            else:
                counts_frontend = self._contact_counts
                indices_frontend = self._contact_start_indices
        except (ImportError, AttributeError):
            counts_frontend = self._contact_counts
            indices_frontend = self._contact_start_indices

        return (
            self._contact_forces_buffer,
            self._contact_points_buffer,
            self._contact_normals_buffer,
            self._contact_separations_buffer,
            counts_frontend,
            indices_frontend,
        )

    def _to_warp_array(self, tensor: wp.array | np.ndarray) -> wp.array:
        """Convert frontend tensor to Warp array for kernel launch.

        Args:
            tensor: Frontend tensor to convert.

        Returns:
            Warp array representation.
        """
        if isinstance(tensor, wp.array):
            return tensor
        # For PyTorch/NumPy tensors, get underlying data pointer
        # This assumes the frontend properly exposes warp arrays
        if hasattr(tensor, "_warp_array"):
            return tensor._warp_array
        # Fallback: try to get the underlying array
        return tensor

    def check(self) -> bool:
        """Check if the view is valid and has sensors.

        Returns:
            True if view has a valid backend and at least one sensor.
        """
        return self._backend is not None and self.count > 0
