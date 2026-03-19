# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.app
import omni.kit.test
import omni.timeline
import omni.usd
import warp as wp
from isaacsim.core.experimental.prims import BufferDtype
from pxr import UsdUtils


class TestPrimDataReaderInterface(omni.kit.test.AsyncTestCase):
    """Tests for the IPrimDataReader C++ Carbonite interface and pybind11 bindings."""

    async def setUp(self):
        """Acquire a fresh reader and initialize it for CPU."""
        self.reader = None
        from isaacsim.core.experimental.prims.impl.extension import get_prim_data_reader

        self.reader = get_prim_data_reader()
        if self.reader is not None:
            self.reader.initialize(0, -1)

    async def tearDown(self):
        """Shut down the reader to clean up all views."""
        if self.reader is not None:
            self.reader.shutdown()

    async def test_acquire_interface(self):
        """Verify the Carbonite interface can be acquired and released."""
        from isaacsim.core.experimental.prims import _prims_reader

        reader = _prims_reader.acquire_prim_data_reader_interface()
        self.assertIsNotNone(reader)
        _prims_reader.release_prim_data_reader_interface(reader)

    async def test_get_prim_data_reader_singleton(self):
        """Verify get_prim_data_reader returns a non-None singleton."""
        self.assertIsNotNone(self.reader)

    # -- View creation and type safety --

    async def test_create_articulation_view_has_correct_methods(self):
        """IArticulationDataView should expose DOF getters and inherited XformPrim getters."""
        view = self.reader.create_articulation_view("art_v", ["/World/test"], "physx")
        self.assertIsNotNone(view)
        for method in [
            "get_dof_positions",
            "get_dof_velocities",
            "get_dof_efforts",
            "get_root_transforms",
            "get_root_velocities",
            "get_world_positions",
            "get_world_orientations",
            "get_dof_index",
            "get_dof_names",
            "get_dof_types",
            "get_dof_types_host",
            "update",
            "allocate_buffer",
            "get_buffer_ptr",
            "get_buffer_size",
            "get_buffer_device",
            "register_field_callback",
        ]:
            self.assertTrue(hasattr(view, method), f"Missing method: {method}")

    async def test_create_rigid_body_view_has_correct_methods(self):
        """IRigidBodyDataView should expose velocity/mass getters and inherited XformPrim getters."""
        view = self.reader.create_rigid_body_view("rb_v", ["/World/test"], "physx")
        self.assertIsNotNone(view)
        for method in [
            "get_linear_velocities",
            "get_angular_velocities",
            "get_world_positions",
            "update",
            "allocate_buffer",
        ]:
            self.assertTrue(hasattr(view, method), f"Missing method: {method}")
        self.assertFalse(hasattr(view, "get_dof_positions"))

    async def test_create_xform_view_has_correct_methods(self):
        """IXformDataView should expose transform getters only."""
        view = self.reader.create_xform_view("xf_v", ["/World/test"], "physx")
        self.assertIsNotNone(view)
        for method in [
            "get_world_positions",
            "get_world_orientations",
            "get_local_translations",
            "get_local_orientations",
            "get_local_scales",
            "update",
            "allocate_buffer",
        ]:
            self.assertTrue(hasattr(view, method), f"Missing method: {method}")
        self.assertFalse(hasattr(view, "get_linear_velocities"))
        self.assertFalse(hasattr(view, "get_dof_positions"))

    # -- Buffer allocation and pointer access --

    async def test_buffer_allocation_returns_valid_pointer(self):
        """allocate_buffer should produce a non-zero pointer and correct size/device."""
        view = self.reader.create_articulation_view("buf_test", ["/World/t"], "physx")
        view.allocate_buffer("field_a", 100, BufferDtype.FLOAT)

        ptr = view.get_buffer_ptr("field_a")
        self.assertIsInstance(ptr, int)
        self.assertGreater(ptr, 0)

        self.assertEqual(view.get_buffer_size("field_a"), 100)
        self.assertEqual(view.get_buffer_device(), -1)

    async def test_buffer_ptr_for_unknown_field_returns_zero(self):
        """Querying a non-existent field should return 0."""
        view = self.reader.create_articulation_view("buf_miss", ["/World/t"], "physx")
        self.assertEqual(view.get_buffer_ptr("no_such_field"), 0)
        self.assertEqual(view.get_buffer_size("no_such_field"), 0)

    async def test_buffer_wrapping_as_warp_array(self):
        """C++ buffer should be wrappable as a wp.array via the ptr constructor."""
        view = self.reader.create_articulation_view("wp_wrap", ["/World/t"], "physx")
        view.allocate_buffer("test_field", 30, BufferDtype.FLOAT)

        ptr = view.get_buffer_ptr("test_field")
        device_ord = view.get_buffer_device()
        device = f"cuda:{device_ord}" if device_ord >= 0 else "cpu"

        arr = wp.array(
            ptr=ptr,
            shape=(5, 6),
            dtype=wp.float32,
            device=device,
            capacity=30 * 4,
            deleter=None,
        )
        self.assertEqual(arr.shape, (5, 6))
        self.assertEqual(arr.dtype, wp.float32)

    async def test_allocate_buffer_uint8_via_dtype(self):
        """allocate_buffer(field_name, count, dtype='uint8') creates uint8 buffer."""
        view = self.reader.create_articulation_view("buf_u8", ["/World/t"], "physx")
        self.assertTrue(view.allocate_buffer("dof_types", 5, BufferDtype.UINT8))
        self.assertEqual(view.get_buffer_size("dof_types"), 5)
        self.assertGreater(view.get_buffer_ptr("dof_types"), 0)

    async def test_allocate_buffer_accepts_buffer_dtype_enum(self):
        """allocate_buffer accepts BufferDtype enum as well as string."""
        view = self.reader.create_articulation_view("buf_enum", ["/World/t"], "physx")
        self.assertTrue(view.allocate_buffer("field_f", 10, BufferDtype.FLOAT))
        self.assertEqual(view.get_buffer_size("field_f"), 10)
        self.assertTrue(view.allocate_buffer("field_u8", 5, BufferDtype.UINT8))
        self.assertEqual(view.get_buffer_size("field_u8"), 5)

    async def test_allocate_buffer_invalid_dtype_raises(self):
        """allocate_buffer with unsupported dtype string should raise."""
        view = self.reader.create_articulation_view("buf_bad", ["/World/t"], "physx")
        with self.assertRaises(ValueError):
            view.allocate_buffer("x", 10, "int32")

    # -- DOF names and types --

    async def test_get_dof_names_returns_list(self):
        """get_dof_names should return a list (may be empty if no articulation/DOFs)."""
        view = self.reader.create_articulation_view("dof_meta_v", ["/World/test"], "physx")
        self.assertIsNotNone(view)
        names = view.get_dof_names()
        self.assertIsInstance(names, list)

    async def test_get_dof_types_returns_tuple_ptr_count(self):
        """get_dof_types should return (ptr, count) like other buffer getters."""
        view = self.reader.create_articulation_view("dof_types_v", ["/World/test"], "physx")
        self.assertIsNotNone(view)
        result = view.get_dof_types()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        ptr, count = result
        self.assertIsInstance(ptr, int)
        self.assertIsInstance(count, int)

    async def test_reader_has_set_articulation_dof_metadata(self):
        """Reader should expose set_articulation_dof_metadata for Newton backend."""
        self.assertTrue(hasattr(self.reader, "set_articulation_dof_metadata"))

    async def test_set_articulation_dof_metadata_then_get_dof_names_and_types(self):
        """After set_articulation_dof_metadata, get_dof_names and get_dof_types return that data (Newton path)."""
        import ctypes

        view_id = "dof_newton_meta"
        view = self.reader.create_articulation_view(view_id, ["/World/t"], "newton")
        self.assertIsNotNone(view)
        self.reader.set_articulation_dof_metadata(view_id, ["joint_a", "joint_b"], [0, 1])
        names = view.get_dof_names()
        ptr, count = view.get_dof_types()
        self.assertEqual(names, ["joint_a", "joint_b"])
        self.assertEqual(count, 2)
        if ptr and count >= 2:
            buf = (ctypes.c_uint8 * count).from_address(ptr)
            self.assertEqual(buf[0], 0)
            self.assertEqual(buf[1], 1)

    # -- Callback registration and invocation --

    async def test_callback_invoked_on_first_getter_call(self):
        """A registered callback should fire on the first getter call."""
        view = self.reader.create_articulation_view("cb_test", ["/World/t"], "newton")
        view.allocate_buffer("dof_positions", 10, BufferDtype.FLOAT)

        call_count = [0]

        def counter_callback():
            call_count[0] += 1

        view.register_field_callback("dof_positions", counter_callback)
        view.get_dof_positions()
        self.assertEqual(call_count[0], 1)

    async def test_callback_not_invoked_twice_same_step(self):
        """Within the same physics step, the callback should fire at most once."""
        view = self.reader.create_articulation_view("cb_dedup", ["/World/t"], "newton")
        view.allocate_buffer("dof_positions", 10, BufferDtype.FLOAT)

        call_count = [0]
        view.register_field_callback("dof_positions", lambda: call_count.__setitem__(0, call_count[0] + 1))

        view.get_dof_positions()
        view.get_dof_positions()
        self.assertEqual(call_count[0], 1)

    async def test_update_batch_prefetch_triggers_all_callbacks(self):
        """update() should trigger callbacks for all stale fields."""
        view = self.reader.create_articulation_view("cb_batch", ["/World/t"], "newton")
        view.allocate_buffer("dof_positions", 10, BufferDtype.FLOAT)
        view.allocate_buffer("dof_velocities", 10, BufferDtype.FLOAT)

        calls = {"pos": 0, "vel": 0}
        view.register_field_callback("dof_positions", lambda: calls.__setitem__("pos", calls["pos"] + 1))
        view.register_field_callback("dof_velocities", lambda: calls.__setitem__("vel", calls["vel"] + 1))

        view.update()
        self.assertEqual(calls["pos"], 1)
        self.assertEqual(calls["vel"], 1)

        # Subsequent getters should not re-trigger
        view.get_dof_positions()
        view.get_dof_velocities()
        self.assertEqual(calls["pos"], 1)
        self.assertEqual(calls["vel"], 1)

    # -- View lifecycle --

    async def test_view_creation_and_removal(self):
        """Views can be created and then cleanly removed."""
        self.reader.create_articulation_view("lifecycle_a", ["/World/t"], "physx")
        self.reader.create_rigid_body_view("lifecycle_b", ["/World/t"], "physx")
        self.reader.remove_view("lifecycle_a")
        self.reader.remove_view("lifecycle_b")

    async def test_shutdown_cleans_all_views(self):
        """shutdown() should destroy all views; subsequent create works after re-initialize."""
        self.reader.create_articulation_view("sd_test", ["/World/t"], "physx")
        self.reader.shutdown()
        self.reader.initialize(0, -1)
        view = self.reader.create_articulation_view("sd_test2", ["/World/t"], "physx")
        self.assertIsNotNone(view)

    async def test_manager_ensure_initialized_is_idempotent_for_same_stage(self):
        """Repeated ensure_initialized(stage, device) should not bump generation."""
        from isaacsim.core.experimental.prims import _prims_reader

        await stage_utils.create_new_stage_async()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        await omni.kit.app.get_app().next_update_async()
        stage = omni.usd.get_context().get_stage()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        self.assertNotEqual(stage_id, 0)

        manager = _prims_reader.acquire_prim_data_reader_manager_interface()
        self.assertIsNotNone(manager)
        try:
            self.assertTrue(manager.ensure_initialized(stage_id, -1))
            gen_after_first = manager.get_generation()
            self.assertGreater(gen_after_first, 0)

            self.assertTrue(manager.ensure_initialized(stage_id, -1))
            gen_after_second = manager.get_generation()
            self.assertEqual(gen_after_first, gen_after_second)
        finally:
            _prims_reader.release_prim_data_reader_manager_interface(manager)
            timeline.stop()
            await omni.kit.app.get_app().next_update_async()

    async def test_manager_ensure_initialized_is_stable_across_multiple_acquirers(self):
        """Multiple manager callers should share one initialization generation for a stage."""
        from isaacsim.core.experimental.prims import _prims_reader

        await stage_utils.create_new_stage_async()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        await omni.kit.app.get_app().next_update_async()
        stage = omni.usd.get_context().get_stage()
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        self.assertNotEqual(stage_id, 0)

        manager_a = _prims_reader.acquire_prim_data_reader_manager_interface()
        manager_b = _prims_reader.acquire_prim_data_reader_manager_interface()
        self.assertIsNotNone(manager_a)
        self.assertIsNotNone(manager_b)
        try:
            self.assertTrue(manager_a.ensure_initialized(stage_id, -1))
            generation_after_first = manager_a.get_generation()
            self.assertGreater(generation_after_first, 0)

            self.assertTrue(manager_b.ensure_initialized(stage_id, -1))
            generation_after_second = manager_b.get_generation()
            self.assertEqual(generation_after_first, generation_after_second)
        finally:
            _prims_reader.release_prim_data_reader_manager_interface(manager_b)
            _prims_reader.release_prim_data_reader_manager_interface(manager_a)
            timeline.stop()
            await omni.kit.app.get_app().next_update_async()
