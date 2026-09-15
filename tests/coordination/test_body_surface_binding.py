"""The male/female body surface deforms with the body, like the skin it is.

It arrives from the sex morph rather than from an STL layer, so it was never
handed to the skinning: posed into a squat it stood still, 0.00 units of
movement, while the BP3D skin beside it moved 72.91.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from faceforge.coordination.demand_loaders import register_body_surface


class FakeGeometry:
    def __init__(self, n: int = 6) -> None:
        self.positions = np.zeros(n * 3, dtype=np.float32)
        self.vertex_count = n


class FakeMesh:
    def __init__(self) -> None:
        self.name = "body_surface"
        self.geometry = FakeGeometry()
        self.rest_positions = None


class FakeSkinning:
    def __init__(self) -> None:
        self.bindings: list = []
        self.calls: list[dict] = []
        self.muscle_field = object()          # already attached

    def register_skin_mesh(self, mesh, **kw):
        self.calls.append(kw)
        self.bindings.append(SimpleNamespace(mesh=mesh))


def _morph(mesh, loaded: bool = True):
    return SimpleNamespace(body_mesh=mesh, loaded=loaded)


CHAINS = {"spine": 0, "arm_R": 1}


def test_the_surface_is_bound_to_every_chain():
    mesh, skinning = FakeMesh(), FakeSkinning()
    assert register_body_surface(skinning, _morph(mesh), CHAINS) is True
    assert [b.mesh for b in skinning.bindings] == [mesh]
    kw = skinning.calls[0]
    assert kw["is_muscle"] is False
    assert kw["allowed_chains"] == {0, 1}
    # The same two-tier spatial filter the skin binds with.
    assert kw["chain_z_margin"] > 0 and kw["spatial_limit"] > 0


def test_binding_twice_does_not_bind_twice():
    mesh, skinning = FakeMesh(), FakeSkinning()
    morph = _morph(mesh)
    register_body_surface(skinning, morph, CHAINS)
    assert register_body_surface(skinning, morph, CHAINS) is True
    assert len(skinning.bindings) == 1


def test_nothing_happens_without_a_loaded_morph_or_chains():
    mesh = FakeMesh()
    assert register_body_surface(FakeSkinning(), _morph(mesh, loaded=False), CHAINS) is False
    assert register_body_surface(FakeSkinning(), _morph(mesh), {}) is False
    assert register_body_surface(FakeSkinning(), _morph(None), CHAINS) is False
    assert register_body_surface(None, _morph(mesh), CHAINS) is False


def test_a_failed_solve_is_reported_not_raised():
    class Angry(FakeSkinning):
        def register_skin_mesh(self, mesh, **kw):
            raise ValueError("no chains reach it")

    assert register_body_surface(Angry(), _morph(FakeMesh()), CHAINS) is False
