"""Load anatomy groups the first time the user asks to see them.

Most of the body is not loaded at startup.  Organs, vasculature, the brain,
skin, six regional muscle groups, ligaments and half a dozen smaller groups are
each loaded the first time their layer checkbox is ticked, because loading all
of them costs tens of seconds and most sessions want two or three.

Every loader here has the same five-part shape, and the shape is the point:

1. **Load-once guard.**  A layer already loaded returns immediately.  The guard
   is set *before* the load runs, so a load that raises is not retried on every
   subsequent tick of the checkbox.
2. **Attach** the loaded group under its parent scene node and register it with
   the visibility manager.
3. **Match the render mode** already on screen
   (:mod:`faceforge.coordination.render_mode_sync`).
4. **Register for deformation** with soft-tissue skinning, naming the kinematic
   chains that group is allowed to follow.  This is the part that is genuinely
   per-group anatomy rather than boilerplate -- see "Chain assignment" below.
5. **Announce** the individual structures on the event bus so the layers tab can
   build per-structure toggles, and run the post-registration hooks.

Every loader swallows its own exceptions into a warning.  That is deliberate:
a missing STL for the pelvic floor must not take down a session that is looking
at the skull.  ``LoadingPipeline.report`` is where startup-critical failures are
recorded; these are not startup-critical.

Chain assignment
----------------
A skinned mesh follows a set of *kinematic chains* (spine, ribs, arm_R, leg_L,
one chain per digit...).  Getting the set wrong is visible as a muscle that
follows the wrong limb, so the mapping is data here rather than inference:

* :data:`MUSCLE_CHAIN_MAP` gives each muscle region its default chains.
* :data:`MUSCLE_CHAIN_OVERRIDES` names the individual muscles that differ --
  chiefly muscles that span from spine or ribs *to* the humerus or scapula and
  so must follow the arm, and rib-cage muscles that must *not*.
* ``"arm"``/``"leg"``/``"hand"``/``"foot"`` are side-neutral tokens resolved to
  the structure's own side at registration time by :func:`resolve_sided_chains`,
  so ``"Latissimus Dorsi R"`` binds to ``arm_R`` and not to both arms.

Hand and foot muscles bind to digit chains *only*.  Digit pivots are children
of the wrist/ankle pivots, so a digit chain's delta already contains all the
arm movement; including the limb chain as well double-counts it and lets an
extrapolated wrist segment fight the digit joints across the palm.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

from faceforge.coordination.render_mode_sync import apply_current_render_mode
from faceforge.core.config_loader import load_config, load_muscle_config
from faceforge.core.events import EventType

logger = logging.getLogger(__name__)


#: Layer id -> muscle config filename, for the six regional muscle groups.
MUSCLE_REGIONS: dict[str, str] = {
    "back_muscles": "back_muscles.json",
    "shoulder_muscles": "shoulder_muscles.json",
    "arm_muscles": "arm_muscles.json",
    "torso_muscles": "torso_muscles.json",
    "hip_muscles": "hip_muscles.json",
    "leg_muscles": "leg_muscles.json",
}

#: Muscle region -> default skinning chains (side-neutral tokens allowed).
MUSCLE_CHAIN_MAP: dict[str, list[str]] = {
    "back_muscles":     ["spine", "ribs"],
    "torso_muscles":    ["spine", "ribs"],
    "shoulder_muscles": ["spine", "arm"],
    "arm_muscles":      ["spine", "arm", "hand"],
    "hip_muscles":      ["spine", "leg"],
    "leg_muscles":      ["spine", "leg", "foot"],
}

#: Ligament category -> chains the ligament follows.
LIGAMENT_CHAIN_MAP: dict[str, list[str]] = {
    "upper_limb": ["spine", "arm"],
    "lower_limb": ["spine", "leg"],
    "trunk":      ["spine", "ribs"],
    "hip":        ["spine", "leg"],
}

#: Single-STL integument toggles: layer id -> FMA-coded STL basename.
INTEGUMENT_STLS: dict[str, str] = {
    "head_hair": "FMA70751",
    "pubic_hair": "FMA70754",
    "epicranial_aponeurosis": "FMA46768",
    "spinal_central_canal": "FMA78497",
}


def _build_muscle_chain_overrides() -> dict[str, list[str]]:
    """Per-muscle chain overrides, keyed by the muscle's config name.

    A muscle not listed here uses its region's default from
    :data:`MUSCLE_CHAIN_MAP`.  Anatomical rationale: ``spine+ribs`` is a rib
    cage structure that must not follow arm movement; ``spine+ribs+arm`` spans
    from the spine or ribs to the humerus or scapula and must.
    """
    out: dict[str, list[str]] = {}

    def sided(names: Iterable[str], chains: list[str]) -> None:
        for name in names:
            for side in ("R", "L"):
                out[f"{name} {side}"] = list(chains)

    # shoulder_muscles: rib-only muscles
    sided(("Serratus Ant.", "Subclavius"), ["spine", "ribs"])
    # torso_muscles: pectorals insert on the humerus, so they need the arm chain
    sided(("Pect. Major Clav.", "Pect. Major Stern.", "Pect. Major Abd.",
           "Pect. Minor"), ["spine", "ribs", "arm"])
    # torso_muscles: pure rib/spine structures (no arm follow), midline names
    for name in ("Ext. Intercostal", "Int. Intercostal", "Innermost Intercostal",
                 "Diaphragm", "Linea Alba"):
        out[name] = ["spine", "ribs"]
    sided(("Trans. Thoracis", "Lev. Costarum Longi", "Lev. Costarum Breves"),
          ["spine", "ribs"])
    # back_muscles: muscles that connect to shoulder/arm
    sided(("Asc. Trapezius", "Trans. Trapezius", "Desc. Trapezius",
           "Latissimus Dorsi", "Rhomboid Major", "Rhomboid Minor"),
          ["spine", "ribs", "arm"])
    # back_muscles: rib-attached posterior serratus
    sided(("Serratus Post. Sup.", "Serratus Post. Inf."), ["spine", "ribs"])
    return out


#: Muscle config name -> chains, overriding the region default.
MUSCLE_CHAIN_OVERRIDES: dict[str, list[str]] = _build_muscle_chain_overrides()


def resolve_chain_set(chain_names: Iterable[str],
                      chain_ids: dict[str, int]) -> set[int] | None:
    """Chain names -> chain id set, dropping names no chain was built for.

    Returns ``None`` rather than an empty set when nothing resolved, because
    ``allowed_chains=None`` means "no restriction" to the skinning registration
    while ``set()`` would mean "follow nothing" and freeze the mesh.
    """
    chains = {chain_ids[name] for name in chain_names if name in chain_ids}
    return chains or None


def resolve_sided_chains(chain_names: Iterable[str], structure_name: str,
                         chain_ids: dict[str, int]) -> set[int] | None:
    """Resolve side-neutral chain tokens to the structure's own side.

    A name ending in ``" R"`` or ``" L"`` binds only to that side's limb chain
    (``"arm"`` -> ``"arm_R"``).  A midline structure has no side, and binds to
    both as a fallback.  ``"hand"``/``"foot"`` expand to all five digit chains
    per side.
    """
    side = None
    if structure_name.endswith(" R"):
        side = "R"
    elif structure_name.endswith(" L"):
        side = "L"

    resolved: list[str] = []
    for name in chain_names:
        if name in ("arm", "leg"):
            if side is not None:
                resolved.append(f"{name}_{side}")
            else:
                resolved.append(f"{name}_R")
                resolved.append(f"{name}_L")
        elif name in ("hand", "foot"):
            for s in ([side] if side is not None else ["R", "L"]):
                for digit in range(1, 6):
                    resolved.append(f"{name}_{s}_{digit}")
        else:
            resolved.append(name)
    return resolve_chain_set(resolved, chain_ids)


def digit_chain_ids(prefix: str, chain_ids: dict[str, int]) -> set[int]:
    """Every digit chain id for ``"hand"`` or ``"foot"``, both sides."""
    out: set[int] = set()
    for side in ("R", "L"):
        for digit in range(1, 6):
            cid = chain_ids.get(f"{prefix}_{side}_{digit}")
            if cid is not None:
                out.add(cid)
    return out


class DemandLoaders:
    """The on-demand loaders for one application context.

    Holds the load-once state, so "has the skin been loaded?" is an attribute
    of an object a test can inspect rather than a closure variable nobody can
    reach.  Deliberately Qt-free: every method here is exercisable with stub
    collaborators.
    """

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        #: Layer ids whose load has been attempted (success or failure).
        self.loaded: set[str] = set()

    # -- helpers ---------------------------------------------------------

    def _claim(self, layer: str) -> bool:
        """Claim *layer* for loading.  False if it was already claimed.

        The claim is recorded before the load runs, so a load that raises is
        not retried every time the checkbox is ticked.
        """
        if layer in self.loaded:
            return False
        self.loaded.add(layer)
        return True

    @property
    def _skinning(self) -> Any:
        sim = self.ctx.simulation
        return getattr(sim, "soft_tissue", None) if sim is not None else None

    def _announce(self, layer: str, items: list[dict]) -> None:
        self.ctx.event_bus.publish(
            EventType.STRUCTURES_LOADED, group_id=layer, items=items)

    def _register_items(self, layer: str, nodes: Iterable[Any],
                        defs: Iterable[dict], *,
                        toggle_prefix: str | None = None,
                        extra_keys: Iterable[str] = ()) -> list[dict]:
        """Register per-structure visibility toggles; return the item dicts.

        ``strict=True`` is deliberate: callers must pass ``result.defs_loaded``,
        not the definition list they handed to the loader.  A failed STL yields
        no node, so the two lists differ in length and every structure after
        the failure would otherwise be registered under another one's name.
        Raising here is far better than silently mislabelling the anatomy.
        """
        items: list[dict] = []
        for node, defn in zip(nodes, defs, strict=True):
            name = defn.get("name", node.name)
            tid = f"{toggle_prefix}{name}" if toggle_prefix else f"{layer}_{name}"
            self.ctx.visibility.register(tid, node)
            item = {"toggle_id": tid, "name": name}
            for key in extra_keys:
                item[key] = defn.get(key, "")
            items.append(item)
        return items

    def _attach(self, layer: str, result: Any,
                parent_key: str = "bodyRoot") -> Any:
        """Attach a load result under its parent and match the render mode."""
        parent = self.ctx.node(parent_key)
        if parent is None:
            return None
        parent.add(result.group)
        self.ctx.visibility.register(layer, result.group)
        apply_current_render_mode(self.ctx.scene, result.meshes)
        return parent

    # -- regional body muscles -------------------------------------------

    def load_body_muscle_region(self, layer: str, config_name: str) -> None:
        """Load one regional muscle group, with per-muscle chain assignment."""
        if not self._claim(layer):
            return
        if self.ctx.node("bodyRoot") is None:
            return
        try:
            result = self.ctx.assets.load_body_muscles(config_name)
            self._attach(layer, result)
            defs = load_muscle_config(config_name)
            default_chains = MUSCLE_CHAIN_MAP.get(layer, ["spine"])

            skinning = self._skinning
            if skinning is not None:
                self._register_muscles(layer, result, defs, default_chains,
                                       skinning)

            physiology = getattr(self.ctx.simulation, "physiology", None)
            if physiology is not None:
                physiology.muscle_groups.append(result.group)
                for mesh, defn in zip(result.meshes, result.defs_loaded,
                                      strict=True):
                    physiology.register_muscle(mesh, defn.get("name", mesh.name))

            if layer == "back_muscles" and skinning is not None:
                self._wire_back_neck_muscles(defs, skinning)

            items = self._register_items(
                layer, result.nodes, result.defs_loaded, toggle_prefix=f"muscle_{layer}_")
            self._announce(layer, items)
            logger.info("Loaded body muscles: %s (%d meshes)",
                        layer, len(result.meshes))
            self.ctx.run_after_registration_hooks()
        except Exception as e:  # noqa: BLE001 - one bad group must not end the session
            logger.warning("Failed to load %s: %s", layer, e)

    def _register_muscles(self, layer: str, result: Any, defs: list[dict],
                          default_chains: list[str], skinning: Any) -> None:
        chain_ids = self.ctx.skin_chain_ids
        for mesh, defn in zip(result.meshes, result.defs_loaded, strict=True):
            muscle_name = defn.get("name", mesh.name)
            chain_names = (MUSCLE_CHAIN_OVERRIDES.get(muscle_name)
                           or default_chains)
            allowed = resolve_sided_chains(chain_names, muscle_name, chain_ids)
            skinning.register_skin_mesh(
                mesh, is_muscle=True, allowed_chains=allowed,
                head_follow_config=defn.get("headFollow"),
                muscle_name=muscle_name,
                # Forbid opposite-side bones. `ribs` is one unsided chain, so
                # allowed_chains cannot express this on its own.
                side=("R" if muscle_name.endswith(" R")
                      else "L" if muscle_name.endswith(" L") else None),
            )
            origin = defn.get("originBones")
            insertion = defn.get("insertionBones")
            if origin and insertion and skinning.attachment_system is not None:
                skinning.attachment_system.register_muscle(
                    skinning.bindings[-1], origin, insertion,
                    fascia_regions=defn.get("fasciaRegions", []),
                    # Optional per-muscle overrides; absent means the module
                    # global applies, so an unsourced muscle keeps today's
                    # behaviour rather than getting an invented value.
                    max_stretch=defn.get("maxStretch"),
                    pin_strength=defn.get("pinStrength"),
                )

        # Arm and leg muscles: remove digit/limb cross-chain blending.  Digit
        # pivots are children of wrist/ankle pivots, so a digit chain's delta
        # already includes the limb movement and blending double-counts it.
        if layer in ("arm_muscles", "leg_muscles"):
            prefix = "hand" if layer == "arm_muscles" else "foot"
            digit_cids = digit_chain_ids(prefix, chain_ids)
            if digit_cids:
                skinning.snap_hierarchy_blends(digit_cids)
                skinning.reassign_orphan_vertices(digit_cids)

    def _wire_back_neck_muscles(self, defs: list[dict], skinning: Any) -> None:
        from faceforge.anatomy.back_neck_muscles import BackNeckMuscleHandler

        handler = BackNeckMuscleHandler()
        handler.register(skinning, {d["name"]: d for d in defs if "name" in d})
        if handler.registered:
            if self.ctx.simulation.fascia is not None:
                handler.set_fascia_system(self.ctx.simulation.fascia)
            self.ctx.simulation.back_neck_muscles = handler

    # -- hand / foot muscles ---------------------------------------------

    def _load_digit_muscles(self, layer: str, loader: Callable[[], Any],
                            config_name: str, prefix: str) -> None:
        if not self._claim(layer):
            return
        if self.ctx.node("bodyRoot") is None:
            return
        try:
            result = loader()
            self._attach(layer, result)
            defs = load_muscle_config(config_name)
            items = self._register_items(
                layer, result.nodes, result.defs_loaded, toggle_prefix=f"muscle_{layer}_")
            self._announce(layer, items)

            skinning = self._skinning
            if skinning is not None:
                chains = digit_chain_ids(prefix, self.ctx.skin_chain_ids)
                for mesh in result.meshes:
                    skinning.register_skin_mesh(
                        mesh, is_muscle=True, allowed_chains=chains or None)
            logger.info("Loaded %s: %d meshes", layer, len(result.meshes))
            self.ctx.run_after_registration_hooks()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", layer, e)

    def load_hand_muscles(self) -> None:
        self._load_digit_muscles(
            "hand_muscles", lambda: self.ctx.assets.load_hand_muscles(),
            "hand_muscles.json", "hand")

    def load_foot_muscles(self) -> None:
        self._load_digit_muscles(
            "foot_muscles", lambda: self.ctx.assets.load_foot_muscles(),
            "foot_muscles.json", "foot")

    # -- spine-following soft structures ---------------------------------

    # NOTE on the ``loader`` parameter: every call site passes a lambda rather
    # than a bound ``assets.load_x`` method, so the attribute is resolved inside
    # the guarded block.  A bound-method argument would be looked up before the
    # load-once claim and the parent-node check had run.

    def _load_spine_following(self, layer: str, loader: Callable[[], Any],
                              config_name: str, *, extra_keys: Iterable[str],
                              physiology_group_attr: str | None,
                              physiology_register: str | None,
                              physiology_keys: tuple[str, ...] = ()) -> None:
        """Load a group that follows the spine only (organs, vasculature).

        Organs and vessels do not follow the limbs -- a kidney that swings with
        the arm is a defect -- so they bind to the spine chain alone.
        """
        if not self._claim(layer):
            return
        if self.ctx.node("bodyRoot") is None:
            return
        try:
            result = loader()
            self._attach(layer, result)
            skinning = self._skinning
            if skinning is not None:
                spine_id = self.ctx.skin_chain_ids.get("spine")
                allowed = {spine_id} if spine_id is not None else None
                for mesh in result.meshes:
                    skinning.register_skin_mesh(
                        mesh, is_muscle=False, allowed_chains=allowed)
            defs = load_config(config_name)
            items = self._register_items(
                layer, result.nodes, result.defs_loaded,
                toggle_prefix=("organ_" if layer == "organs" else "vasc_"),
                extra_keys=extra_keys)
            self._announce(layer, items)

            physiology = getattr(self.ctx.simulation, "physiology", None)
            if physiology is not None and physiology_group_attr:
                setattr(physiology, physiology_group_attr, result.group)
                register = getattr(physiology, physiology_register)
                for mesh, defn in zip(result.meshes, result.defs_loaded,
                                      strict=True):
                    register(mesh, *(defn.get(k, "") for k in physiology_keys))
            logger.info("Loaded %s: %d meshes", layer, len(result.meshes))
            self.ctx.run_after_registration_hooks()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", layer, e)

    def load_organs(self) -> None:
        self._load_spine_following(
            "organs", lambda: self.ctx.assets.load_organs(), "organs.json",
            extra_keys=("category",),
            physiology_group_attr="organ_group",
            physiology_register="register_organ",
            physiology_keys=("name", "category"))

    def load_vasculature(self) -> None:
        self._load_spine_following(
            "vasculature", lambda: self.ctx.assets.load_vasculature(), "vascular.json",
            extra_keys=("type",),
            physiology_group_attr="vascular_group",
            physiology_register="register_vascular",
            physiology_keys=("name", "type"))

    # -- brain ------------------------------------------------------------

    def load_brain(self) -> None:
        """Load the brain under ``brainGroup``.

        The brain hangs off its own group rather than the skull, so it stays
        visible with the skull hidden and follows the head through the explicit
        pivot rotation in ``HeadRotationSystem``.
        """
        if not self._claim("brain"):
            return
        brain_group = self.ctx.node("brainGroup")
        if brain_group is None:
            return
        try:
            result = self.ctx.assets.load_brain()
            brain_group.add(result.group)
            apply_current_render_mode(self.ctx.scene, result.meshes)
            defs = load_config("brain.json")
            items = self._register_items(
                "brain", result.nodes, result.defs_loaded, toggle_prefix="brain_")
            self._announce("brain", items)
            logger.info("Loaded brain: %d meshes", len(result.meshes))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load brain: %s", e)

    # -- ligaments ---------------------------------------------------------

    def load_ligaments(self) -> None:
        if not self._claim("ligaments"):
            return
        if self.ctx.node("bodyRoot") is None:
            return
        try:
            result = self.ctx.assets.load_ligaments()
            self._attach("ligaments", result)
            defs = load_config("ligaments.json")

            skinning = self._skinning
            if skinning is not None:
                for mesh, defn in zip(result.meshes, result.defs_loaded,
                                      strict=True):
                    lig_name = defn.get("name", mesh.name)
                    chain_names = LIGAMENT_CHAIN_MAP.get(
                        defn.get("category", "trunk"), ["spine"])
                    allowed = resolve_sided_chains(
                        chain_names, lig_name, self.ctx.skin_chain_ids)
                    skinning.register_skin_mesh(
                        mesh, is_muscle=True, allowed_chains=allowed,
                        muscle_name=lig_name)
                    origin = defn.get("originBones")
                    insertion = defn.get("insertionBones")
                    if (origin and insertion
                            and skinning.attachment_system is not None):
                        skinning.attachment_system.register_muscle(
                            skinning.bindings[-1], origin, insertion)

            items = self._register_items(
                "ligaments", result.nodes, result.defs_loaded,
                toggle_prefix="ligaments_", extra_keys=("category",))
            self._announce("ligaments", items)
            logger.info("Loaded ligaments: %d meshes", len(result.meshes))
            self.ctx.run_after_registration_hooks()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load ligaments: %s", e)

    # -- skin --------------------------------------------------------------

    def load_skin(self) -> None:
        """Load the body skin surface and bind it to every chain.

        The skin deforms with spine, limbs, digits and breathing, so it binds
        to all chains.  Two-tier spatial filtering keeps that from binding a
        thigh vertex to an arm chain: ``chain_z_margin`` is a per-chain Z-axis
        AABB with a proportional margin (small chains such as hands get tight
        margins), and ``spatial_limit`` is a Euclidean guard that catches the
        remaining overlap where arm and leg chains meet at hip level.
        """
        if not self._claim("skin"):
            return
        if self.ctx.node("bodyRoot") is None:
            return
        try:
            result = self.ctx.assets.load_skin()
            self._attach("skin", result)
            skinning = self._skinning
            if skinning is not None:
                chain_ids = self.ctx.skin_chain_ids
                all_chains = set(chain_ids.values()) if chain_ids else None
                for mesh in result.meshes:
                    skinning.register_skin_mesh(
                        mesh, is_muscle=False, allowed_chains=all_chains,
                        chain_z_margin=15.0, spatial_limit=25.0)
            logger.info("Loaded skin: %d meshes", len(result.meshes))
            self.ctx.run_after_registration_hooks()
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load skin: %s", e)

    # -- generic groups ----------------------------------------------------

    def load_generic_group(self, layer: str, loader: Callable[[], Any],
                           config_name: str,
                           config_loader: Callable[[str], list[dict]] = load_config,
                           *, parent_key: str = "bodyRoot") -> None:
        """Load a group that needs no skinning registration.

        The smaller additional-anatomy groups (pelvic floor, oral cavity,
        additional cardiac and intestinal structures, additional CNS) are
        static: they are attached, made toggleable and announced, and nothing
        deforms them.
        """
        if not self._claim(layer):
            return
        parent = self.ctx.node(parent_key)
        if parent is None:
            return
        try:
            result = loader()
            parent.add(result.group)
            self.ctx.visibility.register(layer, result.group)
            apply_current_render_mode(self.ctx.scene, result.meshes)
            defs = config_loader(config_name)
            items: list[dict] = []
            for node, defn in zip(result.nodes, result.defs_loaded,
                                  strict=True):
                name = defn.get("name", node.name)
                tid = f"{layer}_{name}"
                self.ctx.visibility.register(tid, node)
                item = {"toggle_id": tid, "name": name}
                category = defn.get("category") or defn.get("type")
                if category:
                    item["category"] = category
                items.append(item)
            self._announce(layer, items)
            logger.info("Loaded %s: %d meshes", layer, len(result.meshes))
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", layer, e)

    def load_pelvic_floor(self) -> None:
        self.load_generic_group("pelvic_floor",
                                lambda: self.ctx.assets.load_pelvic_floor(),
                                "pelvic_floor.json")

    def load_oral(self) -> None:
        self.load_generic_group("oral", lambda: self.ctx.assets.load_oral(),
                                "oral.json")

    def load_cardiac_additional(self) -> None:
        self.load_generic_group("cardiac_additional",
                                lambda: self.ctx.assets.load_cardiac_additional(),
                                "cardiac_additional.json")

    def load_intestinal(self) -> None:
        self.load_generic_group("intestinal",
                                lambda: self.ctx.assets.load_intestinal(),
                                "intestinal.json")

    def load_cns_additional(self) -> None:
        self.load_generic_group("cns_additional",
                                lambda: self.ctx.assets.load_cns_additional(),
                                "cns_additional.json", parent_key="brainGroup")

    # -- single STL --------------------------------------------------------

    def load_single_stl(self, layer: str, stl_name: str) -> None:
        """Load one STL file as a standalone toggleable layer."""
        if not self._claim(layer):
            return
        if self.ctx.node("bodyRoot") is None:
            return
        try:
            from faceforge.loaders.stl_batch_loader import load_stl_batch

            result = load_stl_batch(
                [{"name": layer, "stl": stl_name, "color": 0xcccccc}],
                label=layer,
                transform=self.ctx.assets.transform,
                stl_dir=self.ctx.assets.stl_dir,
            )
            self._attach(layer, result)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load %s: %s", layer, e)

    # -- dispatch ----------------------------------------------------------

    def loader_for(self, layer: str) -> Callable[[], None] | None:
        """The zero-argument loader for *layer*, or ``None`` if it has none."""
        table: dict[str, Callable[[], None]] = {
            "organs": self.load_organs,
            "vasculature": self.load_vasculature,
            "brain": self.load_brain,
            "skin": self.load_skin,
            "hand_muscles": self.load_hand_muscles,
            "foot_muscles": self.load_foot_muscles,
            "pelvic_floor": self.load_pelvic_floor,
            "ligaments": self.load_ligaments,
            "oral": self.load_oral,
            "cardiac_additional": self.load_cardiac_additional,
            "intestinal": self.load_intestinal,
            "cns_additional": self.load_cns_additional,
        }
        if layer in table:
            return table[layer]
        if layer in MUSCLE_REGIONS:
            return lambda: self.load_body_muscle_region(
                layer, MUSCLE_REGIONS[layer])
        if layer in INTEGUMENT_STLS:
            return lambda: self.load_single_stl(layer, INTEGUMENT_STLS[layer])
        return None
