"""Constructors and shared source citations for the catalogue modules."""

from __future__ import annotations

from faceforge.exercise.model import (
    EquipmentSpec, MuscleUse, Phase, PhaseKind, Role,
)
from faceforge.exercise.pose_library import (  # noqa: F401  (re-exported for the catalogue)
    arms, combine, flat_foot_ankle, grip, hinge, lunge, merge, neutral, only, pose, squat,
    stand, toes_on_floor, toes_tucked,
)

P, S, ST = Role.PRIMARY, Role.SECONDARY, Role.STABILISER
ECC, CON, ISO, TRN = (PhaseKind.ECCENTRIC, PhaseKind.CONCENTRIC, PhaseKind.ISOMETRIC,
                      PhaseKind.TRANSITION)

#: Body-frame points to rotate about (rest pose, body coordinates).
HIPS = (0.0, 0.0, -81.0)
SHOULDERS = (0.0, 0.0, -15.0)
LOWER_CHEST = (0.0, 0.0, -45.0)

#: Seat height placements: hip joint sits ~10 units above a bench top.
BENCH_TOP = 58.0
#: Where a tilted bench pad hinges, in bench-local x.  A supine lifter is
#: tilted by a wrapper pitch about the HIPS, and the hip sits at world x = -4
#: against a bench at -45 -- so a pad hinged at its own centre comes away from
#: the body by 41 x sin(theta).  Measured 2026-09-17: 18.1 units of daylight
#: under the 30 deg incline and 15.1 of pad through the lifter on the 20 deg
#: decline, against a flat bench that reads +10.6 at the shoulder.  See
#: `equipment.make_bench`.
BENCH_HINGE_X = 41.0
SEATED_ON_BENCH = (0.0, BENCH_TOP + 10.0 + 81.0, 0.0)
#: Hip flexion for a body sitting on that bench.  Not 90: the model's knee
#: stands 57 above the floor, so a horizontal thigh from a hip 10 above a
#: 58-high seat puts the knee at 68 and the foot 8.8 in the air (measured on
#: the seated press).  The thigh slopes down to the knee instead, as a seated
#: person's does -- 1.42 units of foot height a degree, so 90 leaves it 8.8
#: up, 81 drives it 4.0 through the floor, and 84.5 lands it.
SEATED_HIP = 84.5

#: Feet on the floor either side of a flat bench, shared by the whole bench
#: family.  Re-measured 2026-09-16: knee 70 left the lowest foot pivot **7.3
#: above the floor** -- every flat-bench lifter in the catalogue had his feet
#: dangling.  More knee flexion is what lowers them, not less (the thigh is
#: nearly horizontal, so the shank swings the foot down as it folds): 78 gives
#: 3.3, 85 gives 0.2 and 92 goes 2.0 through, all at ankle -5.
#:
#: The ankle is a trade-off and 85/-5 is the measured best of it.  The supine
#: foot-flat rule (``-90 + pitch - hip + knee``) wants +5 here, which does
#: flatten the sole -- but dorsiflexing the ankle also lifts the whole foot,
#: about half a unit a degree, and 85/+5 measured 4.9 in the air while 92/+12
#: was worse again at 5.8.  A foot on its forefoot with the heel 3.7 up is
#: closer to a bench setup than a flat foot 5 units off the floor.
def bench_legs() -> dict[str, float]:
    from faceforge.exercise.pose_library import only

    return only(hip_flex=-10, knee_flex=85, ankle_flex=-5, hip_abduct=22)


#: The same, for a bench pitched 30 deg up about the hips: the pitch carries
#: the legs down with it, and 26 deg more hip flexion keeps the soles on the
#: floor (measured foot height 3.5; the flat-bench legs put them 20 below).
def incline_legs() -> dict[str, float]:
    from faceforge.exercise.pose_library import only

    return only(hip_flex=16, knee_flex=70, ankle_flex=-5, hip_abduct=22)


#: A palm flat on the floor.  Without it the hand simply continues the line of
#: the forearm: measured on the push-up, the middle fingertip sat 17.2 units
#: under the mat and a close-up render showed both hands vanish into it -- the
#: figure was pressing on the ends of its wrists.  Pronation is the half that
#: matters most, because it is what turns the wrist's flexion axis into one
#: that can lift the fingers at all; extension alone moved them 5 units.
#: Pronated 90 and extended 70 (the rig's limit) puts the knuckles level with
#: the wrist, which is a flat palm, with the fingertips a few units low.
def flat_palm() -> dict[str, float]:
    from faceforge.exercise.pose_library import only

    return only(forearm_rotate=-90, wrist_flex=-70)


def ph(name: str, kind: PhaseKind, duration: float, pose_: dict, pitch: float = 0.0,
       cues=(), act: dict | None = None, lift: float = 0.0, travel=(0.0, 0.0), pivot=None,
       position=None, orientation=None, easing: str = "ease_in_out", roll: float = 0.0,
       yaw: float = 0.0) -> Phase:
    return Phase(name=name, kind=kind, duration=duration, pose=dict(pose_), pitch=pitch,
                 roll=roll, yaw=yaw, lift=lift, travel=tuple(travel), pivot=pivot,
                 position=position, orientation=orientation, cues=tuple(cues),
                 activation=dict(act or {}), easing=easing)


def mu(group: str, role: Role, peak: float | None = None, side: str | None = None,
       note: str = "") -> MuscleUse:
    return MuscleUse(group=group, role=role, peak=peak, side=side, note=note)


def eq(kind: str, attach: str = "hands", position=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0),
       hang: float | None = None, **params) -> EquipmentSpec:
    return EquipmentSpec(kind=kind, attach=attach, position=tuple(position),
                         rotation_deg=tuple(rotation_deg), params=dict(params), hang=hang)


# -- sources ---------------------------------------------------------------------
# Fetched during the 2026-09 research pass:
SQUAT_KIN = ("Kinematic analysis of the back squat at different load intensities in "
             "powerlifters and weightlifters, Front Sports Act Living 2024 "
             "(hip/knee relative angles 51-57 deg at full depth) "
             "https://pmc.ncbi.nlm.nih.gov/articles/PMC11565377/")
SQUAT_REVIEW = ("A biomechanical review of the squat exercise: implications for clinical "
                "practice, IJSPT (depth by knee flexion 0-90 / 90-110 / 110-135 deg; gluteus "
                "maximus 28-35 %MVIC) https://ijspt.scholasticahq.com/article/94600")
DEADLIFT_SPM = ("Biomechanical analysis of conventional and sumo deadlift, 2025 "
                "(phase-1 ROM hip 38, knee 33, ankle 13 deg; BF 78 %MVC, TA 82, VL 55-63) "
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC12148905/")
BENCH_INCLINE = ("Effect of five bench inclinations on EMG of pectoralis major, anterior "
                 "deltoid and triceps, 2020 (shoulder abduction ~45 deg at the bottom, 2 s / 2 s "
                 "tempo, pec ~27, AD 26-33, TB ~15 %MVIC at 60 %1RM) "
                 "https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/")
PULLUP_YOUDAS = ("Youdas JW et al. 2010. Surface EMG activation patterns and elbow joint "
                 "motion during a pull-up, chin-up or Perfect-Pullup. JSCR 24(12):3404 "
                 "(lat 117-130, biceps 78-96, lower trapezius 45-56 %MVIC; initiated by lower "
                 "trapezius/pectoralis, completed by biceps/latissimus)")
LATPULL_HD = ("High-density EMG excitation in front vs back lat pull-down prime movers "
              "https://pmc.ncbi.nlm.nih.gov/articles/PMC11057623/")
ECC_CON = ("Eccentric EMG 7-31 % lower than velocity-matched concentric; accentuated "
           "eccentric loading in the bench press https://pmc.ncbi.nlm.nih.gov/articles/PMC8822896/")
DIGIOVINE = "DiGiovine NM et al. 1992. EMG bands: low 0-20, moderate 21-40, high 41-60, very high >60 %MVC"
# Standard references (not fetched in this session):
SCHOENFELD = ("Schoenfeld BJ 2010. Squatting kinematics and kinetics and their application to "
              "exercise performance. JSCR 24(12):3497")
ESCAMILLA_DL = ("Escamilla RF et al. 2000. A three-dimensional biomechanical analysis of sumo "
                "and conventional style deadlifts. MSSE 32(7):1265")
CONTRERAS = ("Contreras B et al. 2015. Gluteus maximus, biceps femoris and vastus lateralis EMG "
             "in the barbell hip thrust, back squat and deadlift. J Appl Biomech 31(6):452")
NSCA = "Haff GG, Triplett NT (eds) 2016. NSCA Essentials of Strength Training and Conditioning, 4th ed., ch. 15 exercise technique"
ACE = "ACE Exercise Library https://www.acefitness.org/resources/everyone/exercise-library/"
EXRX = "ExRx.net exercise and muscle directory https://exrx.net/Lists/Directory"
NEUMANN = "Neumann DA 2017. Kinesiology of the Musculoskeletal System, 3rd ed."
PERRY = "Perry J, Burnfield JM 2010. Gait Analysis: Normal and Pathological Function, 2nd ed. (muscle timing by % gait cycle)"
HUG = "Hug F, Dorel S 2009. Electromyographic analysis of pedaling: a review. J Electromyogr Kinesiol 19(2):182 (crank-angle activation ranges)"
KLESHNEV = "Kleshnev V 2016. The Biomechanics of Rowing. Crowood Press (catch/drive/finish/recovery sequence)"
LAKE = "Lake JP, Lauder MA 2012. Kettlebell swing training improves maximal and explosive strength. JSCR 26(8):2228"
ZEBIS = "Zebis MK et al. 2013. Kettlebell swing targets semitendinosus and supine leg curl targets biceps femoris. BJSM 47:1192"
EKSTROM = ("Ekstrom RA, Donatelli RA, Carp KC 2007. EMG analysis of core trunk, hip and thigh "
           "muscles during 9 rehabilitation exercises. JOSPT 37(12):754")
DISTEFANO = ("Distefano LJ et al. 2009. Gluteal muscle activation during common therapeutic "
             "exercises. JOSPT 39(7):532")
BOREN = ("Boren K et al. 2011. EMG analysis of gluteus medius and gluteus maximus during "
         "rehabilitation exercises. IJSPT 6(3):206")
ESCAMILLA_ABS = ("Escamilla RF et al. 2006. EMG analysis of traditional and nontraditional "
                 "abdominal exercises. JOSPT 36(2):45")
MCGILL = "McGill SM 2007. Low Back Disorders, 2nd ed. (curl-up, side bridge, bird dog)"
CALATAYUD = ("Calatayud J et al. 2015. Bench press and push-up at comparable levels of muscle "
             "activity. JSCR 29(1):246")
SAETERBAKKEN = ("Saeterbakken AH, Fimland MS 2013. Effects of body position and loading "
                "modality on muscle activity and strength in shoulder presses. JSCR 27(7):1824")
SNARR = "Snarr RL, Esco MR 2014. EMG comparison of plank variations. JSCR 28(11):3298"
BOTTON = ("Marcolin G et al. 2018 / Oliveira LF et al. 2009. Elbow flexor EMG in supinated, "
          "neutral and pronated curls (brachioradialis favours the hammer curl)")
SIGNORILE = ("Signorile JF et al. 2002. Muscle activation during lat pull-down variations. "
             "JSCR 16(4):539")
ANDERSEN = ("Andersen V et al. 2014. Effects of grip width on muscle strength and activation "
            "in the lat pull-down. JSCR 28(4):1135")
SCHOENFELD_ROW = ("Fenwick CM, Brown SH, McGill SM 2009. Comparison of different rowing "
                  "exercises: trunk muscle activation and lumbar spine motion. JSCR 23(5):1408")
KOLBER = "Kolber MJ et al. 2010. Shoulder injuries attributed to resistance training. JSCR 24(6):1696"
REINOLD = ("Reinold MM et al. 2004. EMG analysis of the rotator cuff and deltoid during "
           "shoulder external rotation exercises. JOSPT 34(7):385")
YOGA_EMG = ("Ni M et al. 2014. Muscle utilization patterns vary by skill levels of the "
            "practitioners across specific yoga poses (asanas). Complement Ther Med 22(4):662")
YOGA_KIN = ("Salem GJ et al. 2013. Physical demand profiles of hatha yoga postures performed "
            "by older adults. Evid Based Complement Alternat Med 2013:165763")
STRETCH_ACSM = ("Garber CE et al. 2011. ACSM position stand: quantity and quality of exercise. "
                "MSSE 43(7):1334 (flexibility: static holds of 10-30 s, 2-3 days per week)")
STRETCH_PAGE = ("Page P 2012. Current concepts in muscle stretching for exercise and "
                "rehabilitation. IJSPT 7(1):109")
BEHM = ("Behm DG et al. 2016. Acute effects of muscle stretching on physical performance, "
        "range of motion and injury incidence in healthy active individuals: a systematic "
        "review. Appl Physiol Nutr Metab 41(1):1")
