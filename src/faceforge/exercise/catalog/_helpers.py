"""Constructors and shared source citations for the catalogue modules."""

from __future__ import annotations

from faceforge.exercise.model import (
    EquipmentSpec, MuscleUse, Phase, PhaseKind, Role,
)
from faceforge.exercise.pose_library import (  # noqa: F401  (re-exported for the catalogue)
    arms, combine, flat_foot_ankle, grip, hinge, lunge, merge, neutral, only, pose, squat,
    stand,
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
SEATED_ON_BENCH = (0.0, BENCH_TOP + 10.0 + 81.0, 0.0)


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
       **params) -> EquipmentSpec:
    return EquipmentSpec(kind=kind, attach=attach, position=tuple(position),
                         rotation_deg=tuple(rotation_deg), params=dict(params))


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
