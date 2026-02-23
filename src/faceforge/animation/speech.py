"""Speech & Phoneme Animation system.

Converts text → phonemes → visemes → AU target sequences for lip-sync.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Viseme definitions ────────────────────────────────────────────────

@dataclass
class Viseme:
    """A lip shape defined by AU target values."""
    name: str
    AU25: float = 0.0  # Lips part
    AU26: float = 0.0  # Jaw drop
    AU22: float = 0.0  # Lip funneler
    AU20: float = 0.0  # Lip stretch
    AU12: float = 0.0  # Lip corner pull

    def to_au_dict(self) -> dict[str, float]:
        return {
            "AU25": self.AU25,
            "AU26": self.AU26,
            "AU22": self.AU22,
            "AU20": self.AU20,
            "AU12": self.AU12,
        }


# Viseme table
VISEMES = {
    "REST": Viseme("REST", 0.0, 0.0, 0.0, 0.0),
    "PP":   Viseme("PP",   0.0, 0.05, 0.0, 0.0),   # Bilabial: P, B, M
    "FF":   Viseme("FF",   0.2, 0.05, 0.0, 0.1),   # Labiodental: F, V
    "TH":   Viseme("TH",   0.3, 0.1,  0.0, 0.0),   # Dental: TH, DH
    "DD":   Viseme("DD",   0.3, 0.1,  0.0, 0.0),   # Alveolar: T, D, N, L
    "SS":   Viseme("SS",   0.2, 0.05, 0.0, 0.2),   # Sibilant: S, Z
    "SH":   Viseme("SH",   0.3, 0.1,  0.4, 0.0),   # Palatal: SH, ZH, CH, JH
    "KK":   Viseme("KK",   0.2, 0.15, 0.0, 0.0),   # Velar: K, G, NG
    "RR":   Viseme("RR",   0.3, 0.1,  0.3, 0.0),   # Retroflex: R, ER
    "AA":   Viseme("AA",   0.6, 0.4,  0.0, 0.0),   # Open vowel: AA, AE, AH
    "EH":   Viseme("EH",   0.4, 0.2,  0.0, 0.2),   # Mid vowel: EH, EY
    "IH":   Viseme("IH",   0.3, 0.1,  0.0, 0.3),   # Close front: IH, IY
    "OH":   Viseme("OH",   0.5, 0.3,  0.5, 0.0),   # Round back: AO, OW
    "UH":   Viseme("UH",   0.3, 0.15, 0.6, 0.0),   # Close round: UH, UW
    "WW":   Viseme("WW",   0.2, 0.1,  0.3, 0.0),   # Glide: W, Y, HH
}

# ARPAbet phoneme → viseme mapping
_PHONEME_TO_VISEME = {
    "P": "PP", "B": "PP", "M": "PP",
    "F": "FF", "V": "FF",
    "TH": "TH", "DH": "TH",
    "T": "DD", "D": "DD", "N": "DD", "L": "DD",
    "S": "SS", "Z": "SS",
    "SH": "SH", "ZH": "SH", "CH": "SH", "JH": "SH",
    "K": "KK", "G": "KK", "NG": "KK",
    "R": "RR", "ER": "RR", "ER0": "RR", "ER1": "RR", "ER2": "RR",
    "AA": "AA", "AA0": "AA", "AA1": "AA", "AA2": "AA",
    "AE": "AA", "AE0": "AA", "AE1": "AA", "AE2": "AA",
    "AH": "AA", "AH0": "AA", "AH1": "AA", "AH2": "AA",
    "EH": "EH", "EH0": "EH", "EH1": "EH", "EH2": "EH",
    "EY": "EH", "EY0": "EH", "EY1": "EH", "EY2": "EH",
    "IH": "IH", "IH0": "IH", "IH1": "IH", "IH2": "IH",
    "IY": "IH", "IY0": "IH", "IY1": "IH", "IY2": "IH",
    "AO": "OH", "AO0": "OH", "AO1": "OH", "AO2": "OH",
    "OW": "OH", "OW0": "OH", "OW1": "OH", "OW2": "OH",
    "OY": "OH", "OY0": "OH", "OY1": "OH", "OY2": "OH",
    "UH": "UH", "UH0": "UH", "UH1": "UH", "UH2": "UH",
    "UW": "UH", "UW0": "UH", "UW1": "UH", "UW2": "UH",
    "AW": "AA", "AW0": "AA", "AW1": "AA", "AW2": "AA",
    "AY": "AA", "AY0": "AA", "AY1": "AA", "AY2": "AA",
    "W": "WW", "Y": "WW", "HH": "WW",
}


# ── Rule-based phoneme fallback ──────────────────────────────────────

_LETTER_RULES = {
    "a": ["AE1"],
    "b": ["B"],
    "c": ["K"],
    "d": ["D"],
    "e": ["EH1"],
    "f": ["F"],
    "g": ["G"],
    "h": ["HH"],
    "i": ["IH1"],
    "j": ["JH"],
    "k": ["K"],
    "l": ["L"],
    "m": ["M"],
    "n": ["N"],
    "o": ["OW1"],
    "p": ["P"],
    "q": ["K", "W"],
    "r": ["R"],
    "s": ["S"],
    "t": ["T"],
    "u": ["AH1"],
    "v": ["V"],
    "w": ["W"],
    "x": ["K", "S"],
    "y": ["Y"],
    "z": ["Z"],
}


@dataclass
class TimedViseme:
    """A viseme with timing information."""
    viseme: str
    start_time: float
    end_time: float
    au_targets: dict[str, float]


class SpeechEngine:
    """Text → phoneme → viseme → AU target sequence pipeline.

    Parameters
    ----------
    phoneme_duration : float
        Duration of each phoneme in seconds (default 0.08).
    blend_time : float
        Cross-fade time between visemes in seconds (default 0.02).
    """

    def __init__(self, phoneme_duration: float = 0.08,
                 blend_time: float = 0.02):
        self._phoneme_duration = phoneme_duration
        self._blend_time = blend_time
        self._cmu_dict: dict[str, list[str]] = {}
        self._load_dict()

    def _load_dict(self) -> None:
        """Load compact CMU dictionary."""
        data_dir = Path(__file__).resolve().parents[2] / "assets" / "data"
        dict_path = data_dir / "cmu_dict_compact.json"
        if dict_path.exists():
            try:
                with open(dict_path) as f:
                    self._cmu_dict = json.load(f)
                logger.info("Loaded CMU dict: %d words", len(self._cmu_dict))
            except Exception as e:
                logger.warning("Failed to load CMU dict: %s", e)

    def text_to_phonemes(self, text: str) -> list[str]:
        """Convert text to ARPAbet phoneme sequence."""
        words = re.findall(r"[a-zA-Z]+", text.lower())
        phonemes = []
        for word in words:
            word_phonemes = self._cmu_dict.get(word)
            if word_phonemes:
                phonemes.extend(word_phonemes)
            else:
                # Rule-based fallback
                phonemes.extend(self._rule_based(word))
            # Insert silence between words
            phonemes.append("SIL")

        # Remove trailing silence
        if phonemes and phonemes[-1] == "SIL":
            phonemes.pop()

        return phonemes

    def _rule_based(self, word: str) -> list[str]:
        """Simple rule-based letter→phoneme conversion for unknown words."""
        phonemes = []
        for letter in word.lower():
            rule = _LETTER_RULES.get(letter)
            if rule:
                phonemes.extend(rule)
        return phonemes if phonemes else ["AH1"]

    def phonemes_to_visemes(self, phonemes: list[str],
                            speed: float = 1.0) -> list[TimedViseme]:
        """Convert phoneme sequence to timed viseme sequence.

        Parameters
        ----------
        phonemes : list[str]
            ARPAbet phoneme sequence.
        speed : float
            Speed multiplier (1.0 = normal, 2.0 = double speed).
        """
        duration = self._phoneme_duration / max(speed, 0.1)
        current_time = 0.0
        timed_visemes: list[TimedViseme] = []

        for phoneme in phonemes:
            if phoneme == "SIL":
                # Short silence between words
                viseme_name = "REST"
                silence_dur = duration * 0.5
                au_targets = VISEMES["REST"].to_au_dict()
                timed_visemes.append(TimedViseme(
                    viseme=viseme_name,
                    start_time=current_time,
                    end_time=current_time + silence_dur,
                    au_targets=au_targets,
                ))
                current_time += silence_dur
                continue

            # Strip stress markers for lookup
            clean = re.sub(r'\d', '', phoneme)
            viseme_name = _PHONEME_TO_VISEME.get(phoneme,
                          _PHONEME_TO_VISEME.get(clean, "REST"))
            viseme = VISEMES.get(viseme_name, VISEMES["REST"])

            timed_visemes.append(TimedViseme(
                viseme=viseme_name,
                start_time=current_time,
                end_time=current_time + duration,
                au_targets=viseme.to_au_dict(),
            ))
            current_time += duration

        return timed_visemes

    def generate_au_sequence(self, text: str,
                             speed: float = 1.0) -> list[TimedViseme]:
        """Full pipeline: text → phonemes → timed viseme AU targets.

        Parameters
        ----------
        text : str
            Input text to animate.
        speed : float
            Speed multiplier.

        Returns
        -------
        list[TimedViseme]
            Sequence of timed viseme targets for animation playback.
        """
        phonemes = self.text_to_phonemes(text)
        return self.phonemes_to_visemes(phonemes, speed)

    def get_total_duration(self, text: str, speed: float = 1.0) -> float:
        """Get the total duration of the speech animation."""
        visemes = self.generate_au_sequence(text, speed)
        if not visemes:
            return 0.0
        return visemes[-1].end_time
