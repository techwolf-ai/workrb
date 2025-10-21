"""Shared types and enums used across WorkBench."""

from enum import Enum


class Language(str, Enum):
    """Language enum."""

    EN = "en"
    DE = "de"
    FR = "fr"
    IT = "it"
    ES = "es"
    BG = "bg"
    CS = "cs"
    DA = "da"
    ET = "et"
    EL = "el"
    GA = "ga"
    HR = "hr"
    LV = "lv"
    LT = "lt"
    HU = "hu"
    MT = "mt"
    NL = "nl"
    PL = "pl"
    PT = "pt"
    RO = "ro"
    SK = "sk"
    SL = "sl"
    FI = "fi"
    SV = "sv"
    IS = "is"
    NO = "no"
    AR = "ar"
    UK = "uk"


class LabelType(str, Enum):
    """Label type enum."""

    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"


class DatasetSplit(str, Enum):
    """Dataset split enum."""

    VAL = "val"
    TEST = "test"
