# src\data_prep.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

@dataclass
class PrepConfig:
    drop_personal_cols: Tuple[str, ...] = ("Name", "Email", "Address", "Notes")
    meta_cols: Tuple[str, ...] = (
        "ID", "Souches", "bacteria",
        "age", "gender", "Diabetes", "Hypertension",
        "Hospital_before", "Infection_Freq", "Collection_Date"
    )

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_columns(df: pd.DataFrame, cfg: PrepConfig = PrepConfig()) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    drop_cols = [c for c in cfg.drop_personal_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    return df

def split_age_gender(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "age/gender" not in df.columns:
        return df
    
    ag = df["age/gender"].astype(str)

    df["age"] = pd.to_numeric(ag.str.split("/").str[0], errors="coerce")

    df["gender"] = ag.str.split("/").str[1]

    df = df.drop(columns=["age/gender"])
    return df


def extract_bacteria_from_souches(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Souches" not in df.columns:
        raise ValueError("'Souches' not found")
    
    s = df["Souches"].astype(str).str.strip()

    df["bacteria"] = (
        df["Souches"].astype(str)
        .str.strip()
        .str.split()
        .str[1:]
        .str.join(" ")
    )

    df["bacteria"] = df["bacteria"].str.replace(r"\s+", " ", regex=True).str.strip()

    return df

def normalize_bacteria_name(x: str) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" .", ".").replace(". ", ".")
    return s

def apply_bacteria_normalization(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    df["bacteria"] = df["bacteria"].map(normalize_bacteria_name)

    bacteria_map = {
    "E.coi": "Escherichia coli",
    "E.cli": "Escherichia coli",
    "E.coli": "Escherichia coli",
    "Escherichia coli": "Escherichia coli",

    "Enter.bacteria spp.": "Enterobacteria spp.",
    "Enteobacteria spp.": "Enterobacteria spp.",
    "Enterobacteria spp.": "Enterobacteria spp.",

    "Klbsiella pneumoniae": "Klebsiella pneumoniae",
    "Klebsiella pneumoniae": "Klebsiella pneumoniae",
    "Klebsie.lla pneumoniae": "Klebsiella pneumoniae",

    "Proteus mirabilis": "Proteus mirabilis",
    "Protus mirabilis": "Proteus mirabilis",
    "Proeus mirabilis": "Proteus mirabilis",
    "Prot.eus mirabilis": "Proteus mirabilis"
    }

    df["bacteria"] = df["bacteria"].replace(bacteria_map)

    return df

def get_antibiotic_cols(df: pd.DataFrame, cfg: PrepConfig = PrepConfig()) -> List[str]:
    meta = set(cfg.meta_cols)
    return [c for c in df.columns if c not in meta]

def normalize_sr(x) -> str | float:

    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in {"S", "R"}:
        return s
    if s in {"I"}:
        return "I"
    return np.nan

def normalize_antibiotics_table(df: pd.DataFrame, antibiotic_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in antibiotic_cols:
        df[c] = df[c].map(normalize_sr)
    return df

def to_long_format(df: pd.DataFrame, antibiotic_cols: List[str]) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=[c for c in df.columns if c not in antibiotic_cols],
        value_vars=antibiotic_cols,
        var_name="antibiotic",
        value_name="sir",
    )

    long_df = long_df.dropna(subset=["sir"])
    long_df = long_df[long_df["sir"].isin(["S", "R"])].copy()
    long_df["susceptible"] = (long_df["sir"] == "S").astype(int)

    long_df["bacteria"] = long_df["bacteria"].replace(
        {
            "": np.nan,
            "nan": np.nan,
            "NaN": np.nan,
            "None": np.nan
        }
    )
    long_df = long_df.dropna(subset=["bacteria"])

    return long_df

def prepare_long_df(path: str, cfg: PrepConfig = PrepConfig()) -> pd.DataFrame:
    df = load_data(path)
    df = clean_columns(df, cfg=cfg)
    df = split_age_gender(df)
    df = extract_bacteria_from_souches(df)
    df = apply_bacteria_normalization(df)

    antibiotic_cols = get_antibiotic_cols(df, cfg=cfg)
    df = normalize_antibiotics_table(df, antibiotic_cols)

    long_df = to_long_format(df, antibiotic_cols)
    return long_df

