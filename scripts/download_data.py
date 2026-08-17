#!/usr/bin/env python3
"""
AutoDS Data Downloader and Verifier
Downloads, extracts, and validates reference datasets from official sources.
"""

import argparse
import hashlib
import io
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoDS-DataDownloader")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


def calculate_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_bank_marketing() -> Path:
    """
    Download and extract the UCI Bank Marketing Dataset from official repository.
    Source: https://archive.ics.uci.edu/static/public/222/bank+marketing.zip
    """
    target_dir = RAW_DIR / "bank_marketing"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    primary_csv = target_dir / "bank-additional-full.csv"
    if primary_csv.exists():
        logger.info(f"Bank marketing dataset already exists at {primary_csv}")
        df = pd.read_csv(primary_csv, sep=";")
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        return primary_csv

    url = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
    logger.info(f"Downloading Bank Marketing dataset from {url}...")
    
    headers = {"User-Agent": "AutoDS-Agent/1.0 (Research/Education)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()
        
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(target_dir)
        
    # UCI zip contains bank-additional.zip inside
    inner_zip = target_dir / "bank-additional.zip"
    if inner_zip.exists():
        with zipfile.ZipFile(inner_zip) as izf:
            izf.extractall(target_dir)
        inner_zip.unlink(missing_ok=True)
        
    # Also extract bank.zip if present
    inner_bank_zip = target_dir / "bank.zip"
    if inner_bank_zip.exists():
        with zipfile.ZipFile(inner_bank_zip) as izf:
            izf.extractall(target_dir)
        inner_bank_zip.unlink(missing_ok=True)
        
    # Check if nested folder exists
    nested_dir = target_dir / "bank-additional"
    if nested_dir.exists() and (nested_dir / "bank-additional-full.csv").exists():
        for item in nested_dir.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                item.rename(dest)
        try:
            nested_dir.rmdir()
        except Exception:
            pass

    if primary_csv.exists():
        checksum = calculate_sha256(primary_csv)
        df = pd.read_csv(primary_csv, sep=";")
        logger.info(f"Successfully downloaded Bank Marketing data!")
        logger.info(f"Path: {primary_csv}")
        logger.info(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        logger.info(f"SHA-256: {checksum}")
        logger.info(f"Target 'y' distribution:\n{df['y'].value_counts(normalize=True)}")
        return primary_csv
    else:
        # Fallback to bank-full.csv if additional not extracted directly
        alt_csv = target_dir / "bank-full.csv"
        if alt_csv.exists():
            df = pd.read_csv(alt_csv, sep=";")
            logger.info(f"Found bank-full.csv: {len(df)} rows, {len(df.columns)} columns.")
            return alt_csv
        raise RuntimeError("Could not find extracted CSV in Bank Marketing archive.")


def generate_or_download_m5_sample() -> Path:
    """
    Generate or download structured M5 time-series forecasting sample dataset.
    Provides multi-item daily sales, calendar attributes (events, snap, day-of-week), and prices.
    """
    target_dir = RAW_DIR / "m5"
    target_dir.mkdir(parents=True, exist_ok=True)
    m5_csv = target_dir / "m5_sales_sample.csv"
    
    if m5_csv.exists():
        logger.info(f"M5 sample dataset already exists at {m5_csv}")
        return m5_csv
        
    logger.info("Generating realistic multi-series M5-structured retail forecasting dataset...")
    np.random.seed(42)
    dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq="D")
    stores = ["CA_1", "CA_2", "TX_1"]
    items = ["HOBBIES_1_001", "FOODS_1_001", "HOUSEHOLD_1_001", "FOODS_2_005", "HOBBIES_2_010"]
    
    records = []
    for store in stores:
        for item in items:
            base_sales = np.random.uniform(5, 50)
            trend = np.linspace(0, 5, len(dates))
            weekly_cycle = 3 * np.sin(2 * np.pi * dates.dayofweek / 7)
            yearly_cycle = 8 * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            price = round(np.random.uniform(2.5, 19.99), 2)
            
            noise = np.random.poisson(lam=3, size=len(dates)) - 3
            promo_flag = (np.random.rand(len(dates)) > 0.85).astype(int)
            promo_lift = promo_flag * np.random.uniform(5, 15)
            
            sales = np.maximum(0, base_sales + trend + weekly_cycle + yearly_cycle + promo_lift + noise).round().astype(int)
            
            for d, s, p, pr in zip(dates, sales, [price]*len(dates), promo_flag):
                records.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "item_id": item,
                    "store_id": store,
                    "dept_id": item.split("_")[0],
                    "sell_price": p,
                    "is_promo": pr,
                    "day_of_week": d.day_name(),
                    "month": d.month,
                    "sales": s
                })
                
    df = pd.DataFrame(records)
    df.to_csv(m5_csv, index=False)
    logger.info(f"Created M5 forecasting sample dataset: {len(df)} rows across {len(stores)*len(items)} series.")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return m5_csv


def generate_housing_regression_sample() -> Path:
    """
    Download or generate standard tabular regression dataset (Housing Price prediction).
    """
    target_dir = RAW_DIR / "housing"
    target_dir.mkdir(parents=True, exist_ok=True)
    housing_csv = target_dir / "housing_prices.csv"
    
    if housing_csv.exists():
        logger.info(f"Housing regression dataset already exists at {housing_csv}")
        return housing_csv
        
    try:
        from sklearn.datasets import fetch_california_housing
        cal = fetch_california_housing(as_frame=True)
        df = cal.frame.rename(columns={"MedHouseVal": "median_house_value"})
        # Scale to realistic dollar amounts ($100k)
        df["median_house_value"] = (df["median_house_value"] * 100000).round(2)
        df.to_csv(housing_csv, index=False)
        logger.info(f"Fetched California Housing dataset: {len(df)} rows, {len(df.columns)} columns.")
    except Exception as e:
        logger.warning(f"Could not fetch California housing ({e}), creating synthetic housing data.")
        np.random.seed(42)
        n = 5000
        sqft = np.random.normal(1800, 500, n).clip(500, 5000)
        bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.1, 0.25, 0.4, 0.2, 0.05])
        bathrooms = np.maximum(1, (bedrooms * 0.75 + np.random.normal(0, 0.5, n)).round(1))
        location_score = np.random.uniform(1, 10, n)
        age = np.random.uniform(0, 50, n)
        price = 50000 + (sqft * 180) + (bedrooms * 15000) + (bathrooms * 20000) + (location_score * 25000) - (age * 1200) + np.random.normal(0, 25000, n)
        df = pd.DataFrame({
            "sqft": sqft.round(),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location_score": location_score.round(2),
            "age": age.round(),
            "has_garage": np.random.choice([0, 1], n, p=[0.3, 0.7]),
            "median_house_value": price.round(2)
        })
        df.to_csv(housing_csv, index=False)
        logger.info(f"Created synthetic housing dataset: {len(df)} rows.")
        
    return housing_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AutoDS reference datasets")
    parser.add_argument("--dataset", choices=["bank_marketing", "m5_sample", "housing", "all"], default="all", help="Dataset to download")
    args = parser.parse_args()
    
    if args.dataset in ("bank_marketing", "all"):
        download_bank_marketing()
    if args.dataset in ("m5_sample", "all"):
        generate_or_download_m5_sample()
    if args.dataset in ("housing", "all"):
        generate_housing_regression_sample()
    logger.info("Dataset setup completed.")
