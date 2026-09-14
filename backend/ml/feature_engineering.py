from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class KaggleFraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Exact record-level feature engineering used by the TFM benchmark.

    The implementation mirrors the preprocessing study supplied with the TFM.
    It operates only on explanatory fields and never uses ``FraudFound_P``.
    ``PolicyNumber`` and ``RepNumber`` are excluded before this transformer is
    used by InsurGuard.
    """

    MONTH_MAP: Dict[str, int] = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    PAST_CLAIMS_MAP = {"none": 0, "1": 1, "2 to 4": 3, "more than 4": 5}
    SUPPLEMENTS_MAP = {"none": 0, "1 to 2": 1.5, "3 to 5": 4, "more than 5": 6}
    CARS_MAP = {"1 vehicle": 1, "2 vehicles": 2, "3 to 4": 3.5, "more than 4": 5}
    VEHICLE_PRICE_MAP = {
        "less than 20000": 15000,
        "20000 to 29000": 25000,
        "30000 to 39000": 35000,
        "40000 to 59000": 50000,
        "60000 to 69000": 65000,
        "more than 69000": 80000,
    }
    POLICY_HOLDER_AGE_MAP = {
        "16 to 17": 16.5,
        "18 to 20": 19,
        "21 to 25": 23,
        "26 to 30": 28,
        "31 to 35": 33,
        "36 to 40": 38,
        "41 to 50": 45.5,
        "51 to 65": 58,
        "over 65": 70,
    }

    def fit(self, X: pd.DataFrame, y: Any = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Exact cleaning step from the benchmark preprocessing script.
        if "Age" in df.columns:
            df["Age"] = pd.to_numeric(df["Age"], errors="coerce").replace(0, np.nan)
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # 1. Temporal and calendar features.
        if "Month" in df.columns and "MonthClaimed" in df.columns:
            month_num = df["Month"].map(self.MONTH_MAP)
            month_claimed_num = df["MonthClaimed"].map(self.MONTH_MAP)
            df["Claim_Reporting_Lag_Months"] = (month_claimed_num - month_num) % 12
            df["Is_Late_Claim_Report"] = (
                df["Claim_Reporting_Lag_Months"] >= 2
            ).astype(float)

        if "DayOfWeek" in df.columns:
            df["Is_Weekend_Accident"] = df["DayOfWeek"].isin(
                ["Saturday", "Sunday"]
            ).astype(float)

        if "DayOfWeekClaimed" in df.columns:
            df["Is_Weekend_Claimed"] = df["DayOfWeekClaimed"].isin(
                ["Saturday", "Sunday"]
            ).astype(float)

        if "WeekOfMonth" in df.columns:
            df["Is_Month_Boundary"] = df["WeekOfMonth"].isin([1, 5]).astype(float)

        # 2. Magnitude, severity and complexity features.
        if "PastNumberOfClaims" in df.columns:
            df["PastClaims_Num"] = df["PastNumberOfClaims"].map(self.PAST_CLAIMS_MAP)
            df["Has_Past_Claims"] = (df["PastClaims_Num"] > 0).astype(float)

        # NOTE: The original benchmark script checks ``NumberOfSupplements``
        # (with 'e') while the Kaggle source column is ``NumberOfSuppliments``.
        # The condition is deliberately preserved here because reproducing the
        # published benchmark exactly requires the same feature space.
        if "NumberOfSupplements" in df.columns:
            df["Supplements_Num"] = df["NumberOfSupplements"].map(
                self.SUPPLEMENTS_MAP
            )

        if "NumberOfCars" in df.columns:
            df["Cars_Involved_Num"] = df["NumberOfCars"].map(self.CARS_MAP)

        if "VehiclePrice" in df.columns:
            df["Vehicle_Price_Approx"] = df["VehiclePrice"].map(
                self.VEHICLE_PRICE_MAP
            )

        if "Cars_Involved_Num" in df.columns and "Supplements_Num" in df.columns:
            df["Claim_Complexity_Score"] = df["Cars_Involved_Num"] * (
                1 + df["Supplements_Num"]
            )

        if "Deductible" in df.columns and "Vehicle_Price_Approx" in df.columns:
            df["Deductible_to_Price_Ratio"] = df["Deductible"] / (
                df["Vehicle_Price_Approx"] + 1e-5
            )

        # 3. Driver / policy-holder profile discrepancies.
        if "Age" in df.columns:
            df["Is_Young_Driver"] = (df["Age"] < 25).astype(float)
            df["Is_Senior_Driver"] = (df["Age"] > 65).astype(float)

        if "Age" in df.columns and "AgeOfPolicyHolder" in df.columns:
            holder_age_numeric = df["AgeOfPolicyHolder"].map(
                self.POLICY_HOLDER_AGE_MAP
            )
            df["Driver_vs_PolicyHolder_Age_Diff"] = (
                df["Age"] - holder_age_numeric
            ).abs()
            df["Driver_Is_Not_Main_Holder"] = (
                df["Driver_vs_PolicyHolder_Age_Diff"] > 5
            ).astype(float)

        if "Is_Young_Driver" in df.columns and "Vehicle_Price_Approx" in df.columns:
            df["Young_Driver_Expensive_Car"] = (
                (df["Is_Young_Driver"] == 1.0)
                & (df["Vehicle_Price_Approx"] >= 40000)
            ).astype(float)

        # 4. Operational suspicion / control indicators.
        if "PoliceReportFiled" in df.columns and "WitnessPresent" in df.columns:
            df["No_Police_No_Witness"] = (
                (df["PoliceReportFiled"] == "No")
                & (df["WitnessPresent"] == "No")
            ).astype(float)

        if "Fault" in df.columns and "PoliceReportFiled" in df.columns:
            df["At_Fault_Without_Police"] = (
                (df["Fault"] == "Policy Holder")
                & (df["PoliceReportFiled"] == "No")
            ).astype(float)

        if "AddressChange_Claim" in df.columns:
            df["Recent_Address_Change"] = df["AddressChange_Claim"].apply(
                lambda x: (
                    1.0
                    if str(x).lower() in ["under 6 months", "1 year"]
                    else 0.0
                    if pd.notna(x)
                    else np.nan
                )
            )

        if "Days_Policy_Accident" in df.columns:
            df["Accident_Right_After_Policy"] = df["Days_Policy_Accident"].apply(
                lambda x: (
                    1.0
                    if str(x).lower() in ["1 to 7", "8 to 15", "15 to 30"]
                    else 0.0
                    if pd.notna(x)
                    else np.nan
                )
            )

        return df
