from backend.ml.fraud_model import FraudDetectionModel


def test_model_trains_from_kaggle_dataset():
    model = FraudDetectionModel()
    info = model.get_model_info()
    assert info["training_summary"]["status"] == "trained"
    assert info["training_summary"]["total_samples"] == 15420
    assert info["training_summary"]["positive_labels"] == 923
    assert info["training_file"] == "data/kaggle/fraud_oracle.csv"
    assert info["model_version"] == "3.0-kaggle-logistic-l1-benchmark-exact"
    assert info["model_type"] == "LogisticRegressionL1"
    assert info["training_summary"]["regularization"] == "L1"
    assert info["training_summary"]["age_zero_rows_treated_as_missing"] == 320
    assert info["training_summary"]["preprocessing"]["selected_encoded_features"] == 24


def test_holdout_metrics_exactly_reproduce_supplied_benchmark():
    info = FraudDetectionModel().get_model_info()
    holdout = info["metrics"]["holdout"]
    decision = holdout["at_decision_threshold"]
    assert abs(holdout["roc_auc"] - 0.755227) < 1e-6
    assert abs(holdout["pr_auc"] - 0.526455) < 1e-6
    assert abs(decision["threshold"] - 0.574822) < 1e-6
    assert abs(decision["recall"] - 0.9243) < 1e-4
    assert abs(decision["precision"] - 0.1256) < 1e-4
    assert decision["confusion_matrix"] == [[1708, 1191], [14, 171]]
    assert info["training_summary"]["C"] == 0.001
    assert abs(info["training_summary"]["scale_positive_weight"] - 15.715447154471544) < 1e-12
    assert info["training_summary"]["classifier_nonzero_coefficients"] == 2


def test_selected_feature_space_matches_supplied_benchmark():
    info = FraudDetectionModel().get_model_info()
    expected = [
        "At_Fault_Without_Police", "Age_is_missing", "Month_Dec", "Month_Jul",
        "Month_Mar", "Month_Nov", "Make_VW", "MonthClaimed_Aug",
        "MonthClaimed_May", "MonthClaimed_Nov", "Sex_Male", "Fault_Third Party",
        "PolicyType_Sedan - Collision", "PolicyType_Sport - Collision",
        "VehiclePrice_40000 to 59000", "VehiclePrice_less than 20000",
        "AgeOfVehicle_4 years", "AgeOfVehicle_5 years", "PoliceReportFiled_Yes",
        "AgentType_Internal", "NumberOfSuppliments_3 to 5",
        "AddressChange_Claim_2 to 3 years", "AddressChange_Claim_4 to 8 years",
        "BasePolicy_Liability",
    ]
    assert info["selected_features"] == expected


def test_model_requires_all_30_raw_features():
    model = FraudDetectionModel()
    result = model.predict({"Month": "Jan"})
    assert result["ml_status"] == "invalid_features"
    assert len(result["feature_validation"]["missing_features"]) == 29
    assert result["feature_validation"]["required_feature_count"] == 30


def test_model_produces_local_explanation_for_valid_claim():
    import pandas as pd

    model = FraudDetectionModel()
    row = pd.read_csv(model.training_file).iloc[0]
    features = {feature: row[feature] for feature in model.FEATURES}
    result = model.predict(features)
    assert result["ml_status"] == "scored"
    assert 0.0 <= result["ml_fraud_score"] <= 1.0
    assert abs(result["ml_decision_threshold"] - 0.574822) < 1e-6
    assert isinstance(result["local_shap"], list)


def test_validation_accepts_real_kaggle_policy_type_vehicle_category_combination():
    """Kaggle contains Sedan-Liability rows whose VehicleCategory is Sport."""
    from backend.agents.validation_agent import ValidationAgent
    import pandas as pd

    model = FraudDetectionModel()
    df = pd.read_csv(model.training_file)
    row = df[
        (df["PolicyType"] == "Sedan - Liability")
        & (df["VehicleCategory"] == "Sport")
        & (df["BasePolicy"] == "Liability")
    ].iloc[0]
    features = {feature: row[feature] for feature in model.FEATURES}
    result = ValidationAgent().run({"claim_id": "VALID-KAGGLE", "features": features})
    assert result["output"]["validation_status"] == "valid"
    assert not result["output"]["issues"]


def test_validation_warns_when_policy_type_conflicts_with_base_policy():
    from backend.agents.validation_agent import ValidationAgent
    import pandas as pd

    model = FraudDetectionModel()
    row = pd.read_csv(model.training_file).iloc[0]
    features = {feature: row[feature] for feature in model.FEATURES}
    features["BasePolicy"] = "Collision" if features["BasePolicy"] != "Collision" else "Liability"
    result = ValidationAgent().run({"claim_id": "INVALID-POLICY", "features": features})
    assert result["output"]["validation_status"] == "warning"
    assert any("PolicyType no coincide con BasePolicy" in issue for issue in result["output"]["issues"])
