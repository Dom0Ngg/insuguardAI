from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class KaggleClaimFeatures(BaseModel):
    Month: str
    WeekOfMonth: int = Field(ge=1, le=5)
    DayOfWeek: str
    Make: str
    AccidentArea: str
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: int = Field(ge=1, le=5)
    Sex: str
    MaritalStatus: str
    Age: int = Field(ge=0)
    Fault: str
    PolicyType: str
    VehicleCategory: str
    VehiclePrice: str
    Deductible: int = Field(ge=0)
    DriverRating: int = Field(ge=1)
    Days_Policy_Accident: str
    Days_Policy_Claim: str
    PastNumberOfClaims: str
    AgeOfVehicle: str
    AgeOfPolicyHolder: str
    PoliceReportFiled: str
    WitnessPresent: str
    AgentType: str
    NumberOfSuppliments: str
    AddressChange_Claim: str
    NumberOfCars: str
    Year: int = Field(ge=1900, le=2100)
    BasePolicy: str


class ClaimAnalysisRequest(BaseModel):
    claim_id: str = Field(description="Identifier used by InsurGuard for this claim")
    features: KaggleClaimFeatures
    policy_id: Optional[str] = Field(
        default=None,
        description="Optional car-policy identifier from GET /policies",
    )
    coverage_query: Optional[str] = Field(
        default=None,
        description="Optional coverage question evaluated only against the selected policy",
    )
    documents: List[str] = Field(
        default_factory=list,
        description="Optional filenames stored in data/claims/<claim_id>/documents/",
    )


class PolicyQueryRequest(BaseModel):
    query: str = Field(min_length=3)


class AgentResult(BaseModel):
    agent_name: str
    status: str
    duration_ms: Optional[float] = None
    output: Dict[str, Any]


class ClaimAnalysisResponse(BaseModel):
    claim_id: str
    fraud_score: Optional[float]
    risk_level: Optional[str]
    manual_review_required: bool
    coverage_result: str
    explanation: str
    agents_trace: List[AgentResult]


class StoredClaimAnalysisOptions(BaseModel):
    policy_id: Optional[str] = None
    coverage_query: Optional[str] = None


class GroundTruthRequest(BaseModel):
    FraudFound_P: int = Field(ge=0, le=1, description="Confirmed fraud label: 0=no fraud, 1=fraud")


class TelegramSimulationRequest(BaseModel):
    chat_id: str = Field(default="100000001", min_length=1)
    text: str = Field(min_length=1)


class AdminAnalysisRequest(BaseModel):
    policy_id: Optional[str] = None
    coverage_query: Optional[str] = None


class PolicyReviewRequest(BaseModel):
    policy_id: str = Field(min_length=1, description="Policy selected by the supervisor")
    query: str = Field(min_length=3, description="Question asked by the supervisor to the policy RAG")
