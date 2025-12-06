from typing import TypedDict, Optional

from pydantic import BaseModel

from erc.experts.schemas import ExecutionPlan, ConstraintExpertOutput


class Plan(BaseModel):
    plan: ExecutionPlan
    is_validated: bool
    validation_attempts: int
    review: Optional[ConstraintExpertOutput]


class AgentState(TypedDict):
    input_task: str
    plan: Optional[Plan]
