import operator
from typing import TypedDict, Optional, Annotated, List, Union, Dict, Any

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """
    A single step in the execution plan.
    """
    tool_name: str = Field(None, description="Tool Name to be used in order to finish the task")
    arguments: Union[Dict[str, Any], str, None] = Field(None, description="Arguments to be passed to the tool")
    reasoning: str = Field(None, description="Reasoning to be shown when executing the task. Keep in shot")


class ExecutionPlan(BaseModel):
    steps: List[PlanStep] = Field(None, description="List of execution steps")


class AgentState(TypedDict):
    input_task: str
    current_plan: Optional[ExecutionPlan]
    review_feedback: Optional[str]
    plan_is_valid: bool
    consecutive_review_failures: int
    history: Annotated[List[str], operator.add]
    iterations: int
    execution_error: Optional[str]
    is_finished: bool
