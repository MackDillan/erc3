import logging

from langgraph.graph import END

from erc.state import AgentState


def check_review_status(state: AgentState):
    failures = state.get("consecutive_review_failures", 0)
    is_valid = state.get("plan_is_valid", False)

    if failures >= 5:
        logging.info("REVIEW LIMIT EXCEEDED.")
        return END

    if is_valid:
        plan = state.get("current_plan")
        if plan and plan.steps and plan.steps[0].tool_name == "report_completion":
            return END
        return "execute"

    return "replanning"


def check_execution_status(state: AgentState):
    if state.get("is_finished"):
        logging.info("Job Finished inside Executor.")
        return END

    if state.get("execution_error"):
        logging.info("Error detected. Routing back to PLANNER.")
        return "planner"

    plan = state.get("current_plan")
    if plan and len(plan.steps) > 0:
        return "executor"

    logging.info("Queue empty. Back to PLANNER.")
    return "planner"
