import json
import logging
import sys
import time

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from erc.experts.base import BaseExpert
from erc.experts.schemas import ConstraintExpertOutput, PlanStep
from erc.persona import PersonaProvider
from erc.state import AgentState, ExecutionPlan, Plan
from erc.store.tools import TOOLS_DESC


class ConstraintExpert(BaseExpert):

    def __init__(self, persona_path, tool_desc: str, llm: ChatOpenAI, callback):
        self.persona_provider = PersonaProvider("constraint_expert", persona_path)
        self.tools_desc = tool_desc
        self.llm = llm.with_structured_output(ConstraintExpertOutput)
        self.callback = callback

    def node(self, state: AgentState):
        logging.info("REVIEWER Checking...")

        plan = state.get("plan", None)
        print(plan)
        if not plan or not plan.steps:
            attempts = plan.validation_attempts + 1
            new_plan = Plan(
                plan=ExecutionPlan(steps=[]),
                is_validated=True,
                is_valid=False,
                validation_attempts=attempts,
            )
            return state.copy(
                plan=new_plan,
            )

        plan_str = json.dumps(plan.model_dump(), indent=2)

        system_text = self.persona_provider.get_primary_persona()

        user_text = f"""
            You are an expert reviewer that checks whether the proposed execution plan meets all constraints for the given task.
            Available Tools: {self.tools_desc}
            PLAN:\n{plan_str}
            Task: {state['input_task']}
        """

        try:
            started = time.time()
            messages = [SystemMessage(content=system_text), HumanMessage(content=user_text)]
            usage_meta_data = UsageMetadataCallbackHandler()

            response = self.llm.invoke(messages, config={"callbacks": [usage_meta_data]})

            if self.callback:
                self.callback(usage_meta_data, started)

            logging.info(f"Constraints RESPONSE: {response}")

            if response is None:
                # TODO:
                return {
                    "plan_is_valid": True,
                    "review_feedback": "Auto-approved (JSON parsing failed)",
                    "consecutive_review_failures": 0
                }

            if response.is_valid:
                logging.info("Plan Approved.")
                new_plan = plan.model_copy()
                new_plan.is_validated = True
                new_plan.review = response
                return state.copy(
                    plan=new_plan,
                )
            else:
                logging.warn(f"Plan Rejected: {response.review_feedback}")
                # TODO:
                return {
                    "plan_is_valid": False,
                    "review_feedback": response.review_feedback,
                    "consecutive_review_failures": state["consecutive_review_failures"] + 1
                }

        except Exception as e:
            logging.error(f"Reviewer Logic Crash ({e}). Allowing plan to proceed.")
            # TODO:
            return {
                "plan_is_valid": True,
                "consecutive_review_failures": 0
            }


if __name__ == "__main__":
    def meta_callback(meta, started):
        pass


    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    llm = ChatOpenAI(
        model="oss-20b",
        base_url="http://localhost:8080/v1",
        api_key="",
        temperature=0.0,
        max_tokens=1000,
    )
    c = ConstraintExpert(
        persona_path="../../prompts/oss-20b-synthetic-persona",
        llm=llm,
        tool_desc=TOOLS_DESC,
        callback=meta_callback,
    )

    state = AgentState(
        input_task="Count characters is world raspberry",
        plan=ExecutionPlan(steps=[PlanStep(tool_name='report_completion',
                                           arguments={'final_message': "The word 'raspberry' has 9 characters."},
                                           reasoning="The task is to count the characters in the word 'raspberry', which is 9. No shopping tools are relevant.",
                                           summary='Completed the character count.')]),
        is_validated=False,
        is_valid=False,
        validation_attempts=0,
    )
    c.node(state)
