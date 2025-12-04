from erc.state import AgentState


class BaseExpert(object):

    def __init__(self, persona_file):
        with open(persona_file, "r") as f:
            self.persona = f.read()

    def node(self, state: AgentState):
        raise NotImplementedError()
