from erc.state import AgentState


class BaseExpert(object):

    def node(self, state: AgentState):
        raise NotImplementedError()
