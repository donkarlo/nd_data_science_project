from typing import List

from nd_data_science.data.pipe.node.group import Group as NodeGroup
from nd_data_science.data.pipe.node.node import Node
import numpy as np

class Pipe:
    def __init__(self, node_group: NodeGroup, data: np.ndarray):
        self._node_group = node_group

    @classmethod
    def init_by_nodes_list(cls, nodes_lits: List[Node]) -> None:
        pass

    def start(self):
        pass
