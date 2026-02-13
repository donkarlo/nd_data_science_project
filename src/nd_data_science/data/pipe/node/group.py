from nd_utility.data.kind.group.group import Group as BaseGroup


class Group(BaseGroup):
    def __init__(self, members):
        BaseGroup.__init__(self, members)