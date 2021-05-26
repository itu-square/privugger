


__all__ =[
    "Int",
    "Float",
    "List"
]

class Output: pass

class Int(Output): pass
class Float(Output): pass
class List(Output):
    def __init__(self, output):
        self.output = output
