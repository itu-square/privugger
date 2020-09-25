import ast
import astor
import sys


env = {}



"""
The probabilistic translator class implements all the basic functions for generating the new nodes

All functions return: A node that goes into the new AST


"""



class ProbabilisticTranslator(ast.NodeTransformer):
    """
    A generic "catch all" function that assigns Deterministic to all assigments not caught by other cases
    
    Parameter: 

    target_id: string 

        - the name of the variable 

    Return: Assign Node

    """
    
    def create_generic_assig_node(target_id):
        return ast.Assign(
                    targets=[ast.Name(id=target_id, ctx=ast.Store())],
                    value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='Deterministic', ctx=ast.Load()),
                    args=[ast.Constant(target_id, kind=None), ast.Constant(target_id, kind=None)], keywords=[]))
    
    """
    A function to wrap Deterministic around binary operations (+, -) 

    Parameters:

    target_id: string
        
        - the name of the variable

    value_l: dynamic
        
        - the left parameter of the binary operation
    
    value_r: dynamic
        
        - the right parameter of the binary operation

    bin_op: ast.Op
        
        - the operation to be performed (ex: ast.Add(), ast.Sub()) 

    Return: Assign Node

    """


    def create_deterministic_bin_op__node(target_id, value_l, value_r, bin_op):
            return ast.Assign(
                    targets=[ast.Name(id=target_id, ctx=ast.Store())],
                    value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='Deterministic', ctx=ast.Load()),
                    args=[ast.Constant(target_id, kind=None), ast.BinOp(left=ast.Name(id=value_l, ctx=ast.Load()), op=bin_op,
                    right=ast.Name(id=value_r, ctx=ast.Load()))], keywords=[]))

    """
    TODO: Make it possible to add multiple arguments 

    A function to wrap Deterministic around a function call 

    Parameters:

    target_id: string

        - name of the variable

    func_name: 

        - the name of the function we wrap around

    func_arg:
        
        - the argument provided to the function

    Return: Assign Node

    """

    def create_deterministic_func_call_from_assign(var_name, func_name, func_args):
            f_args = []
            for i in range(len(func_args)):
                f_args.append(ast.Name(id=func_args[i], ctx=ast.Load))
            return ast.Assign(
                    targets=[ast.Name(id = var_name, ctx=ast.Store())],
                    value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='Deterministic', ctx=ast.Load()),
                    args=[ast.Constant(var_name, kind=None), ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()), 
                    args=f_args, keywords=[])], keywords=[], type_ignores=[])
            )



"""

The Assign Deterministic Replacer class if responsible for traversing the AST.

By calling the visit() function all nodes are visited and special actions is taken when encountering an Assign Node

"""


class AssignToDeterministicReplacer(ProbabilisticTranslator):

    """
    Traverses the AST

    Parameters:
    node: AST

     - the entire AST for the input program
    
    Return: The new AST 

    """
    
    def visit_Assign(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute) and node.value.func.value.id == 'typed'):
            print()
            target_id = node.targets[0].id
            typed = node.value.func.value.id
            func = node.value.func.attr
            func_name = typed + "." + func
            
            arg_names = []
            for i in range(len(node.value.args)):
                arg_names.append(node.value.args[i].id)
            return ProbabilisticTranslator.create_deterministic_func_call_from_assign(target_id, func_name, arg_names)

        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and isinstance(node.value.args[0], ast.Name)):
            target_id = node.targets[0].id
            func_name = node.value.func.id
            #arg_name = node.value.args[0].id
            arg_names = []
            for i in range(len(node.value.args)):
                arg_names.append(node.value.args[i].id)
            return ProbabilisticTranslator.create_deterministic_func_call_from_assign(target_id,func_name,arg_names)
        
        if isinstance(node.value, ast.BinOp):
            target_id=node.targets[0].id
            left = node.value.left
            right = node.value.right
            binary_operation = node.value.op
            return ProbabilisticTranslator.create_deterministic_bin_op__node(target_id, left, right, binary_operation)
        else:
            #target_id = node.targets[0].id
            #return ProbabilisticTranslator.create_generic_assig_node(target_id)
            
            #For now this else case needs to return the node itself
            return node


def load(program):
    
    tree = ast.parse(open(program).read())

    #print(ast.dump(tree))
    
    new_ast = AssignToDeterministicReplacer().visit(tree)

    return new_ast




def main():

    """
    The input program is give as a argument to main and it prints out the new source

    TODO: Expose a load() function that takes the program as input and returns the AST of the new program

    """
    
    filePath = sys.argv[1]

    #tree = ast.parse(open(filePath).read())

    #print(ast.dump(tree))

    #new_source = AssignToDeterministicReplacer().visit(tree)
    prob_program = load(filePath)
    print (astor.to_source(prob_program))
    #print("---------------------  Below is the new program --------------------------------")
   
    #print(astor.to_source(prob_program))


if __name__ == "__main__":
    main()

