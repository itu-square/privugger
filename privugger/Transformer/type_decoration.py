import ast
import astor
import sys





class FunctionTypeDecorator(ast.NodeTransformer):
    

    """

    Helper function to translate types from python types to theano tensor type
    Parameter: 
    string with the name of the type
    p_type: string

    Return:
    string with name of theano tensor type
    
    """
    
    def translate_type(self, p_type):
        if (p_type == 'float'):
            return 'dscalar'
        
        elif(p_type == 'int'):
            return 'lscalar'
        
        elif(p_type == 'list:int'):
            return 'lvector'
        
        elif(p_type == 'list:float'):
            return 'dvector'

    """
    Function that transforms a list of itypes and a otype from python types to theano tensor types

    """
    def from_python_to_theano_types(self, itypes, otype):
        theano_itypes = []
        for t in itypes:
            theano_itypes.append(self.translate_type(t))
        theano_otype = self.translate_type(otype)
        return (theano_itypes, theano_otype)

    
    """
    This function will turn an output to a theano-friendly type (ex: int -> np.int64)

    """


    def find_return_ast(self, body):
        for i in range(len(body)):
            print(i)
            if( isinstance(body[i], ast.Return)):
                print(i)
                return (body[i],i)
        return None
            
    #Constant(value=1, kind=None)
    #Call(func=Attribute(value=Name(id='np', ctx=Load()), attr='int64', ctx=Load()), args=[Constant(value=1, kind=None)], keywords=[])

    def wrap_output_type(self, out_body, out_type):
        if( isinstance(out_body, ast.Constant)):
            if(out_type == 'int'):
                return ast.Return(ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),attr='int64', ctx=ast.Load()), args=[out_body], keywords=[]))
        if( isinstance(out_body, ast.IfExp)):
            if(out_type == 'int'):
                wrapped_body =  ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),attr='int64', ctx=ast.Load()), args=[out_body.body], keywords=[])
                wrapped_orelse =  ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),attr='int64', ctx=ast.Load()), args=[out_body.orelse], keywords=[])
                out_body.body = wrapped_body
                out_body.orelse = wrapped_orelse
                return ast.Return(out_body)
        return ast.Return(out_body)

    """
    This function decorates a FunctionDef node with tensor types

    """

    def create_decorated_function(self, node, itypes, otype):
        
        number_of_itypes = len(itypes) 
        
        theano_itypes,theano_otype =  self.from_python_to_theano_types(itypes, otype)

        ielts = []
        oelts = []
        for i in range(number_of_itypes):
           attr = ast.Attribute(value=ast.Name(id='tt', ctx=ast.Load()), attr=theano_itypes[i], ctx=ast.Load())
           ielts.append(attr) 
        
        oelts.append(ast.Attribute(value=ast.Name(id='tt', ctx=ast.Load()), attr=theano_otype, ctx=ast.Load()))
        
        theano_keywords = [ast.keyword(arg='itypes', value=ast.List(elts=ielts)), ast.keyword(arg='otypes', value=ast.List(elts=oelts))]

        theano_decorator_list = [ast.Call(func=ast.Attribute(value=ast.Attribute(
        value=ast.Attribute(value=ast.Name(id='theano', ctx=ast.Load()), attr='compile', ctx=ast.Load()), attr='ops', ctx=ast.Load()), attr='as_op', ctx=ast.Load()), args=[], keywords=theano_keywords)]
        
        (return_body, index) = self.find_return_ast(node.body)
        wrapped_return_body = self.wrap_output_type(return_body.value, otype)
        
        node.body[i-1] = wrapped_return_body
        node.decorator_list = theano_decorator_list
        return node 
    """
    The main traversal function that visits all the FunctionDef nodes and returns a decorated FunctionDef

    """

    def visit_FunctionDef(self, node):
        itypes = []
        otype = None


        for a in node.args.args:
            if isinstance(a.annotation, ast.List):
                if (a.annotation.elts[0].id == 'int'):
                    itypes.append('list:int')
                if (a.annotation.elts[0].id == 'float'):
                    itypes.append('list:float')
            else:
                itypes.append(a.annotation.id)
        otype = node.returns.id
        return self.create_decorated_function(node, itypes, otype)


class TheanoImport(ast.NodeTransformer):
    
    """
    This function just blindly puts imports for theano and theano.tensor at top of file
    """
    
    def visit_Module(self, node):
        theano_import = ast.Import(names=[ast.alias(name='theano', asname=None)])
        theano_tensor_import = ast.Import(names=[ast.alias(name='theano.tensor', asname='tt')])
        numpy_import = ast.Import(names=[ast.alias(name='numpy', asname='np')])

        body_with_needed_imports = [theano_import, theano_tensor_import, numpy_import] + node.body
        node.body = body_with_needed_imports
        
        return node


def main():

    filePath = sys.argv[1]

    tree =ast.parse(open(filePath).read())

    new_program = FunctionTypeDecorator().visit(tree)

    new_program_with_imports = TheanoImport().visit(new_program)

    print(astor.to_source(new_program_with_imports))

if __name__ == "__main__":
    main()

