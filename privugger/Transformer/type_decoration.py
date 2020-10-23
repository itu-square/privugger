import ast
import astor
import sys
import argparse



class FunctionTypeDecorator(ast.NodeTransformer):

    function_name = None

    def __init__(self, name):
        self.function_name = name

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
        
        elif(p_type == 'VectorI'):
            return 'lvector'
        
        elif(p_type == 'VectorF'):
            return 'dvector'

        elif(p_type == 'MatrixI'):
            return 'lmatrix'

        elif(p_type== 'MatrixF'):
            return 'fmatrix'

        elif(p_type=='MatrixD'):
            return 'dmatrix'

        elif(p_type=='ListTupleintfloat'):
            return 'listtuplesintfloat'
        
        else:
            raise TypeError("Cannot translate type to any theano type")

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
            if( isinstance(body[i], ast.Return)):
                return (body[i],i)
        return None
            

    def wrap_output_type(self, out_body, out_type):
        if((isinstance(out_body, ast.Name) or isinstance(out_body, ast.Compare))):
            if(out_type == 'int'):
                o_attr = 'int64'
            elif(out_type == 'float'):
                o_attr = 'float32'
            elif(out_type=='VectorI' or out_type == 'VectorF' or out_type == 'MatrixF' or 'MatrixI' or 'MatrixD'):
                o_attr = 'array'
            return ast.Return(ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),attr=o_attr, ctx=ast.Load()), args=[out_body],keywords=[]))
        
        if( isinstance(out_body, ast.Constant)):
            if(out_type == 'int'):
                o_attr = 'int64'
            elif(out_type == 'float'):
                o_attr = 'float32'
            elif(out_type=='VectorI' or out_type == 'VectorF' or out_type == 'MatrixF' or 'MatrixI' or 'MatrixD'):
                o_attr = 'array'
            return ast.Return(ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),attr=o_attr, ctx=ast.Load()), args=[out_body],keywords=[]))
        
        if( isinstance(out_body, ast.IfExp)):
            if(out_type == 'int'):
                o_attr = 'int64'
            elif(out_type == 'float'):
                o_attr = 'float32'
            elif(out_type=='VectorI' or out_type == 'VectorF' or out_type == 'MatrixF' or 'MatrixI' or 'MatrixD'):
                o_attr = 'array'
            wrapped_body =  ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),attr=o_attr, ctx=ast.Load()), args=[out_body.body], keywords=[])
            wrapped_orelse =  ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()),attr=o_attr, ctx=ast.Load()), args=[out_body.orelse], keywords=[])
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
           if(theano_itypes[i] == "listtuplesintfloat"):
               attr = ast.Name(id=theano_itypes[i], ctx=ast.Load())
               ielts.append(attr)
           else:
               attr = ast.Attribute(value=ast.Name(id='tt', ctx=ast.Load()), attr=theano_itypes[i], ctx=ast.Load())
               ielts.append(attr) 
        
        oelts.append(ast.Attribute(value=ast.Name(id='tt', ctx=ast.Load()), attr=theano_otype, ctx=ast.Load()))
        
        theano_keywords = [ast.keyword(arg='itypes', value=ast.List(elts=ielts)), ast.keyword(arg='otypes', value=ast.List(elts=oelts))]

        theano_decorator_list = [ast.Call(func=ast.Attribute(value=ast.Attribute(
        value=ast.Attribute(value=ast.Name(id='theano', ctx=ast.Load()), attr='compile', ctx=ast.Load()), attr='ops', ctx=ast.Load()), attr='as_op', ctx=ast.Load()), args=[], keywords=theano_keywords)]
        
        (return_body, index) = self.find_return_ast(node.body)
        wrapped_return_body = self.wrap_output_type(return_body.value, otype)
        node.body[index] = wrapped_return_body
        node.decorator_list = theano_decorator_list
        return node 
    """
    The main traversal function that visits all the FunctionDef nodes and returns a decorated FunctionDef

    """
    
    #def process_subscript(subscript):



    def get_next_annotation_rec(self, arg, annotation):
        
        if(isinstance(arg, ast.List)):
            return self.get_next_annotation_rec(arg.elts[0], annotation + "List") #annotation.elts[0].id

        if(isinstance(arg, ast.Tuple)):
            if(((arg.elts[0].id == 'int') ) and ((arg.elts[1].id == 'float'))):
                return annotation + "intfloat"
            else:
                raise TypeError("Only support for Tuples(int, float) for now")
        
        if(isinstance(arg, ast.Subscript)):

           return self.get_next_annotation_rec(arg.slice.value, annotation + str(arg.value.id))
        

        else:
            return annotation + str(arg.id)
            


    def visit_FunctionDef(self, node):
        if(isinstance(node, ast.FunctionDef)):
            if(node.name != self.function_name):
                return None
        
        itypes = []
        otype = None
    
        for a in node.args.args:
            annotation = self.get_next_annotation_rec(a.annotation, "")
            
            if(annotation == 'int'):
                itypes.append("int")
           
            elif(annotation == 'float'):
                itypes.append("float")
            
            elif(annotation == "Listint"):
                itypes.append("VectorI")

            elif(annotation == 'Listfloat'):
                itypes.append("VectorF")
            
            elif(annotation == "ListListint"):
                itypes.append("MatrixI")
            
            elif(annotation == "ListListfloat"):
                itypes.append("MatrixF")
            
            elif(annotation == 'ListTupleintfloat'):
                itypes.append("ListTupleintfloat")
             #   itypes.append("MatrixD")
            
            else:
                raise TypeError("Seems that the annotations are wrong")



            #if (isinstance(a.annotation, ast.List)):
             #   if(a.annotation.elts[0].id == 'int'):
              #      itypes.append("VectorI")
               # if(a.annotation.elts[0].id == 'float'):
                #    itypes.append("VectorF")
            #elif(isinstance(a.annotation, ast.Subscript)):
             #   if(a.annotation.value.id == 'List'):
              #      if(isinstance(a.annotation.slice.value, ast.Name)):
               #         if(a.annotation.slice.value.id == 'int'):
                #            itypes.append("VectorI")
                 #       if(a.annotation.slice.value.id == 'float'):
                  #          itypes.append("VectorF")
                    #This is the list of list case
                   # elif(a.annotation.slice.value.value.id == 'List' ):
                    #    if(a.annotation.slice.value.slice.value.id == 'int'):
                     #       itypes.append("MatrixI")
                      #  elif(a.annotation.slice.value.slice.value.id == 'float'):
                       #     itypes.append("MatrixF")
                    #This is list List of tuples case
                    #elif(a.annotation.slice.value.value.id == 'Tuple'):
                     #   itypes.append("MatrixD")
                        #print(a.annotation.slice.value.slice.value)
                        #print(a.annotation.slice.value.slice.value.elts[0].id)
                        #print(a.annotation.slice.value.slice.value.elts[1].id)

               # else:
                #    raise TypeError("The function is not annotated correctly")
                    #print("I Do not know this subscript")
            #else:
             #   itypes.append(a.annotation.id)
        

        if(isinstance(node.returns, ast.List)):
            if((node.returns.elts[0].id) == 'int'):
                otype = 'VectorI'
            if((node.returns.elts[0].id) == 'float'):
                otype = 'VectorF'
        elif(isinstance(node.returns, ast.Subscript)):
            if(node.returns.value.id == 'List'):
                if(isinstance(node.returns.slice.value, ast.Name)):
                    if(node.returns.slice.value.id == 'int'):
                        otype = 'VectorI'
                    elif(node.returns.slice.value.id == 'float'):
                        otype = 'VectorF'
                #This is the list of list case
                elif(node.returns.slice.value.value.id == 'List'):
                    if(node.returns.slice.value.slice.value.id == 'int'):
                        otype = 'MatrixI'
                    elif(node.returns.slice.value.slice.value.id == 'float'):
                        otype = 'MatrixF'
        else:
            otype = node.returns.id
        return self.create_decorated_function(node, itypes, otype)



def find_function_def_idx(body):
    iterable_list = body.body
    for i in range(len(iterable_list)):
        if( isinstance(iterable_list[i], ast.FunctionDef)):
            return (i)
    return -1

def wrap_with_imports(program):
    theano_import = ast.Import(names=[ast.alias(name='theano', asname=None)])
    theano_tensor_import = ast.Import(names=[ast.alias(name='theano.tensor', asname='tt')])
    numpy_import = ast.Import(names=[ast.alias(name='numpy', asname='np')])
    typing_import = ast.ImportFrom(level=0, module='typing', names=[ast.alias(name='List', asname=None), (ast.alias(name='Tuple', asname=None))])
    functools_import = ast.ImportFrom(level=0, module='functools', names=[ast.alias(name='reduce', asname=None)])


    body_with_needed_imports = [theano_import, theano_tensor_import, numpy_import,functools_import, typing_import]+ program.body 
    program.body = body_with_needed_imports

    return program

def wrap_program_with_signature(program):

    idx = find_function_def_idx(program)
    if(idx == -1):
        raise NotImplementedError()
    
    func_name = program.body[idx].name
    func_args = program.body[idx].args
    func_returns = program.body[idx].returns
    arg_identifiers = []
    for a in func_args.args:
        arg_identifiers.append(a.arg)
 
    returns = ast.Return(value=ast.Call(args=[ast.arguments(args=arg_identifiers, defaults=[], vararg=None, kwarg=None)], func=ast.Name(id=func_name, ctx=ast.Load()), keywords=[]))


    new_function = ast.Module(body=[ast.FunctionDef(name='method', decorator_list=[], args=func_args, body=[program.body[idx], returns ], returns=func_returns)])

    return new_function


def load(path, function):
 
    tree =ast.parse(open(path).read())

    #print(ast.dump(tree))

    new_program = FunctionTypeDecorator(function).visit(tree)

    new_program_with_outer_function = wrap_program_with_signature(new_program)


    new_program_with_imports = wrap_with_imports(new_program_with_outer_function)
    #TheanoImport().visit(new_program_with_outer_function)
    print(astor.to_source(new_program_with_imports))


    return new_program_with_imports



