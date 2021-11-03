import ast
import astor
import sys
import inspect
import os
import privugger.transformer.PyMC3.annotation_types as at
from privugger.transformer.PyMC3.theano_types import TheanoToken


class FunctionTypeDecorator(ast.NodeTransformer):

    function_name = None
    itypes = []
    otypes = []
    has_tuples = False
    has_list_tuples = False

 

    def __init__(self, name=None):
        self.function_name = name

    
    def get_function_def_ast(self, tree):
        
        for i in range(len(tree)):
            if(isinstance(tree[i], ast.FunctionDef)):
                return tree[i]
        raise TypeError("did not find any function definition in program")
    
    
    def get_function_return(self, body):
        
        for ast_node in body:
            if(isinstance(ast_node, ast.Return)):
                return ast_node
        raise TypeError("did not find any Return")

    def simple_method_wrap(self, program, name, args):
        
        arg_identifiers = []
        for a in args.args:
            arg_identifiers.append(a.arg)

        #func_returns = self.get_function_return(program.body)
        returns = ast.Return(value=ast.Call(args=[ast.arguments(args=arg_identifiers, defaults=[], vararg=None, kwarg=None)],func=ast.Name(id=name, ctx=ast.Load()), keywords=[]))
        
        new_function = ast.Module(body=[ast.FunctionDef(name='method', decorator_list=[], args=args, body=[program, returns])])
        return new_function
        #print(astor.to_source(new_function))
        #print(ast.dump(new_function))

    def lift(self, program, decorators):
        """
        This funtion provides another path to program lifting, when the decoration types are given directly. 
        The function lifts the program to be used within a pymc3 model.
         
        Parameters
        ------------
        program: path to program or a string of the entire program
        decorators: list of the decorator types

        Return
        -----------
        Python AST node with the lifted program
        
        """

        #NOTE This is for when the program is given as a path to the file
        if(isinstance(program, str)):
            file = open(program)
            tree = ast.parse(file.read())
            file.close()

        else:
            res = "".join(inspect.getsourcelines(program)[0])
            if "lambda" in res:
                res = res.replace(" ", "")
                start = res.find("lambda")
                end_draws = res.find("draws")
                end_chains = res.find("chains")
                end_cores = res.find("cores")

                #NOTE this is when we only use the default values
                if(end_draws == -1 and end_chains==-1 and end_cores == -1):

                    #TODO There are bugs if the program is defined elsewhere and ends with ")"
                    end = len(res)
                    
                    res = res[start:end-1].strip("\\n")

                else:
                    values = [end_draws, end_chains, end_cores]
                    #NOTE filter all the values that are not set 
                    positive_values = list(filter(lambda x: x != -1, values))
                    end = min(positive_values)
                    res = res[start:end-1].strip("\\n")
                
            if("lambda" == res[:6]):
                form = res.strip().split(":")
                res = f"def function({form[0][6:]}): \n return {form[1]}"

            elif("def" != res[:3]):
                raise TypeError("The program needs to be a path to a file, a lambda or a function")

            with open("temp.py", "w") as file:
                file.write(res)
            tree = ast.parse(open("temp.py").read())

            os.remove("temp.py")
        
        #if("lambda")
        
        
        
        function_def = self.get_function_def_ast(tree.body)

        node = self.create_decorated_function(function_def, decorators[0], decorators[1][0])

        if(isinstance(tree.body[0], ast.Import)):
            imports = tree.body[0]
            wrapped_node = self.simple_method_wrap(node, tree.body[1].name, tree.body[1].args)
            wrapped_node.body.insert(0, imports)
            
            
        else:
            wrapped_node = self.simple_method_wrap(node, tree.body[0].name, tree.body[0].args)
        
        return wrapped_node

    
    def translate_type(self, p_type):
        """

        Helper function to translate types from python types to theano tensor type
        Parameter: 
        string with the name of the type
        p_type: string

        Return:
        string with name of theano tensor type
        
        """
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

        elif(p_type=='Single_element_VectorF'):
            return 'TensorType(\'float64\', (True,))'

        elif(p_type=='Single_element_VectorI'):
            return 'TensorType(\'int64\', (True,))'
        
        else:
            raise TypeError("Cannot translate type to any theano type")

    def from_python_to_theano_types(self, itypes, otype):

        """
        Function that transforms a list of itypes and a otype from python types to theano tensor types

        """
        theano_itypes = []
        for t in itypes:
            theano_itypes.append(self.translate_type(t))
        theano_otype = self.translate_type(otype)
        return (theano_itypes, theano_otype)



    def find_return_ast(self, body):
                
        """
        This function will find the return statements in original program

        """
        return_list = []
        for i in range(len(body)):
            if(isinstance(body[i], ast.Return)):
                return_list.append((body[i], i))
            else:
                pass
                #print(body[i])
        if(len(return_list) > 0):
            return return_list
        else:
           return None 

    def wrap_output_type(self, out_body, out_type):
        if((isinstance(out_body, ast.Name) or isinstance(out_body, ast.Compare) or isinstance(out_body, ast.ListComp) or isinstance(out_body, ast.BinOp) )):
            if(out_type == 'int'):
                o_attr = 'int64'
            #elif(out_type == 'float'):
             #   o_attr = 'float64'
            elif(out_type=='VectorI' or out_type == 'VectorF' or out_type == 'MatrixF' or 'MatrixI' or 'MatrixD' or 'float'):
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
            return (ast.Return(out_body))
        
        return (ast.Return(out_body))


    def create_decorated_function(self, node, itypes, otype):

        """
        This function decorates a FunctionDef node with tensor types

        """
        
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
        
        #return_body, index = self.find_return_ast(node.body)
        return_list = self.find_return_ast(node.body)
        if(return_list is not None):
            for r in return_list:
                wrapped_return_body = self.wrap_output_type(r[0].value, otype)
                node.body[r[1]] = wrapped_return_body
        node.decorator_list = theano_decorator_list
        return node 


    def get_next_annotation(self, arg):
        
        if(isinstance(arg, ast.List)):
            l = at.List()
            l.a_type = self.get_next_annotation(arg.elts[0])
            return (l)

        elif(isinstance(arg, ast.Tuple)):
    
            t = at.Tuple()
            t.length = len(arg.elts) 
            for i in range(t.length):
                t.a_type.append(self.get_next_annotation(arg.elts[i]))
            return (t)

        
        elif(isinstance(arg, ast.Subscript)):

            if(arg.value.id == "List"):
                l = at.List()
                l.a_type = self.get_next_annotation(arg.slice.value)
                return l
            else:
                return self.get_next_annotation(arg.slice.value)
          
        else:
            if(arg.id == 'int'):
                return (at.Int())
            if(arg.id == 'float'):
                return (at.Float())
            else:
        
                raise TyperError("Seems that there is a problem with the annotation")

    def visit_FunctionDef(self, node):
        """
        The main traversal function that visits all the FunctionDef nodes and returns a decorated FunctionDef

        """
        if(isinstance(node, ast.FunctionDef)):
            if(node.name != self.function_name):
                return None
        
        itypes = []
        otype = None
    
        for a in node.args.args:
            #annotation = self.get_next_annotation_rec(a.annotation, "")
            annotation_type = self.get_next_annotation(a.annotation)
            if(type(annotation_type) == at.List):
                if(type(annotation_type.a_type) == at.Int):
                    #itypes.append("VectorI")
                    itypes.append(TheanoToken.int_vector)
                elif(type(annotation_type.a_type) == at.Float):
                    #itypes.append("VectorF")
                    itypes.append(TheanoToken.float_vector)
                elif(type(annotation_type.a_type) == at.Tuple):
                    self.has_list_tuples = True
                    for t in annotation_type.a_type.a_type:
                        if(type(t) == at.Int):
                            itypes.append(TheanoToken.int_vector)
                            #itypes.append("VectorI")
                        if(type(t) == at.Float):
                            #itypes.append("VectorF")
                            itypes.append(TheanoToken.float_vector)
        
            elif(type(annotation_type) == at.Int):
                #itypes.append("int")
                itypes.append(TheanoToken.int_scalar)
            
            elif(type(annotation_type) == at.Float):
                #itypes.append("float")
                itypes.append(TheanoToken.float_scalar)
            
            elif(type(annotation_type) == at.Tuple):
                self.has_tuples = True
                for t in annotation_type.a_type:
                    if(type(t) == at.Int):
                        #itypes.append("VectorI")
                        itypes.append(TheanoToken.int_vector)
                    if(type(t) == at.Float):
                            #itypes.append("VectorF")
                            itypes.append(TheanoToken.float_vector)
         
            else:
                raise TypeError("Seems that the annotations are wrong")
            
        

        out_annotation_type = self.get_next_annotation(node.returns)
        if(type(out_annotation_type) == at.List):
            if(type(out_annotation_type.a_type) == at.List):
                if(out_annotation_type.a_type.a_type == at.Int):
                    outype = TheanoTokoen.int_matrix
                else:
                    otype = TheanoToken.float_matrix
            elif(type(out_annotation_type.a_type) == at.Tuple):
                all_ints = all(isinstance(t, at.Int) for t in out_annotation_type.a_type.a_type)
                if all_ints:
                    otype = TheanoToken.int_matrix
                else:
                    otype = TheanoToken.float_matrix
            elif(type(out_annotation_type.a_type) == at.Int):
                otype = TheanoToken.int_scalar
            elif(type(out_annotation_type.a_type) == at.Float):
                otype = TheanoToken.float_scalar
        elif(type(out_annotation_type) == at.Int):
            otype = (TheanoToken.int_scalar)
            
        elif(type(out_annotation_type) == at.Float):
            otype = (TheanoToken.float_scalar)
            
        elif(type(out_annotation_type) == at.Tuple):
            all_ints = all(isinstance(t, at.Int) for t in out_annotation_type.a_type)
            if(all_ints):
                otype = TheanoToken.int_matrix
            else:
                otype = TheanoToken.float_matrix
         
        else:
            raise TypeError("Seems that the annotations are wrong")

        self.itypes = itypes
        self.otypes = otype
    
        decorated_func = self.create_decorated_function(node, itypes, otype)    
        return ((decorated_func))




    def find_function_def_idx(self, body):
        iterable_list = body.body
        for i in range(len(iterable_list)):
            if( isinstance(iterable_list[i], ast.FunctionDef)):
                return (i)
        return (-1)

    
    def wrap_with_theano_import(self, program):

        theano_import = ast.Import(names=[ast.alias(name='theano', asname=None)])
        theano_tensor_import = ast.Import(names=[ast.alias(name='theano.tensor', asname='tt')])
        numpy_import = ast.Import(names=[ast.alias(name='numpy', asname='np')])
        new_program = ast.Module(body=[theano_import, theano_tensor_import,numpy_import,  program])
        
        return new_program

    def wrap_with_imports(self, program):
        theano_import = ast.Import(names=[ast.alias(name='theano', asname=None)])
        theano_tensor_import = ast.Import(names=[ast.alias(name='theano.tensor', asname='tt')])
        numpy_import = ast.Import(names=[ast.alias(name='numpy', asname='np')])
        typing_import = ast.ImportFrom(level=0, module='typing', names=[ast.alias(name='List', asname=None), (ast.alias(name='Tuple', asname=None))])
        functools_import = ast.ImportFrom(level=0, module='functools', names=[ast.alias(name='reduce', asname=None)])


        body_with_needed_imports = [theano_import, theano_tensor_import, numpy_import,functools_import, typing_import]+ program.body 
        program.body = body_with_needed_imports

        return program


    def construct_python_args(self):

        new_args = []

        for i in range(len(self.itypes)):
            new_annotation = None

            if(self.itypes[i] == 'VectorI'):
                new_annotation = ast.Subscript(value=ast.Name(id = "List",ctx = ast.Load()),slice=ast.Name(id='int', ctx=ast.Load()))
            if(self.itypes[i]  == 'VectorF'):
                new_annotation = ast.Subscript(value=ast.Name(id = "List",ctx = ast.Load()),slice=ast.Name(id='float', ctx=ast.Load()))
            if(self.itypes[i]  == 'int'):
                new_annotation = ast.Name(id='int', cts=ast.Load())
            if(self.itypes[i]  == 'float'):
                new_annotation = ast.Name(id='float', cts=ast.Load())
            if(self.itypes[i]  == 'MatrixI'):
                pass
            if(self.itypes[i]  == 'MatrixF'):
                pass
            new_args.append(ast.arg(arg=f'arg_{i}',annotation=new_annotation))

        return ast.arguments(args=new_args, defaults=[], vararg=None, kwarg=None)

    def construct_python_body(self, body, new_args, new_args_idx, original_name):
        
        value_names = []

        for i in new_args_idx:
            value_names.append(ast.Name(id = new_args.args[i].arg, ctx=ast.Load()))
        
        body.args = new_args
        if(self.has_tuples and not self.has_list_tuples):
            temp = [ast.Assign(targets=[ast.Name(id=original_name, ctx=ast.Load())], value=ast.Call(args=value_names, func=ast.Name(id='tuple', ctx=ast.Load()), keywords=[]))] + body.body
            body.body = temp    
        if(self.has_list_tuples):
            temp = [ast.Assign(targets=
                    [ast.Name(id=original_name, ctx=ast.Load())],
                    value=ast.Call(args=[ast.Call(args=value_names, 
                    func=ast.Name(id='zip', ctx=ast.Load()), keywords=[])], func=ast.Name(id='list', ctx=ast.Load()), keywords=[]))] + body.body
            body.body = temp

        return body

    def wrap_program_with_signature(self, program):

        idx = self.find_function_def_idx(program)
        if(idx == -1):
            raise NotImplementedError()
        
        func_name = program.body[idx].name
        func_args = program.body[idx].args

        if(self.has_tuples or self.has_list_tuples):

            new_args = self.construct_python_args()
            new_args_idx = []
            original_name = None
            for i in range(len(new_args.args)):
                new = new_args.args[i].annotation
                if(not i >= len(func_args.args)):
                    old = func_args.args[i].annotation
                    if(isinstance(old, ast.Subscript) and not isinstance(new, ast.Subscript)):
                        new_args_idx.append(i)
                    
                    if(isinstance(old, ast.Subscript)):
                        if(isinstance(old.slice.value, ast.Tuple)):
                            new_args_idx.append(i)
                            original_name = func_args.args[i].arg
                        elif(old.slice.value.value.id == "Tuple"):
                            new_args_idx.append(i)
                            original_name = func_args.args[i].arg

                else:
                    new_args_idx.append(i)
                    # print(new_args_idx)


            new_body = self.construct_python_body(program.body[idx], new_args, new_args_idx, original_name)
        else:
            new_args = func_args
            new_body = program.body[idx]

  
        #Check every arguments, to see if it contains a tuple or a list of tuples
        # for i in range(len(func_args.args)):
        #     idx_has_tuple_arguments.append((i, has_tuple(func_args.args[i].annotation), has_list(func_args.args[i].annotation)))
        # print(idx_has_tuple_arguments)

        func_returns = program.body[idx].returns
        
        arg_identifiers = []
        for a in new_args.args:
            arg_identifiers.append(a.arg)
               
    
        returns = ast.Return(value=ast.Call(args=[ast.arguments(args=arg_identifiers, defaults=[], vararg=None, kwarg=None)], func=ast.Name(id=func_name, ctx=ast.Load()), keywords=[]))

        #program.body[idx]
        new_function = ast.Module(body=[ast.FunctionDef(name='method', decorator_list=[], args=new_args, body=[new_body, returns ], returns=func_returns)])

        return (new_function)


def load(path, function):
 
    tree =ast.parse(open(path).read())

    #print(ast.dump(tree))
    ftp = FunctionTypeDecorator(function)

    new_program = ftp.visit(tree)

    new_program_with_outer_function = ftp.wrap_program_with_signature(new_program)

    new_program_with_imports = ftp.wrap_with_imports(new_program_with_outer_function)
    #TheanoImport().visit(new_program_with_outer_function)
    print(astor.to_source(new_program_with_imports))


    return new_program_with_imports



