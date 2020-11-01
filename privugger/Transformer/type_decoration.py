import ast
import astor
import sys
import argparse
import privugger.privugger.Transformer.annotation_types as at
from privugger.privugger.Transformer.theano_types import TheanoToken


class FunctionTypeDecorator(ast.NodeTransformer):

    function_name = None
    itypes = []
    otypes = []
    has_tuples = False
    has_list_tuples = False

 

    def __init__(self, name):
        self.function_name = name

 
    
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
        This function will turn an output to a theano-friendly type (ex: int -> np.int64)

        """
        for i in range(len(body)):
            if( isinstance(body[i], ast.Return)):
                return (body[i],i)
        return (None)
            

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
        
        (return_body, index) = self.find_return_ast(node.body)
        wrapped_return_body = self.wrap_output_type(return_body.value, otype)
        node.body[index] = wrapped_return_body
        node.decorator_list = theano_decorator_list
        return (node) 


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
                otype = TheanoToken.float_matrix
            elif(type(out_annotation_type.a_type) == at.Tuple):
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
            otype = TheanoToken.float_matrix
         
        else:
            raise TypeError("Seems that the annotations are wrong")


        # print(out_annotation_type)
        # print(out_annotation_type.a_type)
        # if(isinstance(node.returns, ast.List)):
        #     if((node.returns.elts[0].id) == 'int'):
        #         otype = 'VectorI'
        #     if((node.returns.elts[0].id) == 'float'):
        #         otype = 'VectorF'
        # elif(isinstance(node.returns, ast.Subscript)):
        #     if(node.returns.value.id == 'List'):
        #         if(isinstance(node.returns.slice.value, ast.Name)):
        #             if(node.returns.slice.value.id == 'int'):
        #                 otype = 'VectorI'
        #             elif(node.returns.slice.value.id == 'float'):
        #                 otype = 'VectorF'
        #         #This is the list of list case
        #         elif(node.returns.slice.value.value.id == 'List'):
        #             if(node.returns.slice.value.slice.value.id == 'int'):
        #                 otype = 'MatrixI'
        #             elif(node.returns.slice.value.slice.value.id == 'float'):
        #                 otype = 'MatrixF'
        # else:
        #     otype = node.returns.id
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



    def split_tuple(self, arg):
        pass

    def split_list_tuples(self, arg):
        pass


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
                    # if(annotation == 'int'):
            #     itypes.append("int")
           
            # elif(annotation == 'float'):
            #     itypes.append("float")
            
            # elif(annotation == "Listint"):
            #     itypes.append("VectorI")

            # elif(annotation == 'Listfloat'):
            #     itypes.append("VectorF")
            
            # elif(annotation == "ListListint"):
            #     itypes.append("MatrixI")
            
            # elif(annotation == "ListListfloat"):
            #     itypes.append("MatrixF")
            
            # elif(annotation.startswith("Tuple")):
            #     self.has_tuples = True
            #     intidx = -1
            #     floatidx = -1
            #     if("int" in annotation):
            #         intidx = annotation.index("int",0)
            #     if("float" in annotation):
            #         floatidx = annotation.index("float",0)
            #     if(not (floatidx  == -1) and not (intidx  == -1)):
            #         if(intidx <= floatidx):
            #             itypes.append("int")
            #             itypes.append("float")
            #         else:
            #             itypes.append("float")
            #             itypes.append("int")
            #     elif(not (floatidx == -1)):
            #         itypes.append("float")

            #     elif(not (intidx == -1)):
            #         itypes.append("int")    
             
            
            # elif(annotation.startswith("ListTuple")):
            #     self.has_list_tuples  = True
            #     intidx = -1
            #     floatidx = -1
            #     if("int" in annotation):
            #         intidx = annotation.index("int",0)
            #     if("float" in annotation):
            #         floatidx = annotation.index("float",0)
            #     if(not (floatidx  == -1) and not (intidx  == -1)):
            #         if(intidx <= floatidx):
            #             itypes.append("VectorI")
            #             itypes.append("VectorF")
            #         else:
            #             itypes.append("VectorF")
            #             itypes.append("VectorI")
            #     elif(not (floatidx == -1)):
            #         itypes.append("VectorF")

            #     elif(not (intidx == -1)):
            #         itypes.append("VectorI")                        
            
            # else:
            #     raise TypeError("Seems that the annotations are wrong")



    # def get_next_annotation_rec(self, arg, annotation):
        
    #     if(isinstance(arg, ast.List)):
 
    #         return self.get_next_annotation_rec(arg.elts[0], annotation + "List") #annotation.elts[0].id

    #     if(isinstance(arg, ast.Tuple)):
    #         #if(((arg.elts[0].id == 'int') ) and ((arg.elts[1].id == 'float'))):
    #          #   return annotation + "intfloat"
    #         if(len(arg.elts) != 2):
    #             raise TypeError("Only support for tuples with 2 types for now")
    #         else:
        
    #             return self.get_next_annotation_rec(arg.elts[0], annotation + "Tuple" ) + self.get_next_annotation_rec(arg.elts[1], annotation)
        
    #     if(isinstance(arg, ast.Subscript)):
   

    #        return self.get_next_annotation_rec(arg.slice.value, annotation  + str(arg.value.id))
          
    #     else:
    #         return (annotation + str(arg.id))
    