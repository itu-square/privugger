import ast
import inspect
import os
import privugger as pv
import astor
import numpy as np

class FunctionDeterministic(ast.NodeTransformer):

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
        
    def translate_program(self, tree):
        """
        This function translates the program line by line to be understable by the pymc_whitebox model.
         
        Parameters
        ------------
        tree : ast.tree
            abstract syntax tree of the function we sample

        Return
        -----------
        tree.body : ast node
            body of our new function
        file_import : list
            name of the new files created that need to be import in the fucntion file.
        
        """
        file_import = []
        constant = None

        for i in range (0, len(tree.body)):

            if type(tree.body[i]) is ast.FunctionDef:
                tree.body[i].name = 'method'
                j = 0
                lign = 1
                while j < len(tree.body[i].body):
                        
                    if type(tree.body[i].body[j]) is ast.Assign:
                        ################################################
                
                        ###             ASSIGN HANDELING             ###

                        ################################################
                        if type(tree.body[i].body[j].value) is ast.Constant:
                            ###         ASSIGN TO A CONSTANT         ###
                            constant = None
                            self.add_constant(tree, j , i, constant, lign)
                            j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j].targets[0].id)
                        
                        elif type(tree.body[i].body[j].value) is ast.Name:
                            ###      ASSIGN TO A KNOWN VARIABLE      ###
                            self.add_name(tree, j , i)
                            j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j].targets[0].id)
                            
                        elif type(tree.body[i].body[j].value) is ast.Call:
                            ###         ASSIGN TO A FUNCTION         ###
                            library, function_attributes = self.get_function_name(tree.body[i].body[j].value)

                            call = self.add_call(tree.body[i].body[j], lign, file_import, library, function_attributes)
                            
                            if call[4] == True:
                                file_import.append(call[0])
                                file_import.append(call[1])

                            if len(call[2]) > 0:
                                tree.body[i].body[j] = call[2][0]

                                if len(call[2]) > 1:
                                    for k in range (1, len(call[2])):
                                        tree.body[i].body.insert(j+1, call[2][k])
                                        j += 1

                                tree.body[i].body.insert(j+1, call[3])
                                j += 1

                            else:
                                tree.body[i].body[j] = call[3]

                            j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j].targets[0].id)

                        elif type(tree.body[i].body[j].value) is ast.BinOp:
                            ###     ASSIGN TO BINERAL OPERATION     ###
                            if type(tree.body[i].body[j].value.left) is ast.Name and type(tree.body[i].body[j].value.right) is ast.Name:
                                ###     WITH ALREADY KNOWN VARIABLES    ###
                                j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j].targets[0].id)

                            elif type(tree.body[i].body[j].value.right) is ast.Call:
                                ###  WITH A FUNCTION ON THE RIGHT SIDE  ###
                                library, function_attributes = self.get_function_name(tree.body[i].body[j].value.right)

                                binop = tree.body[i].body[j]
                                
                                call = self.add_call(tree.body[i].body[j].value.right, lign, file_import, library, function_attributes, assign=False)
                                
                                if call[4] == True:
                                    file_import.append(call[0])
                                    file_import.append(call[1])
                            
                                tree.body[i].body[j] = call[2][0]

                                if len(call[2]) > 1:
                                    for k in range (1, len(call[2])):
                                        tree.body[i].body.insert(j+1, call[2][k])
                                        j += 1

                                tree.body[i].body.insert(j+1, call[3])
                                j += 1

                                j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j].targets[0].id)

                                binop.value.right = ast.Name(id=call[3].targets[0].id, ctx=ast.Load())

                                tree.body[i].body.insert(j, binop)

                            elif type(tree.body[i].body[j].value.left) is ast.Call:
                                ###  WITH A FUNCTION ON THE LEFT SIDE  ###
                                library, function_attributes = self.get_function_name(tree.body[i].body[j].value.left)
                                
                                binop = tree.body[i].body[j]
                                
                                call = self.add_call(tree.body[i].body[j].value.left, lign, file_import, library, function_attributes, assign=False)
                                
                                if call[4] == True:
                                    file_import.append(call[0])
                                    file_import.append(call[1])
                            
                                tree.body[i].body[j] = call[2][0]

                                if len(call[2]) > 1:
                                    for k in range (1, len(call[2])):
                                        tree.body[i].body.insert(j+1, call[2][k])
                                        j += 1

                                tree.body[i].body.insert(j+1, call[3])
                                j += 1

                                j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j].targets[0].id)

                                binop.value.left = ast.Name(id=call[3].targets[0].id, ctx=ast.Load())

                                tree.body[i].body.insert(j, binop)

                            elif type(tree.body[i].body[j].value.left) is ast.Constant:
                                constant = 'left'
                                self.add_constant(tree, j , i, constant, lign)
                                tree.body[i].body[j+1].value.left = ast.Name(id=tree.body[i].body[j].targets[0].id, ctx=ast.Load())
                                j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j+1].value.left.id)

                            elif type(tree.body[i].body[j].value.right) is ast.Constant:
                                constant = 'right'
                                self.add_constant(tree, j , i, constant, lign)
                                tree.body[i].body[j+1].value.right = ast.Name(id=tree.body[i].body[j].targets[0].id, ctx=ast.Load())
                                j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j+1].value.right.id)
                        else :
                            j += 1

                    elif type(tree.body[i].body[j]) is ast.Expr:
                        ################################################
                
                        ###             METHOD HANDELING             ###

                        ################################################

                        if tree.body[i].body[j].value.func.attr == "append":

                            j = self.add_deterministic(tree, j, i, lign, tree.body[i].body[j].value.args[0].id)

                    else:
                        j += 1

                    lign += 1

        return tree.body, file_import


    def add_function(self, tree, function_attributes, library):
        """
        This function creates a new file, with the use of the pymc3 model, for the function that we use in the ast.Call node.
         
        Parameters
        ------------
        tree : tree : ast.tree
            abstract syntax tree of the function we sample
        function_attributes : List
            The different attributes of the function
        library : List
            The abbreviation of the function's library

        Return
        -----------
        return True
        
        """
        f = open("%s.py" %function_attributes[0], "w")

        output = self.get_output_type(function_attributes, library)

        args_functiondef = []
        args_function = []
        input_specs = []
        for i in range (0, len(tree.args)):
            args_functiondef.append(ast.arg(arg='arg%d' %i))
            args_function.append(ast.Name(id='arg%d' %i, ctx=ast.Load()))
            if isinstance(tree.args[i].value, int):
                input_specs.append(pv.Constant('y', np.int64(0)))
            elif isinstance(tree.args[i].value, float):
                input_specs.append(pv.Constant('y', np.float64(0)))
        
        tree.args = args_function

        if output == pv.Int:
            prog = ast.FunctionDef(name=function_attributes[0], args=ast.arguments(posonlyargs=[], args=args_functiondef, kwonlyargs=[], kw_defaults=[], defaults=[]), body=[ast.Assign(targets=[ast.Name(id='y', ctx=ast.Store())], value=tree), ast.Return(value=ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='array', ctx=ast.Load()), args=[ast.Name(id='y', ctx=ast.Load())], keywords=[]))], decorator_list=[])
        elif output == pv.Float:
            prog = ast.FunctionDef(name=function_attributes[0], args=ast.arguments(posonlyargs=[], args=args_functiondef, kwonlyargs=[], kw_defaults=[], defaults=[]), body=[ast.Assign(targets=[ast.Name(id='y', ctx=ast.Store())], value=tree), ast.Return(value=ast.Name(id='y', ctx=ast.Load()))], decorator_list=[])
        
        f.write(astor.to_source(prog))
        f.close()

        method = getattr(__import__(function_attributes[0], fromlist=[function_attributes[0]]), function_attributes[0])

        ds = pv.Dataset(input_specs)

        program   = pv.Program('%s' %function_attributes[0], dataset=ds, output_type=output, function=method)

        pv.infer(program, cores=4, draws=10000, method='pymc3', return_model=True)

        return True

    def add_deterministic(self, tree, j, i, lign, variable):
        """
        This function adds a deterministic variable for the line j of the program
         
        Parameters
        ------------
        tree : tree : ast.tree
            abstract syntax tree of the function we sample
        j : int
            number of lines in the new function
        i : int
            specify in which node the line is
        lign : int
            specify which line we are translating
        variable : string
            name of the variable

        Return
        -----------
        return the numer of lines in the program
        
        """
        deterministic = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='Deterministic', ctx=ast.Load()), args=[ast.Constant(value='lign%d' %lign), ast.Name(id=variable, ctx=ast.Load())], keywords=[]))
        tree.body[i].body.insert(j+1, deterministic)
        j += 2
        
        return j

    def add_constant(self, tree, j , i, constant, lign) :
        """
        This function translates an ast.Constant node to be understandable by the pymc_whitebox model
         
        Parameters
        ------------
        tree : tree : ast.tree
            abstract syntax tree of the function we sample
        j : int
            number of lines in the new function
        i : int
            specify in which node the line is
        constant : string
            left if from the left side of a bineral operation, same for the right side. And None if neither left or right
        lign : int
            correspond to number of the line we are currently translating

        Return
        -----------
        return True
        
        """
        
        if constant == None : 
            new_tree = tree.body[i].body[j]
            type_var = type(new_tree.value.value)
            id_var = new_tree.targets[0].id
            value_var = new_tree.value.value
            if type_var is int:
                tree.body[i].body[j] = ast.Assign(targets=[ast.Name(id=id_var, ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='DiracDelta', ctx=ast.Load()), args=[ast.Constant(value=id_var), ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='int64', ctx=ast.Load()), args=[ast.Constant(value=value_var)], keywords=[])], keywords=[]))
            elif type_var is float:
                tree.body[i].body[j] = ast.Assign(targets=[ast.Name(id=id_var, ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='DiracDelta', ctx=ast.Load()), args=[ast.Constant(value=id_var), ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64', ctx=ast.Load()), args=[ast.Constant(value=value_var)], keywords=[])], keywords=[]))
            return True
        elif constant == 'left': 
            new_tree = tree.body[i].body[j].value.left
            type_var = type(new_tree.value)
            id_var = 'const%d'%lign
            value_var = new_tree.value
        elif constant == 'right': 
            new_tree = tree.body[i].body[j].value.right
            type_var = type(new_tree.value)
            id_var = 'const%d'%lign
            value_var = new_tree.value

        if type_var is int:
            constant = ast.Assign(targets=[ast.Name(id=id_var, ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='DiracDelta', ctx=ast.Load()), args=[ast.Constant(value=id_var), ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='int64', ctx=ast.Load()), args=[ast.Constant(value=value_var)], keywords=[])], keywords=[]))
        elif type_var is float:
            constant = ast.Assign(targets=[ast.Name(id=id_var, ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='DiracDelta', ctx=ast.Load()), args=[ast.Constant(value=id_var), ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64', ctx=ast.Load()), args=[ast.Constant(value=value_var)], keywords=[])], keywords=[]))
        
        tree.body[i].body.insert(j, constant)

        return True

    def add_name(self, tree, j , i) :
        """
        This function translates an ast.Name node to be understandable by the pymc_whitebox model
         
        Parameters
        ------------
        tree : tree : ast.tree
            abstract syntax tree of the function we sample
        j : int
            specify which line we are translating
        i : int
            specify in which node the line is
        lign : int
            correspond to number of the line we are currently translating
            
        Return
        -----------
        return True
        
        """
        tree.body[i].body[j] = ast.Assign(targets=[ast.Name(id=tree.body[i].body[j].targets[0].id, ctx=ast.Store())], value=ast.Name(id=tree.body[i].body[j].value.id, ctx=ast.Load()))
        return True

    def add_call(self, tree, lign, file_import, library, function_attributes, assign=True) :
        """
        This function translates an ast.Call node to be understandable by the pymc_whitebox model
         
        Parameters
        ------------
        tree : tree : ast.tree
            abstract syntax tree of the function we sample
        lign : int
            correspond to number of the line we are currently translating
        file_import : List
            list of all the files already created and their abbreviation
        library : List
            The abbreviation of the function's library
        function_attributes : List
            The different attributes of the function
        assign : Boolean
            True if the ast.Call node is inside an ast.Assign node

        Return
        -----------
        function_attributes[0] : string
            name of the function
        abbre : string
            abbreviation of the import file
        variable_declare : List
            list of the variables that need to be declare as aesara variables
        function_call : ast node
            ast.node which contains the call to the function
        new : Boolean
         specify if the function is a new one or not
        
        """
        if assign == True:
            old_variable = tree.targets[0].id
            value_arg = []
            if len(tree.value.args) > 0:
                for i in range (0, len(tree.value.args)):
                    value_arg.append(tree.value.args[i].value)
            self.add_function(tree.value, function_attributes, library)

        elif assign == False:
            old_variable = 'var%d' %lign
            value_arg = []
            if len(tree.args) > 0:
                for i in range (0, len(tree.args)):
                    value_arg.append(tree.args[i].value)
            self.add_function(tree, function_attributes, library)
            
        abbre = self.existing_abbre(file_import, function_attributes[0])

        if abbre == -1:
            abbre = self.new_abbre(file_import, function_attributes[0])
            new = True
        else:
            abbre = file_import[abbre+1]
            new = False

        variable_declare = []
        args_function = []
        if len(value_arg) > 0:
            for j in range (0, len(value_arg)): 
                variable_declare.append(ast.Assign(targets=[ast.Name(id='var%d%d'%(lign,j), ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name(id='pm', ctx=ast.Load()), attr='DiracDelta', ctx=ast.Load()), args=[ast.Constant(value='var%d%d'%(lign,j)), ast.Call(func=ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='int64', ctx=ast.Load()), args=[ast.Constant(value=value_arg[j])], keywords=[])], keywords=[])))
                args_function.append(ast.Name(id='var%d%d'%(lign,j), ctx=ast.Load()))

        function_call = ast.Assign(targets=[ast.Name(id=old_variable, ctx=ast.Store())], value=ast.Call(func=ast.Attribute(value=ast.Name(id=abbre, ctx=ast.Load()), attr='method', ctx=ast.Load()), args=args_function, keywords=[]))                                

        return function_attributes[0], abbre, variable_declare, function_call, new

    def new_abbre(self, file_import, name):
        """
        This function creates an abbreviation for the file while avoiding creating a duplicate
         
        Parameters
        ------------
        file_import : List
            list of all the files already created and their abbreviation
        name : string
            name of the function

        Return
        -----------
        abbre : string
            abbreviation of the import file
        
        """
        for k in range (0, len(name)):
            dupli = 0
            for l in range (0, len(file_import)):
                if name[k:k+2] == file_import[l]:
                    dupli += 1
            if dupli == 0:
                abbre = name[k:k+2]
                break
        return abbre

    def existing_abbre(self, file_import, name):
        """
        This function verifies if a file related to that function has already been created
         
        Parameters
        ------------
        file_import : List
            list of all the files already created and their abbreviation
        name : string
            name of the function

        Return
        -----------
        -1 if yes
        
        k if no
        
        """
        for k in range (0, len(file_import)):
            if name == file_import[k]:
                return k
        return -1

    def translate(self, program):
        """
        This function translates the program to be used within a pymc_whitebox model.
         
        Parameters
        ------------
        program: path to program or a string of the entire program

        Return
        -----------
        Python AST node with the translated program
        
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
        
        function_w_pymc = self.translate_program(tree)
        
        function_def = self.get_function_def_ast(function_w_pymc[0])

        return function_def, function_w_pymc[1]
    
    def wrap_with_aesara_import(self, program, file_import):
        """
        Add the necessary imports to the new file created

        Parameters
        ----------

        program : ast node
            Python AST node with the lifted program
        file_import : List
            Which contains the files created and their import abbreviation that need to be add in the main program

        Return 
        ------

        new_program : ast node
            lifted program with imports added
        """
        all_imports = []
        numpy_import = ast.Import(names=[ast.alias(name='numpy', asname='np')])
        pymc_import = ast.Import(names=[ast.alias(name='pymc', asname='pm')])
        aesara_import = ast.Import(names=[ast.alias(name='aesara.tensor', asname='at')])
        all_imports.append(numpy_import)
        all_imports.append(pymc_import)
        all_imports.append(aesara_import)
        i = 0
        while i < len(file_import):
            all_imports.append(ast.Import(names=[ast.alias(name=file_import[i], asname=file_import[i+1])]))
            i += 2
        all_imports.append(program)
        new_program = ast.Module(body=all_imports)

        
        return new_program




    def get_output_type(self, function_attributes, library):
        """
        Collect the output type of the function

        Parameters
        ----------

        function_attributes : List
            The different attributes of the function
        library : List
            The abbreviation of the function's library

        Return 
        ------

        output_type : privugger type
            output type of the functtion
        
        Example : for np.random.randint
                  output_type = pv.Int
        """
        handled_functions = ["np.random.randint", pv.Int, "np.random.uniform", pv.Float, ]
        function = library[0]
        for i in range (len(function_attributes), 0, -1):
            function = function + "." + function_attributes[i-1]
        output_type = ""
        for j in range (0, len(handled_functions)):
            if function == handled_functions[j]:
                output_type = handled_functions[j+1]
        return output_type


    def get_function_name(self, tree):
        """
        Collect the name of the function

        Parameters
        ----------
        tree : ast.tree
            abstract syntax tree of the function we sample

        Return 
        ------

        library : List
            The abbreviation of the function's library
        function_attributes : List
            The different attributes of the function
        
        Example : for np.random.randint
                  library = [np]
                  function_attributes = [randint, random]
        """
        library = []
        function_attributes = []
        try:
            function_attributes.append(tree.func.attr)
            try:
                function_attributes.append(tree.func.value.attr)
                library.append(tree.func.value.value.id)
            except:
                library.append(tree.func.value.id)
        except:
            print("Function not handled")
        return library, function_attributes