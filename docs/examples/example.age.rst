Password example
=================

This tutorial shows how to use privugger on measuring leakage on a simple password checker

**Imports**

.. code-block:: python
    :linenos:

    from privugger.attacker import simulate
    from privugger.transformer.type_decoration import load
    from typing import *



**Define the method**

IMPORTANT! Remember to add type annotation to your method using the tool typing

.. code-block:: python
    :linenos:

    def privacy_protection_mechanism(a: int) -> int:
        secret = 1024
        res = a == 1024
        return res

Since the program currently loads from a file, you can either manually place the function in the file program_to_be_analysed.py or use the code below

.. code-block:: python
    :linenos:

    import inspect
    with open("program_to_be_analysed.py", "w") as f:
        f.write("from typing import * \n \n \n")
        f.write(inspect.getsource(privacy_protection_mechanism))


**Lift the program to a probability monad**

By the use of the transformer plugin the program will be analysed and converted to a method that works on probability distributions

After this step a function called method(...) exist

.. code-block:: python
    :linenos:

    lifted_program = load("program_to_be_analysed.py", "privacy_protection_mechanism")
    exec(astor.to_source(lifted_program))


**Simulate attackers on the method**

This step is done to find the most vulnerable attackers

.. code-block:: python
    :linenos:
    
    results = simulate(method, max_examples=20, num_samples=1000, ranges=[(1000, 1050)])


**Plotting the results**

.. code-block:: python
    :linenos:

    results.plot_mutual_information()