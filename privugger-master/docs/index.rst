.. privugger documentation master file, created by
   sphinx-quickstart on Tue Apr 13 14:22:25 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to privugger's documentation!
=====================================

Privugger (/prɪvʌɡə(r)/) is a privacy risk analysis library for python programs.  Privugger, takes as input a python program and a specification of the adversary's knowledge about the input of the program (the *prior knowledge*), and it returns a wide variety of privacy risk analyses, including the following leakage measures:

* Knowledge-based probability queries
* Entropy
* Mutual Information
* KL-divergence
* min-entropy
* Bayes risk
* ...

Furthermore, Privugger is equipped with a module to perform *automatic attacker synthesis*. That is, given a program and a leakage measure, it finds the adversary's prior knowledge that maximizes the leakage. In other words, it tells us what is the minimum amount of information that the adversary must know in order for the program to exhibit privacy risks. If this knowledge is publicly available, then the program does not effectively protect users' privacy.

.. toctree::
   :maxdepth: 1
   :caption: Privugger Tutorials & Examples

   tutorials/Tutorial.ipynb
   tutorials/Open-dp-Tutorial.ipynb
   tutorials/Governor.ipynb
   tutorials/Duplicate.ipynb


.. toctree::
   :maxdepth: 1
   :caption: API and Developer Reference
   :hidden:
	     
   privugger.transformer
   privugger.datastructures
   privugger.distributions
   privugger.inference
   privugger.measures
   privugger.attacker
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
