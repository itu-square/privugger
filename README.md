# Privugger: Data Privacy Debugger

Docs and tutorials: https://itu-square.github.io/privugger/

Privugger (/prɪvʌɡə(r)/) is a privacy risk analysis library for python
programs.  Privugger, takes as input a python program and a
specification of the adversary's knowledge about the input of the
program (the _prior knowledge_), and it returns a wide variety of
privacy risk analyses, including the following leakage measures:

* Knowledge-based probability queries
* Entropy
* Mutual Information
* KL-divergence
* min-entropy
* Bayes risk
* ...

Furthermore, Privugger is equipped with a module to perform _automatic
attacker synthesis_. That is, given a program and a leakage measure,
it finds the adversary's prior knowledge that maximizes the
leakage. In other words, it tells us what is the minimum amount of
information that the adversary must know in order for the program to
exhibit privacy risks. If this knowledge is publicly available, then
the program does not effectively protect users' privacy.



## Installation 

Privugger is a tool written entirely in Python and can be installed using the pip packet manager.

To install write following in command line: 

`pip install privugger`

Usage:

`import privugger as pv`

`x = pv.Normal(...)`

`ds = pv.Dataset(...)`

 `program = pv.Program(...)`

`trace = pv.infer(...)`

