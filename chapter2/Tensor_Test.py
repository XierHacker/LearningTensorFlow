#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 15:29:40 2016

@author: xierhacker
"""
from __future__ import print_function,division
import tensorflow as tf

#build a graph
print("build a graph")
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[1,1],[0,1]])
print("a:",a)
print("b:",b)
print("type of a:",type(a))
c=tf.matmul(a,b)
print("c:",c)
print("\n")
#construct a 'Session' to excute the graph
sess=tf.Session()

## Execute the graph and store the value that `c` represents in `result`.
print("excuted in Session")
result_a=sess.run(a)
result_a2=a.eval(session=sess)
print("result_a:\n",result_a)
print("result_a2:\n",result_a2)

result_b=sess.run(b)
print("result_b:\n",result_b)

result_c=sess.run(c)
print("result_c:\n",result_c)
print("type of c:",type(result_c))


#attribute test
print("element type of a",a.dtype)
print("the name of tensor",a.name)
print("this tensor belongs to:",a.graph)
print("operator to produce this tensor:",a.op)

#fuction test
print("Operations that consume this tensor:",a.consumers())