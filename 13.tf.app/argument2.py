import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
def main(argv):
    print(FLAGS.open)
    print(FLAGS.learning_rate)
    print(FLAGS.filter)
    print(FLAGS.string)

if __name__ =="__main__":
    tf.app.flags.DEFINE_bool(flag_name="open", default_value=False, docstring="is open?")
    tf.app.flags.DEFINE_float(flag_name="learning_rate", default_value=0.001, docstring="learning rate")
    tf.app.flags.DEFINE_integer(flag_name="filter", default_value=128, docstring="filter numbers")
    tf.app.flags.DEFINE_string("string", default_value="yes", docstring="demo string")
    tf.app.run()