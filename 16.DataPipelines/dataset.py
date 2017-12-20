import tensorflow as tf
import numpy as np

dataset1=tf.data.Dataset.from_tensors(np.array([1.,2.,3.,4.,5.]))

dataset2=tf.data.Dataset.from_tensor_slices(tensors=np.array([1.,2.,3.,4.,5.]))

print("element shape of dataset1:",dataset1.output_shapes)
print("element shape of dataset2:",dataset2.output_shapes)
print("element type of dataset2:",dataset2.output_types)

iterator=dataset1.make_one_shot_iterator()
iterator2=dataset2.make_one_shot_iterator()

element=iterator.get_next()
element2=iterator2.get_next()
with tf.Session() as sess:
    print("elements of dataset1:")
    print(sess.run(element))

    print("elements of dataset2:")
    for i in range(5):
        print(sess.run(element2))

