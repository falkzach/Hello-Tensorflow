import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


path = "zeroscope.jpg"
raw_image_data = mpimg.imread(path)

image = tf.placeholder("uint8", [None, None, 3])

transpose_op = lambda perm: tf.transpose(image, perm=perm) # rotate 90 to the left
reverse_op = lambda vector: tf.reverse(image, vector) # flip vertically
slice_op = lambda start, size: tf.slice(image, start, size) # slice out a portion

feed = {image: raw_image_data}

init_op = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_op)
    transpose_left_result = session.run(transpose_op([1, 0, 2]), feed_dict=feed)
    vertical_flip_result = session.run(reverse_op([0]), feed_dict=feed)
    transpose_right_result = session.run(transpose_op([1, 0, 2]), feed_dict={image: vertical_flip_result})
    left_slice_result = session.run(slice_op([100, 0, 0], [250, 250, 3]), feed_dict=feed)
    right_slice_result = session.run(slice_op([200, 350, 0], [250, 250, 3]), feed_dict=feed)
    center_slice_result = session.run(slice_op([50, 300, 0], [250, 250, 3]), feed_dict=feed)

fig, axarr = plt.subplots(3, 3)

for i in range(3):
    for j in range(3):
        axarr[i, j].set_xticks([])
        axarr[i, j].set_yticks([])

axarr[0, 1].imshow(center_slice_result)

axarr[1, 0].imshow(left_slice_result)
axarr[1, 1].imshow(raw_image_data)
axarr[1, 2].imshow(right_slice_result)

axarr[2, 0].imshow(transpose_left_result)
axarr[2, 1].imshow(vertical_flip_result)
axarr[2, 2].imshow(transpose_right_result)

plt.savefig("./image_translations.png")
plt.show()
