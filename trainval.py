import numpy as np
import tensorflow as tf
import datetime

from VGGNet.vgg16 import vgg16
# from GoogleNet.googlenet import googlenet
# from ResNet.resnet import resnet

from config.config import cfg
from utils.preprocessor import BatchPreprocessor



def train():
    num_epochs = cfg.NUM_EPOCHS
    learning_rate = cfg.LEARNING_RATE
    num_classes = cfg.NUM_CLASSES
    training_file = cfg.TRAINING_FILE
    val_file = cfg.CAL_FILE
    train_layers = cfg.TRAIN_LAYERS.split(',')
    kp = cfg.KEEP_PROB
    vgg16_ckpt_path = cfg.CKPT_PATH
    batch_size = cfg.BATCH_SIZE

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    model = vgg16(vgg16_ckpt_path, keep_prob=kp)

    loss = model.loss(x, y)
    optimizer = model.optimize(learning_rate, train_layers)

    saver = tf.train.Saver()

    train_preprocessor = BatchPreprocessor(dataset_file_path=training_file, num_classes=num_classes,
                                        output_size=[227, 227], horizontal_flip=True, shuffle=True)
    val_preprocessor = BatchPreprocessor(dataset_file_path=val_file, num_classes=num_classes, output_size=[227, 227])

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / batch_size).astype(np.int16)

    init = tf.global_variables_initializer()
    with tf.device('/GPU:0'):
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                print('{} Epoch number: {}'.format(datetime.datetime.now(), epoch+1))
                step = 1
                avg_cost = 0.
                # Start training
                while step < train_batches_per_epoch:
                    batch_xs, batch_ys = train_preprocessor.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: kp})
                    avg_cost += sess.run(loss, feed_dict={x:batch_xs, y:batch_ys, keep_prob: 1.0}) / train_batches_per_epoch

                    # Logging
                    # if step % FLAGS.log_step == 0:
                    #     s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.})
                    step += 1
                print('{} training cost = {:.4f}'.format(epoch+1, avg_cost))
                
                # Epoch completed, start validation
                print('{} Start validation'.format(datetime.datetime.now()))
                test_acc = 0.
                test_count = 0
                for _ in range(val_batches_per_epoch):
                    batch_xs, batch_ys = val_preprocessor.next_batch(batch_size)
                    acc = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
                    test_acc += acc
                    test_count += 1

                test_acc /= test_count
                # s = tf.Summary(value=[tf.Summary.Value(tag='validation_accuracy', simple_value=test_acc)])
                # val_writer.add_summary(s, epoch+1)
                print('{} Validation Accuracy = {:.4f}'.format(datetime.datetime.now(), test_acc))

                # Reset the dataset pointers
                val_preprocessor.reset_pointer()
                train_preprocessor.reset_pointer()

                # if (epoch+1) % 100 == 0:
                print('{} Saving checkpoint of model...'.format(datetime.datetime.now()))

                # save checkpoint of model
                # checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
                # save_path = saver.save(sess, checkpoint_path)

                print('{} Model checkpoint saved at {}'.format(datetime.datetime.now(), checkpoint_path))
    pass

if __name__ == '__main__':
    train()