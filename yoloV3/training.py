
import tensorflow as tf
import csv
import numpy as np
from demo import load_coco_names, FLAGS
from yolo_v3 import yolo_v3, load_weights, detections_boxes, non_max_suppression
from PIL import Image

def batch_generator(X, y, batch_size=100, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i + batch_size], y[i:i + batch_size])

def calculate_loss(X, y):
    classes = load_coco_names(FLAGS.class_names)



    with tf.variable_scope('detector'):
        # detections = yolo_v3(X, len(classes), data_format='NCHW')
        detections = yolo_v3(X, len(classes), data_format='NHWC')

    detected_boxes = detections_boxes(detections)
    filtered_boxes = non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)


    ##TODO: change ground truth-output
    ##TODO: calculate loss



def train(sess, X_data, y_data, validation_set=None, initialize=True, epochs=20, shuffle=True,
          dropout=0.5, random_seed=None):

    training_loss = []
    # placeholder for detector inputs
    X = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], name='X')
    y = tf.placeholder(tf.float32, [None, 1, 5], name='y')

    # Loss and optimizer
    cost = calculate_loss(X,y)
    ##TODO: change optimizer
    ##TODO:add learning_rate=learning_rate
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.5).minimize(3)



    if initialize:
        sess.run(tf.global_variables_initializer())
        with tf.variable_scope('detector'):
            load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)
            sess.run(load_ops)

    np.random.seed(random_seed)  # for shuffling in batch_generator
    for epoch in range(1, epochs + 1):
        batch_gen = batch_generator(X_data, y_data, shuffle=shuffle)
        avg_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'X': batch_x,
                    'y': batch_y
                    }
            _,loss = sess.run(optimizer, cost, feed_dict=feed)
            avg_loss += loss
            print('Batch Loss : %7.3f' % loss, end=' ')

        training_loss.append(avg_loss / (batch_idx + 1))
        print('Epoch %02d Training Avg. Loss: %7.3f' % (epoch, avg_loss), end=' ')



def training_yolo():
    X = []
    y = []
    with open('./preprocessing/train_ship_segmentations_bbox.csv', newline='') as segments_csv:
        reader = csv.DictReader(segments_csv)
        for row in reader:
            img = Image.open('../../Ship_Detection/train/'+row['ImageId'])
            img_resized = img.resize(size=(FLAGS.size, FLAGS.size))
            X.append(np.array(img_resized, dtype=np.float32))
            y.append(row['BBox'])

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

    with tf.Session() as sess:
        train(sess, np.array(x_train), np.array(y_train), initialize=True, random_seed=123)



    ##TODO:freeze weights


training_yolo()