# coding=UTF-8
"""
将 tensorflow 训练过程中生成的文件(checkpoint,model-7800.data-00000-of-00001,model-7800.index, model-7800.meta)转成pb文件
"""


import tensorflow as tf
import os.path

MODEL_DIR = "/home/zhwpeng/project/p0305/forecast_extraction/abcft_algorithm_forecast_extraction/text_classify/c_model/checkpoints"
MODEL_NAME = "model.pb"

checkpoint = tf.train.get_checkpoint_state(MODEL_DIR)  # 检查目录下ckpt文件状态是否可用
input_checkpoint = checkpoint.model_checkpoint_path  # 得ckpt文件路径
output_graph = os.path.join(MODEL_DIR, MODEL_NAME)  # PB模型保存路径

with tf.Session() as sess:
    # Restore the graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta')

    # Load weights
    ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
    if ckpt:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # Freeze the graph
    output_node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_node_names)

    # Save the frozen graph
    with open(output_graph, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())



