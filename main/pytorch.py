import numpy as np
import os
import pandas as pd
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

cwd = os.getcwd()
print('current dir: ', cwd)
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
print('root path: ', root_path)
sys.path.insert(0, root_path)

from dataset.breakfast.data_io import load_data
from main.config import get_global_parameters


class XWPlusBLayer(tf.keras.layers.Layer):
    def __init__(self, W, b, **kwargs):
        super(XWPlusBLayer, self).__init__(**kwargs)
        self.W = W
        self.b = b

    def call(self, inputs):
        return tf.nn.xw_plus_b(inputs, self.W, self.b)


class V2C(tf.keras.Model):
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, bias_init_vector=None):
        super(V2C, self).__init__()
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps

        self.Wemb = tf.Variable(tf.random.uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = tf.keras.layers.LSTMCell(dim_hidden)
        self.lstm2 = tf.keras.layers.LSTMCell(dim_hidden)

        self.encode_image_W = tf.Variable(tf.random.uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random.uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(
            bias_init_vector.astype(np.float32) if bias_init_vector is not None else tf.zeros([n_words]),
            name='embed_word_b')

    def call(self, video, video_mask, caption, caption_mask):
        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])

        state1 = self.lstm1.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        state2 = self.lstm2.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)

        padding = tf.zeros([self.batch_size, self.dim_hidden])
        probs = []
        loss = 0.0

        for i in range(self.n_lstm_steps):
            if i > 0:
                state1 = self.lstm1.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
                state2 = self.lstm2.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)

            output1, state1 = self.lstm1(image_emb[:, i, :], state1)
            output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(self.n_lstm_steps):
            if i == 0:
                current_embed = tf.zeros([self.batch_size, self.dim_hidden])
            else:
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i - 1])

            if i > 0:
                state1 = self.lstm1.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
                state2 = self.lstm2.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)

            output1, state1 = self.lstm1(padding, state1)
            output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            labels = tf.expand_dims(caption[:, i], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse.to_dense(tf.SparseTensor(indices=concated, values=tf.ones(tf.shape(concated)[0]),
                                                               dense_shape=[self.batch_size, self.n_words]),
                                               default_value=0.0)

            logit_words = XWPlusBLayer(self.embed_word_W, self.embed_word_b)(output2)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_words)
            cross_entropy = cross_entropy * caption_mask[:, i]
            probs.append(logit_words)
            current_loss = tf.reduce_sum(cross_entropy)
            loss += current_loss

        loss = loss / tf.reduce_sum(caption_mask)
        return loss, probs

    def build_generator(self):
        video = tf.keras.Input(shape=[1, self.n_lstm_steps, self.dim_image], dtype=tf.float32)
        video_mask = tf.keras.Input(shape=[1, self.n_lstm_steps], dtype=tf.float32)

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_lstm_steps, self.dim_hidden])

        state1 = self.lstm1.zero_state(1, tf.float32)
        state2 = self.lstm2.zero_state(1, tf.float32)
        padding = tf.zeros([1, self.dim_hidden])

        generated_words = []

        probs = []
        embeds = []

        for i in range(self.n_lstm_steps):
            if i > 0: tf.compat.v1.get_variable_scope().reuse_variables()

            with tf.compat.v1.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( image_emb[:,i,:], state1 )

            with tf.compat.v1.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat([padding,output1], 1), state2 )

        for i in range(self.n_lstm_steps):

            tf.compat.v1.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.dim_hidden])

            with tf.compat.v1.variable_scope("LSTM1"):
                output1, state1 = self.lstm1( padding, state1 )

            with tf.compat.v1.variable_scope("LSTM2"):
                output2, state2 = self.lstm2( tf.concat([current_embed,output1], 1), state2 )

            logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        print('Build generator done!')
        return video, video_mask, generated_words, probs, embeds

def preProBuildWordVocab(sentence_iterator, word_count_threshold=5):
    print(f'Preprocessing word counts and creating vocab based on word count threshold {word_count_threshold}')
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print(f'Filtered words from {len(word_counts)} to {len(vocab)}')

    ixtoword = {0: '.'}
    wordtoix = {'#START#': 0}
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)
    return wordtoix, ixtoword, bias_init_vector


def create_folder(output_dir, net_id, name_folder):
    created_path = os.path.join(output_dir, net_id, name_folder)
    if not os.path.exists(created_path):
        os.makedirs(created_path)
    return created_path


def train(net_id, train_file, dim_image, dim_hidden, n_frame_step, n_epochs, learning_rate, batch_size):
    output_dir = os.path.join(root_path, 'output')
    train_data = load_data(train_file)
    captions = train_data['caption'].values

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions, word_count_threshold=1)
    ixw_path = os.path.join(output_dir, 'ixtoword.npy')
    np.save(ixw_path, ixtoword)

    logs_tensor_path = create_folder(output_dir, net_id, 'tensorboard')
    pred_path = create_folder(output_dir, net_id, 'prediction')
    log_plot_path = create_folder(output_dir, net_id, 'plot_loss')
    model_path = create_folder(output_dir, net_id, 'log_model') + '/saved_model.weights.h5'

    model = V2C(dim_image=dim_image,
                n_words=len(wordtoix),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_lstm_steps=n_frame_step,
                bias_init_vector=bias_init_vector)

    model.build(
        input_shape=[(None, n_frame_step, dim_image), (None, n_frame_step), (None, n_frame_step), (None, n_frame_step)])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs_tensor_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(model_path, save_weights_only=True, save_best_only=True)

    loss_history = []
    for epoch in range(n_epochs):
        index = list(train_data.index)
        np.random.shuffle(index)
        train_data = train_data.iloc[index]
        current_train_data = train_data.groupby('video_path').apply(lambda x: x.sample(1)).reset_index(drop=True)

        list_current_epoch_loss = []
        for start, end in zip(range(0, len(current_train_data), batch_size),
                              range(batch_size, len(current_train_data), batch_size)):
            current_batch = current_train_data[start:end]
            current_videos = current_batch['video_path'].values

            current_feats = np.zeros((batch_size, n_frame_step, dim_image))
            current_feats_vals = list(map(lambda vid: np.load(vid), current_videos))
            current_video_masks = np.zeros((batch_size, n_frame_step))

            for ind, feat in enumerate(current_feats_vals):
                current_feats[ind][:len(feat)] = feat
                current_video_masks[ind][:len(feat)] = 1

            current_captions = current_batch['caption'].values
            current_caption_ind = list(map(
                lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions))
            current_caption_matrix = tf.keras.preprocessing.sequence.pad_sequences(sequences = current_caption_ind, padding='post',
                                                                                   maxlen=n_frame_step - 1)
            current_caption_matrix = np.hstack(
                [current_caption_matrix, np.zeros([len(current_caption_matrix), 1])]).astype(int)

            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix)))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            _, loss_val = model.train_on_batch(
                x={'video': current_feats, 'caption': current_caption_matrix},
                y=current_caption_matrix,
                sample_weight={'video_mask': current_video_masks, 'caption_mask': current_caption_masks}
            )

            list_current_epoch_loss.append(loss_val)
            print('epoch:', epoch, '-- loss:', loss_val)

        if epoch % 50 == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            model.save_weights(model_path)

        current_loss = np.mean(list_current_epoch_loss)
        print('current epoch loss: ', current_loss)
        loss_history.append(current_loss)

        plt.plot(loss_history, color='r')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper right')
        plot_file_name = 'plot_epoch_e' + str(epoch) + '.png'
        out_plot_path = os.path.join(log_plot_path, plot_file_name)
        plt.savefig(out_plot_path)

    print('Training complete!')

if __name__ == '__main__':

    # get global param
    dic_param = get_global_parameters()


    dim_hidden= dic_param.get('dim_hidden')
    n_frame_step = dic_param.get('n_frame_step')
    n_epochs = dic_param.get('n_epochs')
    learning_rate = dic_param.get('learning_rate')
    batch_size = dic_param.get('batch_size')


    cnn_name = 'resnet50_keras_feature_no_sub_mean'


    net_dic_param = dic_param.get(cnn_name)
    dim_image = net_dic_param.get('dim_image')
    train_path = net_dic_param.get('train_path')

    net_id = 'BasicLSTM_' + cnn_name + '_batchsize' + str(batch_size) + '_dimhidden' + str(dim_hidden) + '_learningrate' + str(learning_rate)


    print('START TRAINING ...')
    train(net_id, train_path, dim_image, dim_hidden, n_frame_step, n_epochs, learning_rate, batch_size)
    print('ALL DONE!')