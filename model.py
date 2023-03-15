import numpy as np
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
from config import *
from utils import load_batch_data
import sys


class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder,self).__init__()
        self.preprocess_layer = hub.KerasLayer(hub_preprocessor_model, name='preprocessing')
        self.transformer_encoder = hub.KerasLayer(hub_transformer_model)

    def call(self,input):
        layer_out = self.preprocess_layer(input)
        layer_out = self.transformer_encoder(layer_out)
        pooled_output = layer_out["pooled_output"]      # [batch_size, dim].
        sequence_output = layer_out["sequence_output"]  # [batch_size, seq_length, dim].
        return pooled_output,sequence_output


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.Wa = tf.keras.layers.Dense(units)

    def call(self, encoder_h, decoder_s):
        # encoder_h should be N x T x H
        # decoder_s should be N x H
        state = self.Wa(decoder_s)  # N x H
        state = tf.expand_dims(state, axis=1)  # N x 1 x H
        score = tf.linalg.matmul(state, encoder_h, transpose_b=True)  # N x 1 x T
        score = tf.transpose(score, [0, 2, 1])  # N x T x 1
        alpha = tf.nn.softmax(score, axis=-1)  # N x T x 1 , score normalized
        ct = tf.multiply(encoder_h, alpha)  # N x T x dim
        ct_sum = tf.reduce_sum(ct, axis=1)  # N x dim
        return ct_sum, alpha


class AttentionClassifier():
    def __init__(self, nb_classes,save_path="model/"):
        self.encoder_layer = Encoder()
        self.attention_layer = LuongAttention(attention_dim)
        self.classifier = tf.keras.layers.Dense(nb_classes, activation="softmax")
        self.optimizer = tf.keras.optimizers.Adam(0.00001)
        self.checkpoint = tf.train.Checkpoint(enc=self.encoder_layer, attn= self.attention_layer, cls=self.classifier)
        self.weight_manager = tf.train.CheckpointManager(self.checkpoint,save_path,max_to_keep=1)

    def predict(self, text):
        encoding_state, encoding_h = self.encoder_layer(text)
        batch_size, dim = encoding_state.shape[0], encoding_state.shape[1]
        decoder_s = tf.constant(0.1, shape=[batch_size, attention_dim])
        context_vector, attention_score = self.attention_layer(encoding_h, decoder_s)
        prob = self.classifier(context_vector)  # N x nb_classes
        return prob

    def train_step(self, text_batch, label_batch):
        with tf.GradientTape() as tape:
            prob_vector = self.predict(text_batch)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(label_batch, prob_vector))
            train_weights = self.encoder_layer.trainable_variables + self.attention_layer.trainable_variables + self.classifier.trainable_variables
            gradients = tape.gradient(loss, train_weights)
        self.optimizer.apply_gradients(zip(gradients, train_weights))
        # measure accuracy
        preds = tf.argmax(prob_vector,axis=-1)
        truths = tf.argmax(label_batch,axis=-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(truths,preds),tf.float32))
        return loss,acc

    def fit(self, x_train, y_train,label2ind, batch_size=64, epochs=5):
        loss_history = []
        acc_history = []
        total = len(x_train)
        nb_batches = int(np.ceil(total/batch_size))
        for e in range(epochs):
            start = 0
            batch_loss = []
            batch_acc = []
            for b in range(nb_batches):
                end = min(total, start + batch_size)
                bX, bY = load_batch_data(x_train, y_train, start, end,label_map=label2ind)
                b_loss,b_acc = self.train_step(bX,bY)
                batch_loss.append(b_loss)
                batch_acc.append(b_acc)
                sys.stdout.write("\rReading from %d-%d - Batch Loss %0.3f, Accuracy %0.3f" % (start, end,b_loss,b_acc))
                sys.stdout.flush()
                start = end
                if start >= total:
                    break
            avg_loss = sum(batch_loss) / len(batch_loss)
            avg_accuracy = sum(batch_acc) / len(batch_acc)
            loss_history.append(avg_loss)
            acc_history.append(avg_accuracy)
            self.weight_manager.save()
            print("\tEpoch %d/%d Loss %.3f, Accuracy %0.3f"%(e+1,epochs,avg_loss,avg_accuracy))

    def load_model(self,model_path):
        latest_model = tf.train.latest_checkpoint(model_path)
        self.checkpoint.restore(latest_model)
        print("Model restored weights from %s"%latest_model)
