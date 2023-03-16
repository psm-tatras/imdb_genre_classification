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
        mask = layer_out["input_mask"]
        layer_out = self.transformer_encoder(layer_out)
        pooled_output = layer_out["pooled_output"]      # [batch_size, dim].
        sequence_output = layer_out["sequence_output"]  # [batch_size, seq_length, dim].
        return pooled_output,sequence_output,mask


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongAttention, self).__init__()
        self.Wa = tf.keras.layers.Dense(units)

    def call(self, encoder_h, decoder_s,mask):
        # encoder_h should be N x T x H
        # decoder_s should be N x H
        state = self.Wa(decoder_s)  # N x H
        state = tf.expand_dims(state, axis=1)  # N x 1 x H
        score = tf.linalg.matmul(state, encoder_h, transpose_b=True)  # N x 1 x T
        score = tf.transpose(score, [0, 2, 1])  # N x T x 1
        # alpha = tf.nn.softmax(score, axis=-1)  # N x T x 1 , score normalized
        alpha = AttentionMasking(tf.squeeze(score,-1),mask) # normalization in presence of masking
        ct = tf.multiply(encoder_h, alpha)  # N x T x dim
        ct_sum = tf.reduce_sum(ct, axis=1)  # N x dim
        return ct_sum, alpha


def process_sequence(x, i, g):
    nz = tf.where(x[i])  # find the non zero elements
    pad_value = x[i].shape[0] - nz.shape[0]
    temp = tf.squeeze(tf.gather(x[i], nz),-1)  # pack the non-zero elements in another tensor
    temp = tf.expand_dims(tf.nn.softmax(temp),0)
    temp = tf.pad(temp, [[0,0], [0,pad_value]])
    temp = tf.squeeze(temp, 0)
    g.write(i, temp).mark_used()
    # print(temp.shape)
    return x, i + 1, g


def stop_condition(x, i, g):
    # print(i)
    return tf.less(i, x.shape[0])


def AttentionMasking(alpha,mask):
    i = tf.constant(0)
    stack = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True, clear_after_read=False)
    mask = tf.cast(mask,tf.float32)
    alpha_m = tf.multiply(alpha,mask)
    _, _, stacked_vectors = tf.while_loop(stop_condition, process_sequence, [alpha_m, i, stack])
    new_alpha = stacked_vectors.gather(range(len(alpha)))
    new_alpha = tf.expand_dims(new_alpha,-1)
    return new_alpha


class AttentionClassifier():
    def __init__(self, nb_classes,save_path="model/"):
        self.encoder_layer = Encoder()
        self.attention_layer = LuongAttention(attention_dim)
        self.classifier = tf.keras.layers.Dense(nb_classes, activation="softmax")
        self.optimizer = tf.keras.optimizers.Adam(0.00001)
        self.checkpoint = tf.train.Checkpoint(enc=self.encoder_layer, attn= self.attention_layer, cls=self.classifier)
        self.weight_manager = tf.train.CheckpointManager(self.checkpoint,save_path,max_to_keep=1)

    def predict(self, text):
        encoding_state, encoding_h,input_mask = self.encoder_layer(text)
        batch_size, dim = encoding_state.shape[0], encoding_state.shape[1]
        decoder_s = tf.constant(0.1, shape=[batch_size, attention_dim])
        context_vector, attention_score = self.attention_layer(encoding_h, decoder_s,input_mask)
        prob = self.classifier(context_vector)  # N x nb_classes
        return prob,attention_score

    def evaluate_step(self,texts,labels):
        prob_vector,a_score = self.predict(texts)
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, prob_vector))
        preds = tf.argmax(prob_vector, axis=-1)
        truths = tf.argmax(labels, axis=-1)
        acc = tf.reduce_mean(tf.cast(tf.equal(truths, preds), tf.float32))
        return loss,acc

    def train_step(self, text_batch, label_batch):
        with tf.GradientTape() as tape:
            l,a = self.evaluate_step(text_batch,label_batch)
            train_weights = self.encoder_layer.trainable_variables + self.attention_layer.trainable_variables + self.classifier.trainable_variables
            gradients = tape.gradient(l, train_weights)
        self.optimizer.apply_gradients(zip(gradients, train_weights))
        # measure accuracy
        return l,a

    def fit(self, x_train, y_train,label2ind, batch_size=64, epochs=5,validation_data=None):
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
            if validation_data is not None:
                val_x,val_y = validation_data[0],validation_data[1]
                val_loss,val_acc = self.evaluate(val_x,val_y,batch_size,label2ind)
                print("Validation after Epoch %d:[Loss %0.3f, Accuracy %0.2f]"%(e+1,val_loss,val_acc))

    def load_model(self,model_path):
        latest_model = tf.train.latest_checkpoint(model_path)
        self.checkpoint.restore(latest_model)
        print("Model restored weights from %s"%latest_model)

    def evaluate(self,val_text,val_label,batch_size,label2ind):
        batch_loss = []
        batch_acc = []
        total = len(val_text)
        nb_batches = int(np.ceil(total / batch_size))
        start = 0
        for b in range(nb_batches):
            end = min(total, start + batch_size)
            bX, bY = load_batch_data(val_text, val_label, start, end, label_map=label2ind)
            b_loss, b_acc = self.evaluate_step(bX, bY)
            batch_loss.append(b_loss)
            batch_acc.append(b_acc)
        avg_loss = sum(batch_loss) / len(batch_loss)
        avg_accuracy = sum(batch_acc) / len(batch_acc)
        return avg_loss,avg_accuracy
