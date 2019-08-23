import numpy as np
import tensorflow as tf
import model
import parameter
import process
import time

def train():
    '''
    :param file_list:
    :return:
    '''
    encoder = model.Encoder(
        vocab_size=parameter.SRC_VOCAB_SIZE,
        embedding_dim=parameter.EMBEDDINGS_DIM,
        enc_units=parameter.HIDDEN_UNITS,
        batch_sz=parameter.BATCH_SIZE
    )

    decoder=model.Decoder(
        vocab_size=parameter.TARGET_VOCAB_SIZE,
        embedding_dim=parameter.EMBEDDINGS_DIM,
        dec_units=parameter.HIDDEN_UNITS,
        batch_sz=parameter.BATCH_SIZE
    )

    optimizer = tf.keras.optimizers.Adam(parameter.LEARNING_RATE)
    cce=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = cce(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    #log
    file_writer=tf.summary.create_file_writer(logdir=parameter.LOG_DIR)

    # data set API
    input_tensor, target_tensor, inp_lang, targ_lang = process.load_dataset(
        path=parameter.CORPUS_PATH,
        num_examples=None
    )
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(parameter.DATASET_SIZE)
    dataset = dataset.batch(parameter.BATCH_SIZE, drop_remainder=True)


    def train_step(inp,targ,init_enc_hidden):
        loss=0.0
        with tf.GradientTape() as tape:
            #encoder
            enc_output, enc_hidden = encoder(inp, init_enc_hidden)

            #decoder
            dec_hidden=enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * parameter.BATCH_SIZE, 1)
            #print("dec_input:\n",dec_input)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                print("predictions:\n",predictions)

                loss += loss_function(targ[:, t], predictions)
                # using teacher forcing
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return batch_loss


    for epoch in range(parameter.MAX_EPOCH):
        total_loss=0
        init_enc_hidden=encoder.initialize_hidden_state()

        for (batch, (inp, targ)) in enumerate(dataset.take(parameter.STEPS_PER_EPOCH)):
            batch_loss = train_step(inp, targ, init_enc_hidden)
            total_loss += batch_loss
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,batch,batch_loss.numpy()))
            # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=parameter.CHECKPOINT_PATH)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,total_loss / parameter.STEPS_PER_EPOCH))


if __name__=="__main__":
    train()






