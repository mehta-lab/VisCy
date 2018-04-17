"""Custom callbacks"""
from keras.callbacks import TensorBoard

class TensorBoardWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback

    https://github.com/keras-team/keras/issues/3358
    """

    def __init__(self, val_gen, **kwargs):
        """Add val_generator to init"""

        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.val_gen = val_gen
        self.num_val_batches = int(len(self.val_gen.num_samples) / batch_size)

    def on_epoch_end(self, epoch, logs=None):
        """Overwrite this class to use val_gen"""
        logs = logs or {}

        if self.val_gen and self.histogram_freq:
            if epoch % self.histogram_freq == 0:
                val_gen = self.val_gen
                for batch_idx in range(self.num_val_batches):
                    val_input, val_target = val_gen.__next__()
                    tensors = (self.model.inputs +
                               self.model.targets +
                               self.model.sample_weights)

                    if self.model.uses_learning_phase:
                        tensors += [K.learning_phase()]

                    assert len(val_input) == len(tensors)
                    val_size = val_input[0].shape[0]
                    i = 0
                    #while i < val_size:
                    #step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]
                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_ckpt_path:
            if epoch % self.embeddings_freq == 0:
                self.saver.save(self.sess,
                                self.embeddings_ckpt_path,
                                epoch)


