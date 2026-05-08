import tensorflow as tf
from tensorflow.keras import layers, models, losses, metrics

class UNetModel:
    def __init__(self, input_shape, dropout_rate, loss, metrics_list, learning_rate=0.000003):
        self.input_shape   = input_shape
        self.dropout_rate  = dropout_rate
        self.loss          = loss
        self.metrics_list  = metrics_list
        self.learning_rate = learning_rate
        self.model         = self.build_model()

    def conv_block(self, input_tensor, num_filters):
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
        encoder = layers.BatchNormalization()(encoder)
        encoder = layers.Activation('relu')(encoder)
        return encoder

    def encoder_block(self, input_tensor, num_filters):
        encoder = self.conv_block(input_tensor, num_filters)
        encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
        return encoder_pool, encoder

    def decoder_block(self, input_tensor, concat_tensor, num_filters):
        decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
        decoder = layers.BatchNormalization()(decoder)
        decoder = layers.Activation('relu')(decoder)
        return decoder

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape)
        encoder0_pool, encoder0 = self.encoder_block(inputs, 64)
        encoder1_pool, encoder1 = self.encoder_block(encoder0_pool, 128)
        encoder2_pool, encoder2 = self.encoder_block(encoder1_pool, 256)
        encoder3_pool, encoder3 = self.encoder_block(encoder2_pool, 512)
        center = self.conv_block(encoder3_pool, 1024)
        decoder4 = self.decoder_block(center, encoder3, 512)
        decoder3 = self.decoder_block(decoder4, encoder2, 256)
        decoder2 = self.decoder_block(decoder3, encoder1, 128)
        decoder1 = self.decoder_block(decoder2, encoder0, 64)
        dropout = layers.Dropout(self.dropout_rate, name="dropout")(decoder1)
        outputs = layers.Conv2D(1, (1, 1), activation=tf.nn.sigmoid, padding='same', 
                                kernel_initializer=tf.keras.initializers.GlorotNormal())(dropout)

        model = models.Model(inputs=[inputs], outputs=[outputs])
        optimizer = tf.keras.optimizers.Nadam(self.learning_rate, name='optimizer')

        model.compile(
            optimizer=optimizer,
            loss=losses.get(self.loss),
            metrics=[metrics.get(metric) for metric in self.metrics_list],
            run_eagerly=True
        )
        return model

    def get_model(self):
        return self.model