import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.activations import elu, softplus, sigmoid
from tensorflow.keras.initializers import Ones, Zeros


from .jacobi_polynomials import *


class JacobiRKAN(Layer):
    def __init__(self, degree, **kwargs):
        """
        Initialize the JacobiRKAN layer.

        Args:
            degree (int): Degree of the Jacobi polynomial.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super(JacobiRKAN, self).__init__(**kwargs)
        self.degree = degree

        if not 0 < degree < 7:
            raise ValueError('Parameter degree must be between one and six (inclusive).')

        self.rational_jacobi_polynomial = tf.function(rational_jacobi_polynomial)
        

    def build(self, input_shape):
        """
        Create the weights of the layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer="ones",
            trainable=True
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(1,),
            initializer="ones",
            trainable=True
        )
        self.iota = self.add_weight(
            name="iota",
            shape=(1,),
            initializer="ones",
            trainable=True
        )
        super(JacobiRKAN, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass of the layer.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the Jacobi polynomial transformation.
        """
        normalized_alpha = elu(self.alpha, 1)
        normalized_beta = elu(self.beta, 1)
        normalized_iota = softplus(self.iota)

        return self.rational_jacobi_polynomial(
            inputs, self.degree, normalized_alpha, normalized_beta, 1, normalized_iota, backend=tf
        )
    def get_config(self):
        """
        Returns the layer configuration.

        This method is used to serialize the layer during saving and
        deserialization during loading.

        Returns:
            dict: Configuration dictionary of the layer.
        """
        config = super(JacobiRKAN, self).get_config()
        config.update({
            'degree': self.degree
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a layer from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary of the layer.

        Returns:
            PadeRKAN: Instantiated PadeRKAN layer.
        """
        config = config.copy()
        layer = cls(degree=config.pop('degree'), **config)
        return layer



class PadeRKAN(Layer):
    def __init__(self, degree_p, degree_q, **kwargs):
        """
        Initialize the PadeRKAN layer.

        Args:
            degree_p (int): Degree of the P polynomial.
            degree_q (int): Degree of the Q polynomial.
            **kwargs: Additional keyword arguments for the parent class.
        """
        super(PadeRKAN, self).__init__(**kwargs)

        if not 0 < degree_p < 7 or not 0 < degree_q < 7:
            raise ValueError('Both degree_p and degree_q must be between one and six (inclusive).')

        self.degree_p = degree_p
        self.degree_q = degree_q
        self.shifted_jacobi_polynomial = tf.function(shifted_jacobi_polynomial)

    def build(self, input_shape):
        """
        Create the weights of the layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.alpha_p = self.add_weight(
            name="alpha_p",
            shape=(1,),
            initializer=Ones(),
            trainable=True
        )
        self.beta_p = self.add_weight(
            name="beta_p",
            shape=(1,),
            initializer=Ones(),
            trainable=True
        )
        self.zeta_p = self.add_weight(
            name="zeta_p",
            shape=(1,),
            initializer=Zeros(),
            trainable=True
        )
        self.w_p = self.add_weight(
            name="w_p",
            shape=(self.degree_p,),
            initializer=Ones(),
            trainable=True
        )

        self.alpha_q = self.add_weight(
            name="alpha_q",
            shape=(1,),
            initializer=Ones(),
            trainable=True
        )
        self.beta_q = self.add_weight(
            name="beta_q",
            shape=(1,),
            initializer=Ones(),
            trainable=True
        )
        self.zeta_q = self.add_weight(
            name="zeta_q",
            shape=(1,),
            initializer=Zeros(),
            trainable=True
        )
        self.w_q = self.add_weight(
            name="w_q",
            shape=(self.degree_q,),
            initializer=Ones(),
            trainable=True
        )
        super(PadeRKAN, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass of the layer.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the Pade rational function.
        """
        normalized_alpha_p = elu(self.alpha_p, 1)
        normalized_beta_p = elu(self.beta_p, 1)
        normalized_zeta_p = sigmoid(self.zeta_p)

        normalized_alpha_q = elu(self.alpha_q, 1)
        normalized_beta_q = elu(self.beta_q, 1)
        normalized_zeta_q = sigmoid(self.zeta_q)

        normalized_inputs = sigmoid(inputs)

        p = self.w_p[0] + self.w_p[1] * normalized_inputs
        for deg in range(2, self.degree_p):
            p += self.w_p[deg] * self.shifted_jacobi_polynomial(
                normalized_inputs,
                deg,
                normalized_alpha_p,
                normalized_beta_p,
                normalized_zeta_p,
                0,
                1,
                backend=tf
            )

        q = self.w_q[0] + self.w_q[1] * normalized_inputs
        for deg in range(2, self.degree_q):
            q += self.w_q[deg] * self.shifted_jacobi_polynomial(
                normalized_inputs,
                deg,
                normalized_alpha_q,
                normalized_beta_q,
                normalized_zeta_q,
                0,
                1,
                backend=tf
            )

        return p / q

    def get_config(self):
        """
        Returns the layer configuration.

        This method is used to serialize the layer during saving and
        deserialization during loading.

        Returns:
            dict: Configuration dictionary of the layer.
        """
        config = super(PadeRKAN, self).get_config()
        config.update({
            'degree_p': self.degree_p,
            'degree_q': self.degree_q,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a layer from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary of the layer.

        Returns:
            PadeRKAN: Instantiated PadeRKAN layer.
        """
        config = config.copy()
        layer = cls(degree_p=config.pop('degree_p'), degree_q=config.pop('degree_q'), **config)
        return layer
