## Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# short name
layers = tf.keras.layers
tfm = tf.math

## Optimizer
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

class PowerSign (optimizer.Optimizer):
    def __init__ (self,
                  learning_rate=0.1,
                  alpha=0.01,
                  beta=0.5,
                  use_locking=False,
                  name="PowerSign"):
        super (PowerSign, self).__init__ (use_locking, name)
        self.lr    = learning_rate
        self.alpha = alpha
        self.beta  = beta
        # tensor versions
        self._lr_t    = None
        self._alpha_t = None
        self._beta_t  = None
 
    def _prepare (self):
        self._lr_t    = ops.convert_to_tensor (self.lr,    name="learning_rate")
        self._alpha_t = ops.convert_to_tensor (self.alpha, name="alpha_t")
        self._beta_t  = ops.convert_to_tensor (self.beta,  name="beta_t")
 
    def _create_slots (self, var):
        for v in var:
            self._zeros_slot (v, "m", self._name)
 
    def _resource_apply_dense (self, grad, var):
        lr    = math_ops.cast (self._lr_t,    var.dtype.base_dtype)
        alpha = math_ops.cast (self._alpha_t, var.dtype.base_dtype)
        beta  = math_ops.cast (self._beta_t,  var.dtype.base_dtype)
        eps = 1e-7
        m = self.get_slot (var, "m")
        m_t = m.assign (tf.maximum (beta*m + eps, tfm.abs (grad)))
        var_update = state_ops.assign_sub (var,
                #lr*grad*tfm.exp (tfm.log (alpha)*tfm.sign (grad)*tfm.sign (m_t)))
                lr*grad*tfm.exp (tfm.sign (grad)*tfm.sign (m_t)))
        return control_flow_ops.group (*[var_update, m_t])
    
    def _resource_apply_sparse (self, grad, var):
        raise NotImplementedError ("Sparse gradients not supported.")

class AddSign (optimizer.Optimizer):
    def __init__ (self,
                  learning_rate=0.1,
                  alpha=0.01,
                  beta=0.5,
                  use_locking=False,
                  name="AddSign"):
        super (AddSign, self).__init__ (use_locking, name)
        self.lr    = learning_rate
        self.alpha = alpha
        self.beta  = beta
        # tensor versions
        self._lr_t    = None
        self._alpha_t = None
        self._beta_t  = None
 
    def _prepare (self):
        self._lr_t    = ops.convert_to_tensor (self.lr,    name="learning_rate")
        self._alpha_t = ops.convert_to_tensor (self.alpha, name="alpha_t")
        self._beta_t  = ops.convert_to_tensor (self.beta,  name="beta_t")
 
    def _create_slots (self, var):
        for v in var:
            self._zeros_slot (v, "m", self._name)

    def _resource_apply_dense (self, grad, var):
        lr    = math_ops.cast (self._lr_t,    var.dtype.base_dtype)
        alpha = math_ops.cast (self._alpha_t, var.dtype.base_dtype)
        beta  = math_ops.cast (self._beta_t,  var.dtype.base_dtype)
        eps = 1e-7
        m = self.get_slot (var, "m")
        m_t = m.assign (tf.maximum (beta*m + eps, tf.abs (grad)))
        var_update = state_ops.assign_sub (var,
                lr*grad*(1.0 + alpha*tf.sign (grad)*tf.sign (m_t)))
        return control_flow_ops.group (*[var_update, m_t])

    def _resource_apply_sparse (self, grad, var):
        raise NotImplementedError ("Sparse gradients not supported.")

class NDAdam (optimizer.Optimizer):
    def __init__ (self,
                  learning_rate=0.001,
                  beta1=0.9,
                  beta2=0.999,
                  epsilon=1e-8,
                  use_locking=False,
                  name="NDAdam"):
        print ("> Init")
        super (NDAdam, self).__init__ (use_locking, name)
        self.lr = learning_rate
        self.beta1 = ops.convert_to_tensor (beta1)
        self.beta2 = ops.convert_to_tensor (beta2)
        self.epsilon = epsilon
        self.m = None
        self.u = None
        self.t = tf.Variable (0, dtype=tf.int64, trainable=False)
        print ("< Init")

    def _create_slots (self, var):
        print ("> Create")
        v_shape = (len (var),)
        self.m = tf.Variable (tf.zeros (v_shape), trainable=False)
        self.u = tf.Variable (tf.zeros (v_shape), trainable=False)
        print ("< Create")

    def _resourse_apply_dense (self, grad, var):
        print ("> Apply")
        update_ops = []
        t = self.t.assign_add (1)
        update_ops.append (t)
        t = tf.to_float (t)

        for (g, v) in grad:
            g2 = tfm.square (g)
            m = self.m[v].assign (self.beta1*self.m[v] + (1 - self.beta1)*g)
            u = self.u[v].assign (self.beta2*self.u[v] + (1 - self.beta2)*g2)
            m_hat = m / (1 - tfm.pow (self.beta1, t))
            u_hat = u / (1 - tfm.pow (self.beta1, t))
            update_v = v - self.lr*m_hat/(tfm.sqrt (u_hat) + self.epsilon)
            update_v = v.assign (update_v)
            update_ops.append (update_v)
        print ("< Apply")
        return update_ops

## Parameters
num_epochs = 500
use_custom = False

## Plot settings
plt.rc ('text', usetex=True)
plt.rc ('font', family='serif')
#Does not work well
plot_filename = False#"ResultFlat.pgf"
data_filename = False#"DataFlat{}.csv"
data_stride = 1

tf.random.set_seed (123456)

## Load data
char_set = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = char_set.load_data ()

sample_size  = x_train.shape[0]
sample_shape = x_train[0].shape

## Normalize
x_train = x_train/255.0
x_test  = x_test /255.0

## Create model
model = tf.keras.models.Sequential (
[   layers.Flatten (input_shape=sample_shape)
,   layers.Dense (128, activation='relu')
,   layers.Dense (10, activation="softmax")
])

opti = NDAdam()

model.compile (optimizer=opti,#'adam',
               loss='sparse_categorical_crossentropy',
               metrics=[])

## Train
result = model.fit (x_train, y_train,
                    epochs=num_epochs,
                    batch_size=60000,
                    use_multiprocessing=True,
                    workers=12,
                    validation_data=(x_test, y_test))
print ("initial loss:", result.history['loss'][0])
print ("initial vali:", result.history['val_loss'][0])
print ("final loss:  ", result.history['loss'][-1])
print ("final loss:  ", result.history['val_loss'][-1])

## Plot
if data_filename:
    for att in [["loss","Loss"],["val_loss","Vali"]]:
        with open (data_filename.format (att[1]), "w") as f:
            prop = result.history[att[0]]
            for i in range (0, num_epochs, data_stride):
                f.write ("{0:d},{1:f}\n".format (i, prop[i]))
plt.plot (range (num_epochs), result.history['loss'], label='loss')
plt.plot (range (num_epochs), result.history['val_loss'], label='validation')
plt.legend ()
plt.show ()

pred_train = np.argmax (model.predict (x_train), axis=1)
pred_test  = np.argmax (model.predict (x_test),  axis=1)
diff_train = pred_train - y_train
diff_test  = pred_test  - y_test
errs_train = np.sum (np.absolute (diff_train))
errs_test  = np.sum (np.absolute (diff_test))
print ("training size:    ", x_train.shape[0])
print ("training errors:  ", errs_train)
print ("validation size:  ", x_test.shape[0])
print ("validation errors:", errs_test)

