import tensorflow as tf
import keyboard
from Utility import Utility
from pathlib import Path
from CapsuleMemory import CapsuleMemory

class NeuralNet:

    def __init__(self, inputMapping : dict, outputMapping : dict, neuralNetName : str, swapInputOutput : bool):
        self._name          : str   = neuralNetName
        self._inputMapping  : dict  = inputMapping                      # Attribute - Index
        self._outputMapping : dict  = outputMapping                     # Attribute - Index
        self._numInputs     : int   = max(inputMapping.values())  + 1
        self._numOutputs    : int   = max(outputMapping.values()) + 1
        self._swapInOut     : bool  = swapInputOutput

        # Tensorflow Tensors and Co.
        self._nnX                   = None
        self._nnY                   = None
        self._nnDropOutRate         = None
        self._nnFullTensor          = None

        self.loadFromFile()

    
    def hasTraining(self):
        fpath = Path("Models/" + self._name + ".ckpt.meta")
        if fpath.is_file():
            return True
        else:
            return False


    def loadFromFile(self):
        if self.hasTraining():
            return 1


    def createGraph(self):     
        # TODO: Infer number   
        numHidden1 = 256  # 256 
        numHidden2 = 128  # 128

        self._nnX           = tf.placeholder("float", [None, self._numInputs], name="X")
        self._nnY           = tf.placeholder("float", [None, self._numOutputs], name="Y")
        self._nnDropOutRate = tf.placeholder(tf.float32)

        weights = {
            'h1': tf.Variable(tf.truncated_normal([self._numInputs, numHidden1])),
            'h2': tf.Variable(tf.truncated_normal([numHidden1, numHidden2])),
            'out': tf.Variable(tf.truncated_normal([numHidden2, self._numOutputs]))
        } 
        biases = {
            'b1': tf.Variable(tf.random_uniform([numHidden1])),
            'b2': tf.Variable(tf.random_uniform([numHidden2])),
            'out': tf.Variable(tf.random_uniform([self._numOutputs]))
        }

        tf.summary.histogram("h1", weights['h1'])
        tf.summary.histogram("h2", weights['h2'])
        tf.summary.histogram("outh", weights['out'])
        
        tf.summary.histogram("b1", biases['b1'])
        tf.summary.histogram("b2", biases['b2'])
        tf.summary.histogram("outb", biases['out'])

        layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(self._nnX, weights['h1']), biases['b1'])), self._nnDropOutRate)
        layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])), self._nnDropOutRate)
        self._nnFullTensor = tf.matmul(layer2, weights['out']) + biases['out']


    def trainFromData(self, trainingData : CapsuleMemory, showDebugOutput : bool = False):

        tf.reset_default_graph()

        learningRate = 0.001 # 0.001
        batchSize =  64 # 64
        numSteps = 6000 #60000 # 60000
        
        self.createGraph()

        loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=self._nnFullTensor, labels=self._nnY, weights=100.0))
        train = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        tf.summary.histogram("loss", loss)
        count = 0

        with tf.Session() as sess:

            trainWriter = tf.summary.FileWriter( './logs/train', sess.graph)
            sess.run(init)
            
            if self.hasTraining():
                saver.restore(sess, "Models/" + self._name + ".ckpt") 

            merge = tf.summary.merge_all()

            for step in range(0, numSteps+1):        

                if self._swapInOut is True:
                    batchY, batchX = trainingData.nextBatch(batchSize, self._outputMapping, self._inputMapping)
                else:
                    batchX, batchY = trainingData.nextBatch(batchSize, self._inputMapping, self._outputMapping)

                sess.run(train, feed_dict={self._nnX: batchX, self._nnY: batchY, self._nnDropOutRate: 0.7})

                if keyboard.is_pressed('#'):
                    print('Exiting training early...')
                    break
                    
                if showDebugOutput is True and step % (int(numSteps / 500)) == 0:
                    count += 1
                    summary, currloss = sess.run([merge, loss], feed_dict={self._nnX: batchX, self._nnY: batchY, self._nnDropOutRate: 1.0})
                    print(str(100 * step / numSteps) + "% done")
                    print("Current Loss = " + "{:.4f}".format(currloss))

                    trainWriter.add_summary(summary, count)

                    if keyboard.is_pressed('#'):
                        print('Exiting training early...')
                        break

            saver.save(sess, "Models/" + self._name + ".ckpt")




    def forwardPass(self, inputs : dict):
        # inputs        # Attribute  -  Value

        if self.hasTraining() == False:
            print("Can't perform forward pass, as Neural Net has not been trained")
            return {}

        tf.reset_default_graph()

        self.createGraph()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, "Models/" + self._name + ".ckpt") 

            results = sess.run(self._nnFullTensor , feed_dict ={self._nnX : [Utility.mapDataOneWayDictRev(inputs, self._inputMapping)], self._nnDropOutRate : 1.0}) 
            return Utility.mapDataOneWayRev(results[0], self._outputMapping)        # Attribute - Value