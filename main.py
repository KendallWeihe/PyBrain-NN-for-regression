# from __future__ import print_function
import csv
import itertools
import math
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer
from pybrain.datasets            import ClassificationDataSet, SequenceClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkWriter
from pybrain.tools.customxml import NetworkReader
from pybrain.datasets import SupervisedDataSet
from numpy.random import multivariate_normal

############################################################################
# PyBrain Tutorial "Networks, Modules, Connections"
#
# Author: Tom Schaul, tom@idsia.ch
############################################################################

__author__ = 'Tom Schaul, tom@idsia.ch'

from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork

""" This tutorial will attempt to guide you for using one of PyBrain's most basic structural elements:
Networks, and with them Modules and Connections.
Let us start with a simple example, building a multi-layer-perceptron (MLP).
First we make a new network object: """

min_error = 1000000.0
num_n = 0
min_vals = []
count = 0
error = 100000
while 1:
    n = FeedForwardNetwork()

    """ Next, we're constructing the input, hidden and output layers. """

    inLayer = LinearLayer(10)
    hiddenLayer = SigmoidLayer(30)
    hiddenLayer2 = SigmoidLayer(30)
    hiddenLayer3 = SigmoidLayer(30)
    outLayer = LinearLayer(1)

    """ (Note that we could also have used a hidden layer of type TanhLayer, LinearLayer, etc.)
    Let's add them to the network: """

    n.addInputModule(inLayer)
    n.addModule(hiddenLayer)
    n.addModule(hiddenLayer2)
    n.addModule(hiddenLayer3)
    n.addOutputModule(outLayer)

    """ We still need to explicitly determine how they should be connected. For this we use the most
    common connection type, which produces a full connectivity between two layers (or Modules, in general):
    the 'FullConnection'. """

    in2hidden = FullConnection(inLayer, hiddenLayer)
    hidden12 = FullConnection(hiddenLayer, hiddenLayer2)
    hidden23 = FullConnection(hiddenLayer2, hiddenLayer3)
    hidden2out = FullConnection(hiddenLayer3, outLayer)
    n.addConnection(in2hidden)
    n.addConnection(hidden12)
    n.addConnection(hidden23)
    n.addConnection(hidden2out)

    """ All the elements are in place now, so we can do the final step that makes our MLP usable,
    which is to call the 'sortModules()' method. """

    n.sortModules()

    # """ Let's see what we did. """
    #
    # print(n)


    with open('converted stats.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        elements = list(reader)

    temp_inputs = []
    inputs = []
    outputs = []
    counter = 0

    all_data = list(itertools.chain.from_iterable(elements))
    for item in all_data:
        if counter < 10:
            temp_inputs.append(float(item))
            counter = counter + 1
        else:
            inputs.append(temp_inputs)
            outputs.append(float(item))
            temp_inputs = []
            counter = 0

    with open('predictions set.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        elements = list(reader)

    temp_inputs = []
    prediction_inputs = []
    prediction_outputs = []
    counter = 0

    all_data = list(itertools.chain.from_iterable(elements))
    for item in all_data:
        if counter < 10:
            temp_inputs.append(float(item))
            counter = counter + 1
        else:
            prediction_inputs.append(temp_inputs)
            prediction_outputs.append(float(item))
            temp_inputs = []
            counter = 0


    # for x in outputs:
    #     x = x + 30

    # DS = ClassificationDataSet(20, 1, nb_classes=60)
    DS = SupervisedDataSet(10,1)
    for x, y in zip(inputs, outputs):
        DS.appendLinked(x,y)
        # DS.addSample(x,y)

    # DS._convertToOneOfMany(bounds=[0,1])
    error2 = 0.0
    local_min_error = 100000
    for x in range(0,35):

        if count == 16 and local_min_error > 5.4:
            break

        error2 = error

        fnn = buildNetwork( DS.indim, 30, DS.outdim, hiddenclass = SigmoidLayer, outclass=SoftmaxLayer )
        target = [16,14,9,16,-7,-2,16,-1,-6,8,7,10,-6,-2,12,2,3]

    # while 1:


        trainer = BackpropTrainer( n, DS, verbose=True)
        trainer.trainUntilConvergence(dataset = None, maxEpochs=750, continueEpochs=10, validationProportion=0.35)
        # trainer.trainEpochs(50)
        # trainer.trainOnDataset(DS, 1500)
        # trainer.testOnData(verbose = True)

        vals = []


        for x in prediction_inputs:
            vals.append(float(n.activate(x)))

        error = 0.0
        num = 0.0;
        for o, t in zip(vals, prediction_outputs):
            if abs(t - o) < 10:
                error += abs(t - o)
                num = num + 1

        error = error / num

        if error < local_min_error:
            local_min_error = error

        if error < min_error and num >= 16:
            NetworkWriter.writeToFile(n, "20 prediction games with num = 16.xml")
            min_error = error
            num_n = num
            min_vals = []
            for x in vals:
                x = float(x)
                min_vals.append(x)

        print("\n")
        for x in vals:
            print x
        print("\n")
        print(min_error)
        print(num_n)
        print("\n")
        for x in min_vals:
            print x
        print "\n"
        print count
        print "num = 16"
        print "-------------------------------------------"
        count = count + 1


# temp = 0
# indx = []
# for x in vals:
#     maxi = 0
#     index = 0
#     best_index = 0
#     for y in x:
#         temp = y
#         if temp > maxi:
#             best_index = index
#             maxi = temp
#         index = index + 1
#     indx.append(best_index)
#
# for x in indx:
#     print(x)
