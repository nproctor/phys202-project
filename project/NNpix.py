from PIL import Image
import numpy as np

neuron1 = (Image.open('Neuron.png')).resize([500,300], Image.ANTIALIAS)

neuron_signal = (Image.open('NeuronBigEnough.png')).resize([900,250], Image.ANTIALIAS)

neuron_no_signal = (Image.open('NeuronSmallEnough.png')).resize([900,250], Image.ANTIALIAS)

circle_neuron = (Image.open('CircleNueron.jpg')).resize([600,250], Image.ANTIALIAS)

show_weight = (Image.open('SigmaNeuron.jpg')).resize([600,300], Image.ANTIALIAS)

show_weight_answer = (Image.open('SigmaAnswer.jpg')).resize([600,300], Image.ANTIALIAS)

derivative1 = (Image.open('derivation.jpg')).resize([600,200], Image.ANTIALIAS)


net = (Image.open('ComplicatedNeuralNet.jpg')).resize([600,300], Image.ANTIALIAS)

bias = (Image.open('Biases.jpg')).resize([600,300], Image.ANTIALIAS)

activation = (Image.open('ActivationFunctions.png')).resize([800,400], Image.ANTIALIAS)

morse1 = (Image.open('morse1.png')).resize([500,300], Image.ANTIALIAS)

derivation2 = (Image.open('derivation2.jpg')).resize([850,200], Image.ANTIALIAS)

cneuron1 = (Image.open('ComplicatedNeuralNet.jpg')).resize([500,400], Image.ANTIALIAS)

cneuron2 = (Image.open('ComplicatedNeuralNet2.jpg')).resize([600,400], Image.ANTIALIAS)
