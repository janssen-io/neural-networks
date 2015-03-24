import random

class MultilayerNetwork:
    def __init__(self, features, activation_functions, activation_derivatives, net_weights=None, alpha=0.1):
        '''Creates a neural network using a list of strings as input '''
        self.alpha = alpha

        # initialise an empty network
        self.network = []

        # create layers
        self.init_layers(features, activation_functions, activation_derivatives, net_weights)

    def init_layers(self, features, activation_functions, activation_derivatives, net_weights):
        # for all lists of activation functions
        for i, functions in enumerate(activation_functions):

            # create a new layer
            layer = []

            # for all activation functions in one layer
            for j, function in enumerate(functions):

                # get the derivative
                derivative = activation_derivatives[i][j]

                # use given weights if available
                if net_weights:
                    weights = net_weights[i][j]

                # else define them for the right layer
                elif i == 0:
                    weights = {f: random.randrange(-5, 5, 1)/10 for f in features}
                else:
                    weights = {n.name: random.randrange(-5, 5, 1)/10 for n in self.network[i-1]}

                # Create a Node and append it to the current layer
                layer.append(Node(
                    name=j,
                    weights=weights,
                    fn=function,
                    deriv=derivative,
                ))
            # append the layer to the network
            self.network.append(layer)

            # if its not the input layer
            if i > 0:

                # adjust references foreach node in the previous layer
                for node in self.network[i-1]:
                    node.next = [l for l in layer]

    def train(self,
         targets: "List of target outputs equal to the size of output nodes",
         inputs: "List of input values equal to the size of input nodes"):
        '''Trains the neural network using a single test case'''

        # classify the input
        guesses = self.classify(inputs)

        # calculate the error term
        errors = [target - guess for target, guess in zip(targets, guesses)]

        # propogate the error backwards through the net
        for layer in self.network:
            for i, node in enumerate(layer):
                node.update_weights(self.alpha, errors)

    def test(self,
            testset: "A tuple containing a list of target ouputs and input values"):
        '''Test the network against a testset'''
        n = len(testset)
        false_positives, false_negatives = 0, 0  
        false_negative_results = []
        for target, values in testset:
            if not self.test_sample(target, values):
                if target:
                    false_negatives += 1
                    false_negative_results.append(values)
                else:
                    false_positives += 1
        errors = false_negatives + false_positives
        accuracy = (n - errors) / n
        return {'size': n, 'false positive': false_positives, 'false negatives': false_negatives, 'errors': errors, 'accuracy': accuracy, 'values': false_negative_results}

    def test_sample(self, target, inputs):
        '''Test a single input/output pair'''
        return target == self.classify(inputs)

    def classify(self, inputs):
        '''Classify the given inputs'''
        for layer in range(len(self.network)):
            inputs = self.layer_classify(inputs, layer)
        return [v for k, v in inputs.items()]

    def layer_classify(self, inputs, layer):
        '''Put the given inputs through the specified layer'''
        new_inputs = {}
        for node in self.network[layer]:
            new_inputs[node.name] = node.output(inputs)
        return new_inputs

    def weights(self):
        '''Get the weights of all nodes per layer'''
        weights = []
        for layer in self.network:
            layer_weights = []
            for node in layer:
                layer_weights.append(node.weights)
            weights.append(layer_weights)
        return weights

    def __str__(self):
        string = ""
        for node in self.network[0]:
            string += '{}\n'.format(str(node))
        return string


class Node:
    def __init__(self, name, weights, fn, deriv, next_layer=[]):
        self.weights = weights
        self.fn = fn
        self.derivative = deriv
        self.inputs = {}
        self._delta = None
        self.wsum = 0
        self.next = []
        self.name = name

    def __str__(self):
        children = [node for node in self.next]
        string = '{}: {}'.format(self.name, str(self.weights))
        for child in children:
            string += '\n\t{}:{}\n'.format(self.name, str(child))
        return string

    def update_weights(self,
         alpha: "learning rate",
         error: "Backwards propogated error term"):
        for k, v in self.weights.items():
            self.weights[k] += alpha * self.inputs[k] * self.delta(error)

    def delta(self, errors):
        '''Get the error term of the node'''

        # If it's not been calculated this round
        if not self._delta:
            self._delta = 0

            # if this is an output node
            if len(self.next) == 0:
                self._delta = errors[self.name]
            else:
                # Calculate error using: error of all nodes * their weight of the edge to this node
                for n in self.next:
                    self._delta += n.delta(errors) * n.weights[self.name]
            self._delta *= self.derivative(self.weighted_sum())
        return self._delta

    def weighted_sum(self):
        '''Get the weighted sum of the inputs'''
        if not self.wsum:
            wsum = 0
            for k, v in self.inputs.items():
                wsum += self.weights[k] * v
            self.wsum = wsum
        return self.wsum

    def output(self, inputs):
        '''Determine the output given inputs'''

        # first reset old data
        self.reset()
        self.inputs = inputs
        self._output = self.fn(self.weighted_sum())
        return self._output

    def reset(self):
        '''Removes all input and calculated data'''
        self.inputs = None
        self._output = None
        self._delta = None
        self.wsum = None
